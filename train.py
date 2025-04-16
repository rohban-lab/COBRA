import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from arguments import parse_args
import models.classifier as C
from datasets import get_dataset, get_superclass_list, get_subclass_dataset, set_dataset_count
from utils import load_checkpoint

from utils import Logger
from utils import save_checkpoint
from training.COBRA import train
from copy import deepcopy
import subprocess
import time

def evaluate_model(adv, P, eval_test_batch_size, eval_batch_size, logger):
    arguments_to_pass = [
        "--batch_size", str(eval_batch_size),
        "--test_batch_size", str(eval_test_batch_size),
        "--dataset", str(P.dataset),
        "--model", str(P.model),
        "--ood_score", "COBRA",
        "--print_score",
        "--one_class_idx" , str(P.one_class_idx),
        "--load_path", str(P.load_path),
        '--eps', str(P.epsilon)
    ]
    if adv:
        logger.log("Adversarialy evaluating:")
        arguments_to_pass.append("--out_attack")
        arguments_to_pass.append("--in_attack")
    else:
        logger.log("Clean evaluating:")

    result = subprocess.run(["python", "eval.py"] + arguments_to_pass, capture_output=True, text=True)

    if result.returncode == 0:
        logger.log("Script executed successfully.")
        logger.log("Output:")
        logger.log(result.stdout)
    else:
        logger.log("Script execution failed.")
        logger.log("Error:")
        logger.log(result.stderr)


def main():
    start_time = time.time()

    P = parse_args()
    ### Set torch device ###
    if torch.cuda.is_available():
        torch.cuda.set_device(P.local_rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    P.n_gpus = torch.cuda.device_count()
    P.multi_gpu = False

    ### only use one ood_layer while training
    P.ood_layer = P.ood_layer[0]
    ### Initialize dataset ###
    train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset, download=True)
    P.image_size = image_size
    P.n_classes = n_classes

    if P.one_class_idx is not None:
        cls_list = get_superclass_list(P.dataset)
        P.n_superclasses = len(cls_list)
        full_test_set = deepcopy(test_set)  # test set of full classes
        if P.dataset == 'mvtecad':
            train_set = set_dataset_count(train_set, count=10000)
            test_set = get_subclass_dataset(test_set, classes=cls_list[0])
        elif (P.dataset=='dagm') or (P.dataset=="cityscape"):
            train_set = set_dataset_count(train_set, count=5000)
            test_set = get_subclass_dataset(test_set, classes=cls_list[0])
        else:
            train_set = get_subclass_dataset(train_set, classes=cls_list[P.one_class_idx]) # this line just in train (no eval)
            test_set = get_subclass_dataset(test_set, classes=cls_list[P.one_class_idx])
        print('number of test data:', len(test_set))
        print('number of test data:', len(train_set))

    kwargs = {'pin_memory': False, 'num_workers': 4}
    train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

    if (P.ood_dataset is None) and (P.dataset!='mvtecad') and (P.dataset!='dagm') and (P.dataset!="cityscape"):
        if P.one_class_idx is not None:
            P.ood_dataset = list(range(P.n_superclasses))
            P.ood_dataset.pop(P.one_class_idx)
        elif P.dataset == 'cifar10':
            P.ood_dataset = ['svhn', 'cifar100', 'mnist', 'imagenet', "fashion-mnist"]

    if (P.dataset == 'mvtecad') or (P.dataset=='dagm') or (P.dataset=="cityscape"):
        P.ood_dataset = [1]
    else:
        P.K_classification = 4
    ood_test_loader = dict()
    for ood in P.ood_dataset:
        if P.one_class_idx is not None:
            ood_test_set = get_subclass_dataset(full_test_set, classes=cls_list[ood])
            ood = f'one_class_{ood}'  # change save name
        else:
            ood_test_set = get_dataset(P, dataset=ood, test_only=True, image_size=P.image_size, download=True)
        ood_test_loader[ood] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
        print(f"number of data of class {ood}:", len(ood_test_set))

    ### Initialize model ###

    simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
    
    model = C.get_classifier(P.model, n_classes=P.n_classes).to(device)
    model = C.get_shift_classifer(model, P.K_classification).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    if P.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
        lr_decay_gamma = 0.1
    elif P.optimizer == 'lars':
        try:
            from torchlars import LARS
            base_optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
            optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
        except:
            print("Warning: Due to LARs not install on this device we use SGD optimizer!")
            optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
        lr_decay_gamma = 0.1
    else:
        raise NotImplementedError()

    if P.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, P.epochs)
    elif P.lr_scheduler == 'step_decay':
        milestones = [int(0.5 * P.epochs), int(0.75 * P.epochs)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
    else:
        raise NotImplementedError()

    from training.scheduler import GradualWarmupScheduler
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10.0, total_epoch=P.warmup,
                                              after_scheduler=scheduler)

    if P.resume_path is not None:
        resume = True
        model_state, optim_state, config = load_checkpoint(P.resume_path, mode='last')
        model.load_state_dict(model_state, strict=not P.no_strict)
        optimizer.load_state_dict(optim_state)
        start_epoch = config['epoch']
        best = 100.0
        error = 100.0
    else:
        resume = False
        start_epoch = 1
        best = 100.0
        error = 100.0

    fname = f'{P.dataset}_{P.model}'
    if P.one_class_idx is not None:
        fname += f'_one_class_{P.one_class_idx}'
    if P.suffix is not None:
        fname += f'_{P.suffix}'

    P.PGD_constant = 2.5
    P.alpha = (P.PGD_constant * P.epsilon) / P.k
    P.ood_layer = ('simclr', 'classification')

    logger = Logger(fname, ask=not resume, local_rank=P.local_rank)
    logger.log(P)
    logger.log(model)

    
    if (P.dataset=='mvtecad') or (P.dataset=='dagm') or (P.dataset=="cityscape"):
        eval_batch_size = 10
        eval_test_batch_size = 10
    else:
        eval_batch_size = P.batch_size // 2
        eval_test_batch_size = P.test_batch_size // 2
    
    
    print(f"start_epoch={start_epoch}, P.epochs={P.epochs}")
    epoch = 0
    for epoch in range(start_epoch, P.epochs + 1):
        if P.timer < (time.time() - start_time):
            break
        logger.log_dirname(f"Epoch {epoch}")
        model.train()
        kwargs = {}
        kwargs['simclr_aug'] = simclr_aug

        train(P, epoch, model, criterion, optimizer, scheduler_warmup, train_loader, logger=logger, **kwargs)

        model.eval()
        save_states = model.state_dict()
        save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)

        if (epoch % P.evaluate_save_step == 0):
            torch.cuda.empty_cache()
            P.load_path = './'+logger.logdir + '/last.model'
            if P.train_time_adv_evaluate:
                evaluate_model(adv=True, P=P, eval_test_batch_size=eval_test_batch_size, eval_batch_size=eval_batch_size, logger=logger)
            if P.train_time_clean_evaluate:
                evaluate_model(adv=False, P=P, eval_test_batch_size=eval_test_batch_size, eval_batch_size=eval_batch_size, logger=logger)
        
    epoch += 1
    save_states = model.state_dict()
    save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)


if __name__ == '__main__':
    main()
