import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arguments import parse_args
import models.classifier as C
from datasets import get_dataset, get_superclass_list, get_subclass_dataset
from copy import deepcopy
from evaluation.eval import evaluate


def main():
    P = parse_args()
    P.resize_factor = 0.54
    P.resize_fix = True
    ### Set torch device ###
    P.n_gpus = torch.cuda.device_count()
    P.multi_gpu = False
    if torch.cuda.is_available():
        torch.cuda.set_device(P.local_rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    ### Initialize dataset ###
    if P.dataset == 'imagenet':
        P.batch_size = 1
        P.test_batch_size = 1
    train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset, download=True, eval=True)
    P.image_size = image_size
    P.n_classes = n_classes

    if P.one_class_idx is not None:
        cls_list = get_superclass_list(P.dataset)
        P.n_superclasses = len(cls_list)
        full_test_set = deepcopy(test_set)  # test set of full classes
        if (P.dataset=='mvtecad') or (P.dataset=='dagm') or (P.dataset=="cityscape"):
            test_set = get_subclass_dataset(test_set, classes=cls_list[0])
        else:
            train_set = get_subclass_dataset(train_set, classes=cls_list[P.one_class_idx])
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

    if (P.dataset=='mvtecad') or (P.dataset=='dagm') or (P.dataset=="cityscape"):
        P.ood_dataset = [1]
    else:
        P.K_classification = 4 

    ood_test_loader = dict()
    for ood in P.ood_dataset:
        if P.one_class_idx is not None:
            ood_test_set = get_subclass_dataset(full_test_set, classes=cls_list[ood])
            ood = f'one_class_{ood}'
        else:
            ood_test_set = get_dataset(P, dataset=ood, test_only=True, image_size=P.image_size, download=True, eval=True)
        ood_test_loader[ood] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

    ### Initialize model ###
    simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
    

    model = C.get_classifier(P.model, n_classes=P.n_classes).to(device)
    model = C.get_shift_classifer(model, P.K_classification).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    if P.load_path is not None:
        checkpoint = torch.load(P.load_path)
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    print(P)

    P.desired_attack = "PGD"
    P.PGD_constant = 2.5
    P.alpha = (P.PGD_constant * P.eps) / P.steps
    print("Attack targets: ")
    if P.in_attack:
        print("- Normal")
    if P.out_attack:
        print("- Anomaly")

    if P.out_attack or P.in_attack:
        print("Desired Attack:", P.desired_attack)
        print("Epsilon:", P.eps)
        if P.desired_attack == 'PGD':
            print("Steps:", P.steps)

    auroc_dict = evaluate(P, model, test_loader, ood_test_loader, P.ood_score,
                                    train_loader=train_loader, simclr_aug=simclr_aug)

    if P.one_class_idx is not None:
        mean_dict = dict()
        for ood_score in P.ood_score:
            mean = 0
            for ood in auroc_dict.keys():
                mean += auroc_dict[ood][ood_score]
            mean_dict[ood_score] = mean / len(auroc_dict.keys())
        auroc_dict['one_class_mean'] = mean_dict

    bests = []
    for ood in auroc_dict.keys():
        message = ''
        best_auroc = 0
        for ood_score, auroc in auroc_dict[ood].items():
            message += '[%s %s %.4f] ' % (ood, ood_score, auroc)
            if auroc > best_auroc:
                best_auroc = auroc
        message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
        if P.print_score:
            print(message)
        bests.append(best_auroc)

    bests = map('{:.4f}'.format, bests)


if __name__ == '__main__':
    main()
