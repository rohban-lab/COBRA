import torch.optim
import models.transform_layers as TL
from training.contrastive_loss import get_similarity_matrix, NT_xent
from utils import AverageMeter, normalize
from adv_training.attack import RepresentationAdv
import time
import random
from datasets.pseudo_anomaly_generation import PseudoAnomalyGenerator

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)

def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None,
          simclr_aug=None):
    assert simclr_aug is not None
    assert P.sim_lambda == 1.0  # to avoid mistake
    assert P.K_classification > 1

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    pseudo_anomaly_gen = PseudoAnomalyGenerator(P)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['sim'] = AverageMeter()
    Rep = RepresentationAdv(model, epsilon=P.epsilon, alpha=P.alpha,
                            min_val=P.min, max_val=P.max, max_iters=P.k,
                            _type=P.attack_type, loss_type=P.loss_type,
                            regularize=P.regularize_to, criterion=criterion)

    check = time.time()
    print(f"number of batch={len(loader)}")
    print("P.K_classification", P.K_classification)
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples
        data_time.update(time.time() - check)
        check = time.time()
    
        
        ### SimCLR loss ###
        if P.dataset != 'imagenet':
            batch_size = images.size(0)
            images = images.to(device)
            images1, images2 = hflip(images.repeat(2, 1, 1, 1)).chunk(2)  # hflip
        else:
            batch_size = images[0].size(0)
            images1, images2 = images[0].to(device), images[1].to(device)
        labels = labels.to(device)
        
        images1 = torch.cat([images1.clone() if k==0 else pseudo_anomaly_gen.create_pseudo_anomaly(images1.clone(), rot_k=k) for k in range(P.K_classification)])
        images2 = torch.cat([images2.clone() if k==0 else pseudo_anomaly_gen.create_pseudo_anomaly(images2.clone(), rot_k=k) for k in range(P.K_classification)])

        images1, images2 = images1.to(device), images2.to(device)

        classification_labels = torch.cat([torch.ones_like(labels) * k for k in range(P.K_classification)], 0)  # B -> 4B

        images1 = simclr_aug(images1)
        images2 = simclr_aug(images2)

        # Adversarial Training on negative transformations (ATNT)
        chu = images2.chunk(P.K_classification)
        a = [i for i in range(P.K_classification)]
        for i in range(0, len(a) - 1):
            pick = random.randint(i + 1, len(a) - 1)
            a[i], a[pick] = a[pick], a[i]
        # For P.K_classification=4 --> neg_img = torch.cat((chu[a[0]], chu[a[1]], chu[a[2]], chu[a[3]]))
        neg_img = torch.cat([chu[a[i]].clone() for i in range(P.K_classification)])
    
        adv_atnt_img = Rep.get_adversarial_contrastive_img(original_images=images2, target=neg_img, optimizer=optimizer,
                                                      weight=P.lamda, random_start=P.random_start, reduce_loss=True)
        
        # robust transformation prediction (RTD)
        adv_rtp = Rep.get_adversarial_classification_img(original_images=images2, optimizer=optimizer,
                                                      weight=P.lamda, classification_labels=classification_labels,
                                                      random_start=P.random_start)

        # Adversarial training on positive transformations (ATPT)
        adv_atpt_img = Rep.get_adversarial_contrastive_img(original_images=images1, target=images2, optimizer=optimizer,
                                                      weight=P.lamda, random_start=P.random_start)

        images_pair = torch.cat([images1, images2, adv_atpt_img, adv_atnt_img], dim=0)  # 8B

        _, outputs_aux = model(images_pair, simclr=True, penultimate=False, classification=True)
        outputs_aux['simclr'] = normalize(outputs_aux['simclr'])  # normalize

        _, outputs_cls_adv = model(adv_rtp, simclr=False, penultimate=False, classification=True)

        sim_matrix = get_similarity_matrix(outputs_aux['simclr'], multi_gpu=P.multi_gpu)
        loss_cobra = NT_xent(sim_matrix, temperature=0.5, chunk=4) * P.sim_lambda
        loss_classification = criterion(
            torch.cat([outputs_aux['classification'][:int(outputs_aux['classification'].shape[0] // 2)], outputs_cls_adv['classification']]),
            classification_labels.repeat(3))
        
        ### total loss ###
        loss = loss_cobra + loss_classification

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']
        torch.cuda.empty_cache()

        batch_time.update(time.time() - check)

        losses['sim'].update(loss_cobra.item(), batch_size)
        losses['cls'].update(loss_classification.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossSim %f] [LossClassification %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                 losses['sim'].value, losses['cls'].value))

    log_('[DONE] [Time %.3f] [Data %.3f] [LossSim %f] [LossClassification %f]' %
         (batch_time.average, data_time.average,
          losses['sim'].average, losses['cls'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cobra', losses['sim'].average, epoch)
        logger.scalar_summary('train/LossClassification', losses['cls'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
