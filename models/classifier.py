import torch
import torch.nn as nn
from torchvision import transforms

from models.resnet import ResNet18, ResNet34, ResNet50
from models.resnet_imagenet import resnet18, resnet50
import models.transform_layers as TL
from datasets.cutpast_transform import CutPasteUnion, CutPasteNormal, CutPasteScar, High_CutPasteUnion

def get_simclr_augmentation(P, image_size):
    # parameter for resizecrop
    resize_scale = (P.resize_factor, 1.0)  # resize scaling factor
    if P.resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (P.resize_factor, P.resize_factor)

    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)

    # Transform define #
    if P.dataset=='imagenet' or P.dataset=='mvtecad' or P.dataset=='dagm':  # Using RandomResizedCrop at PIL transform
    # if P.dataset == 'imagenet':  # Using RandomResizedCrop at PIL transform
        transform = nn.Sequential(
            color_jitter,
            color_gray,
        )
    else:
        transform = nn.Sequential(
            color_jitter,
            color_gray,
            resize_crop,
        )

    return transform


def get_shift_classifer(model, K_classification):
    model.shift_cls_layer = nn.Linear(model.last_dim, K_classification)
    return model


def get_classifier(mode, n_classes=10):
    if mode == 'resnet18':
        classifier = ResNet18(num_classes=n_classes)
    elif mode == 'resnet34':
        classifier = ResNet34(num_classes=n_classes)
    elif mode == 'resnet50':
        classifier = ResNet50(num_classes=n_classes)
    elif mode == 'resnet18_imagenet':
        classifier = resnet18(num_classes=n_classes)
    elif mode == 'resnet50_imagenet':
        classifier = resnet50(num_classes=n_classes)
    else:
        raise NotImplementedError()

    return classifier
    
