import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from utils import set_random_seed
import random
from datasets.custom_datasets import *

DATA_PATH = './data/'
IMAGENET_PATH = './data/ImageNet'

CIFAR10_SUPERCLASS = list(range(10))  # one class
IMAGENET_SUPERCLASS = list(range(30))  # one class
MNIST_SUPERCLASS = list(range(10))
SVHN_SUPERCLASS = list(range(10))
FashionMNIST_SUPERCLASS = list(range(10))
MVTECAD_SUPERCLASS = list(range(2))
DAGM_SUPERCLASS = list(range(2))
CITYSCAPE_SUPERCLASS = list(range(2))

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],
    [1, 33, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 29, 61],
    [0, 51, 53, 57, 83],
    [22, 25, 40, 86, 87],
    [5, 20, 26, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 38, 68, 76],
    [23, 34, 49, 60, 71],
    [15, 19, 21, 32, 39],
    [35, 63, 64, 66, 75],
    [27, 45, 77, 79, 99],
    [2, 11, 36, 46, 98],
    [28, 30, 44, 78, 93],
    [37, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]

def get_transform(image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_cityscape_globs():
    from glob import glob
    import random
    normal_path = glob('./cityscapes-5-10-threshold/cityscapes/ID/*')
    anomaly_path = glob('./cityscapes-5-10-threshold/cityscapes/OOD/*')

    random.seed(42)
    random.shuffle(normal_path)
    train_ratio = 0.7
    separator = int(train_ratio * len(normal_path))
    normal_path_train = normal_path[:separator]
    normal_path_test = normal_path[separator:]

    return normal_path_train, normal_path_test, anomaly_path

def get_gta_globs():
    from glob import glob
    nums = [f'0{i}' for i in range(1, 10)] + ['10']
    folder_paths = []
    globs_id = []
    globs_ood = []
    for i in range(10):
        id_path = f'./gta5-15-5-{nums[i]}/gta5_{i}/gta5_{i}/ID/*'
        ood_path = f'./gta5-15-5-{nums[i]}/gta5_{i}/gta5_{i}/OOD/*'
        globs_id.append(glob(id_path))
        globs_ood.append(glob(ood_path))
        print(i, len(globs_id[-1]), len(globs_ood[-1]))

    glob_id = []
    glob_ood = []
    for i in range(len(globs_id)):
        glob_id += globs_id[i]
        glob_ood += globs_ood[i]

    random.seed(42)
    random.shuffle(glob_id)
    train_ratio = 0.7
    separator = int(train_ratio * len(glob_id))
    glob_train_id = glob_id[:separator]
    glob_test_id = glob_id[separator:]

    return glob_train_id, glob_test_id, glob_ood

def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform


def get_dataset(P, dataset, test_only=False, image_size=None, download=False, eval=False):
    if dataset in ['imagenet']:
        if eval:
            train_transform, test_transform = get_simclr_eval_transform_imagenet(P.ood_samples,
                                                                                 P.resize_factor, P.resize_fix)
        else:
            train_transform, test_transform = get_transform_imagenet()
    else:
        train_transform, test_transform = get_transform(image_size=image_size)

    if dataset == 'cifar10':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)
    
    elif dataset == 'cityscape':
        image_size = (224, 224, 3)
        n_classes = 2
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        normal_path_train, normal_path_test, anomaly_path = get_cityscape_globs()
        test_path = normal_path_test + anomaly_path
        test_label = [0] * len(normal_path_test) + [1] * len(anomaly_path)
        train_label = [0] * len(normal_path_train)
        train_set = Cityscape(image_path=normal_path_train, labels=train_label,
                            transform=transform)
        test_set = Cityscape(image_path=test_path, labels=test_label,
                                transform=transform)

        print("train_set data shapes: ", train_set[0][0].shape)
        print("test_set data shapes: ", test_set[0][0].shape)
        
    elif dataset == 'dagm':
        image_size = (224, 224, 3)
        n_classes = 2
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        train_set = DAGM_DATASET(normal_class=P.one_class_idx, train=True, count=None, transform=transform)
        test_set = DAGM_DATASET(normal_class=P.one_class_idx, train=False, count=None, transform=transform)
        
        print("train_set data shapes: ", train_set[0][0].shape)
        print("test_set data shapes: ", test_set[0][0].shape)
        
    elif dataset == 'mvtecad':
        image_size = (224, 224, 3)
        n_classes = 2
        categories = ['toothbrush', 'zipper', 'transistor', 'tile', 'grid', 'wood', 'pill', 'bottle', 'capsule', 'metal_nut', 'hazelnut', 'screw', 'carpet', 'leather', 'cable']
        train_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        test_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
            ])
        root = "./mvtec_anomaly_detection"
        test_set = MVTecDataset(root=root, train=False, category=categories[P.one_class_idx], transform=test_transform, count=-1)
        train_set = MVTecDataset(root=root, train=True, category=categories[P.one_class_idx], transform=train_transform, count=-1)

        print("test_ds_mvtech shapes: ", test_set[0][0].shape)
        print("train_ds_mvtech_normal shapes: ", train_set[0][0].shape)
        
    elif dataset == 'fashion-mnist':
        image_size = (32, 32, 3)
        n_classes = 10
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        train_set = datasets.FashionMNIST(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.FashionMNIST(DATA_PATH, train=False, download=download, transform=test_transform)
    elif dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=download, transform=test_transform)
    elif dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        image_size = (32, 32, 1)
        n_classes = 10
        train_set = datasets.MNIST(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.MNIST(DATA_PATH, train=False, download=download, transform=test_transform)
    elif dataset == 'svhn-10':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=test_transform)
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)

    elif dataset == 'svhn':
        assert test_only and image_size is not None
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)

    elif dataset == 'imagenet':
        image_size = (224, 224, 3)
        n_classes = 30
        train_dir = os.path.join(IMAGENET_PATH, 'one_class_train')
        test_dir = os.path.join(IMAGENET_PATH, 'one_class_test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    else:
        raise NotImplementedError()

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset == 'svhn-10':
        return SVHN_SUPERCLASS
    if dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    if dataset == 'fashion-mnist':
        return FashionMNIST_SUPERCLASS
    elif dataset == 'mnist':
        return MNIST_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    elif dataset == 'mvtecad':
        return MVTECAD_SUPERCLASS
    elif dataset == 'dagm':
        return DAGM_SUPERCLASS
    elif dataset == 'cityscape':
        return CITYSCAPE_SUPERCLASS
    else:
        raise NotImplementedError()


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    try:
        for idx, tgt in enumerate(dataset.targets):
            if tgt in classes:
                indices.append(idx)
    except:
        # SVHN
        for idx, (_, tgt) in enumerate(dataset):
            if tgt in classes:
                indices.append(idx)
    dataset = Subset(dataset, indices)
    return dataset


def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):
    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    if resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=resize_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    clean_trasform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform


def set_dataset_count(dataset, count=-1):
    if count==-1:
        pass
    elif len(dataset)>count:
        unique_numbers = []
        while len(unique_numbers) < count:
            number = random.randint(0, len(dataset)-1)
            if number not in unique_numbers:
                unique_numbers.append(number)
        dataset = Subset(dataset, unique_numbers)
    else:
        num = int(count / len(dataset))
        remainding = (count - num*len(dataset))
        trnsets = [dataset for i in range(num)]
        unique_numbers = []
        while len(unique_numbers) < remainding:
            number = random.randint(0, len(dataset)-1)
            if number not in unique_numbers:
                unique_numbers.append(number)
        dataset = Subset(dataset, unique_numbers)
        trnsets = trnsets + [dataset]
        dataset = torch.utils.data.ConcatDataset(trnsets)

    return dataset