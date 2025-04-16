import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from glob import glob



class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)


class MVTecDataset(Dataset):
    def __init__(self, root, category, transform=None, train=True, count=-1):
        self.transform = transform
        self.image_files = []
        print("category MVTecDataset:", category)
        if train:
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files
        if count != -1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count-t):
                    self.image_files.append(random.choice(self.image_files[:t]))
        self.image_files.sort(key=lambda y: y.lower())
        self.train = train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1
        return image, target
        
    def __len__(self):
        return len(self.image_files)

class DAGM_DATASET(Dataset):
    def __init__(self, normal_class, train=True, count=None, transform=None):
        self.transform = transform
        if train:
            self.image_files = glob(f"./DAGM_Dataset/{normal_class}/train/normal/*.PNG")
            self.label = [0]*len(self.image_files) 
        else:
            image_files_normal = glob(f"./DAGM_Dataset/{normal_class}/test/normal/*.PNG")
            image_files_abnormal = glob(f"./DAGM_Dataset/{normal_class}/test/abnormal/*.PNG")
            self.image_files = image_files_normal+image_files_abnormal
            self.label = ([0]*len(image_files_normal)) + ([1]*len(image_files_abnormal))
        if train:
          if count is not None:
              if count<len(self.image_files):
                  self.image_files = self.image_files[:count]
              else:
                  t = len(self.image_files)
                  for i in range(count-t):
                      self.image_files.append(random.choice(self.image_files[:t]))
              self.label = [0]*len(self.image_files) 

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.label[index]

    def __len__(self):
        return len(self.image_files)



class Cityscape(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count-t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)