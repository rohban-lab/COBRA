import os
import numpy as np
import torch
import torch.nn.functional as F
import random
import models.transform_layers as TL
import json
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

from sklearn.mixture import GaussianMixture
from scipy.stats import percentileofscore
import gdown

def extract_embeddings(loader, model, device):
    embeddings = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten
            embeddings.append(outputs.cpu().numpy())

    return np.vstack(embeddings)
    

class PseudoAnomalyGenerator:
    def __init__(self, P, train_loader=None):
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        if train_loader is not None:
            self.feature_extractor = torchvision.models.resnet18(pretrained=False)
            model_path = './resnet18_classifier.pth'
            if not os.path.exists(model_path):
                url = 'https://drive.google.com/uc?id=15kLFakDxG88hKiARr66tewftHMAAZdVL'
                gdown.download(url, model_path, quiet=False)
            state_dict = torch.load(model_path, map_location=self.device)
            self.feature_extractor.load_state_dict(state_dict)
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1]).to(self.device)

            self.train_features = extract_embeddings(train_loader, self.feature_extractor, self.device)
            self.gmm = GaussianMixture(n_components=1, covariance_type='full')
            self.gmm.fit(self.train_features)
            self.train_scores = self.gmm.score_samples(self.train_features)

        with open('./datasets/config.json', 'r') as json_file:
            self.probabilities = json.load(json_file)
        try: 
            self.probabilities = self.probabilities[P.dataset]
        except:
            self.probabilities = self.probabilities["other-dataset"]

        self.auto_aug = transforms.Compose([
                transforms.ToPILImage(),
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        self.elastic_aug = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ElasticTransform(alpha=200.0, sigma=7.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

        self.rotation_shift = TL.Rotation()
        self.cutperm_shift = TL.CutPerm()
        self.cutpaste_shift = TL.CutPasteLayer()
        self.rotation_shift = self.rotation_shift.to(self.device)
        self.cutperm_shift = self.cutperm_shift.to(self.device)
        self.cutpaste_shift = self.cutpaste_shift.to(self.device)
        
        trans = transforms.Compose([
                transforms.Resize((P.image_size[0], P.image_size[1])),
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
        ])
        self.aug_to_func = {'elastic': self.apply_elastic, 'rotation': self.apply_rotation, 'cutperm': self.apply_cutperm, 'cutout': self.apply_cutout, 'cutpaste': self.apply_cutpaste}
        self.rot_k = 1

    def apply_rotation(self, img):
        # input:torch.rand(3, 224, 224)
        # output:torch.rand(3, 224, 224)
        img = self.rotation_shift(img.unsqueeze(0), self.rot_k)
        #  = {
        #     "1": 2,
        #     "2": 3,
        #     "3": 1
        # }[str(self.rot_k)]
        return img.squeeze().to(self.device)

    def apply_elastic(self, img):
        # input:torch.rand(3, 224, 224)
        # output:torch.rand(3, 224, 224)
        if self.rot_k==0:
            return img
        return self.elastic_aug(img).to(self.device)
        
    def apply_cutperm(self, img):
        if self.rot_k==0:
            return img
        # input:torch.rand(3, 224, 224)
        # output:torch.rand(3, 224, 224)
        return self.cutperm_shift(img.unsqueeze(0), np.random.randint(1, 4)).squeeze()
        
    def apply_cutout(self, image):
        # input:torch.rand(3, 224, 224)
        # output:torch.rand(3, 224, 224)
        if self.rot_k==0:
            return image
        _, h, w = image.shape
        mask_size = (np.random.randint(h // 6, h // 4), np.random.randint(w // 6, w // 4))
        mask_x = random.randint(0, h - mask_size[0])
        mask_y = random.randint(0, w - mask_size[1])
        mask = torch.ones_like(image)
        mask[:, mask_x:mask_x + mask_size[0], mask_y:mask_y + mask_size[1]] = 0.0
        
        # Apply mask to image
        return image * mask
    
    def apply_cutpaste(self, img):
        # input:torch.rand(3, 224, 224)
        # output:torch.rand(3, 224, 224)
        return self.cutpaste_shift(img.unsqueeze(0), self.rot_k).squeeze()

    def create_pseudo_anomaly(self, batch_image, rot_k=None, accept_reject_enabler=False):
        if rot_k is None:
            self.rot_k = random.choice([1, 2, 3])
        else:
            if rot_k==0:
                return batch_image
            self.rot_k = rot_k

        batch_image = batch_image.to(self.device)   
        augs = list(self.probabilities.keys())
        probs = list(self.probabilities.values())            
        batch_transforms = np.random.choice(augs, size=batch_image.shape[0], p=probs)

        if accept_reject_enabler:
            neg_pair = []
            for i, img in enumerate(batch_image):
                cnt = 0
                while True:
                    img_ =self.aug_to_func[batch_transforms[i]](img.clone())
                    output = self.feature_extractor(img_.unsqueeze(0))
                    output = output.view(output.size(0), -1).detach().cpu().numpy()
                    score = self.gmm.score_samples(output)
                    p_value = percentileofscore(self.train_scores, score, kind='weak') / 100.0
                    anomalies = p_value < 0.99
                    if anomalies[0] or (cnt > 3):
                        print(cnt)
                        break
                    cnt+=1
                neg_pair.append(img_)
            neg_pair = torch.stack(neg_pair)
        else:
            neg_pair = torch.stack([self.aug_to_func[batch_transforms[i]](img.clone()) for i, img in enumerate(batch_image)])
        return neg_pair