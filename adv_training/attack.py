"""
this code is modified from

https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks
https://github.com/louis2889184/pytorch-adversarial-training
https://github.com/MadryLab/robustness
https://github.com/yaodongyu/TRADES

"""

import torch
import torch.nn.functional as F
from adv_training.loss import pairwise_similarity
from adv_training.loss import NT_xent
from utils import AverageMeter, normalize
import copy


def project(x, original_x, epsilon, _type='linf'):
    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)
    else:
        raise NotImplementedError

    return x


class RepresentationAdv():
    def __init__(self, model, epsilon, alpha, min_val, max_val, max_iters, _type='linf', loss_type='sim',
                 regularize='original', criterion=None):

        # Model
        self.model = model
        # self.projector = projector
        self.regularize = regularize
        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type
        # loss type
        self.loss_type = loss_type
        self.criterion = criterion

    def get_adversarial_contrastive_img(self, original_images, target, optimizer, weight, random_start=True,
                                        reduce_loss=False):
        # get PAT or NAT adversarial image
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.float().cuda()
            x = original_images.float().clone() + rand_perturb
            x = torch.clamp(x, self.min_val, self.max_val)
        else:
            x = original_images.clone()
        x.requires_grad = True
        self.model.eval()
        with torch.enable_grad():
            for _iter in range(self.max_iters):
                self.model.zero_grad()
                inputs = torch.cat((x, target))
                _, outputs_aux = self.model(inputs, simclr=True, penultimate=False, classification=False)
                outputs_aux['simclr'] = normalize(outputs_aux['simclr'])  # normalize
                similarity, _ = pairwise_similarity(outputs_aux['simclr'], temperature=0.5, multi_gpu=False,
                                                    adv_type='None')
                loss = NT_xent(similarity, 'None')
                grads = torch.autograd.grad(loss, x, grad_outputs=None, only_inputs=True, retain_graph=False)[0]
                if self._type == 'linf':
                    scaled_g = torch.sign(grads.data)
                if reduce_loss:
                    x.data -= self.alpha * scaled_g
                else:
                    x.data += self.alpha * scaled_g
                x = torch.clamp(x, self.min_val, self.max_val)
                x = project(x, original_images, self.epsilon, self._type)
        self.model.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        return x.detach()

    def get_adversarial_classification_img(self, original_images, optimizer, weight, classification_labels, random_start=True,
                                        reduce_loss=False):
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.float().cuda()
            x = original_images.float().clone() + rand_perturb
            x = torch.clamp(x, self.min_val, self.max_val)
        else:
            x = original_images.clone()
        x.requires_grad = True
        self.model.eval()
        with torch.enable_grad():
            for _iter in range(self.max_iters):
                self.model.zero_grad()
                _, outputs_aux = self.model(x, simclr=False, penultimate=False, classification=True)
                loss = self.criterion(outputs_aux['classification'], classification_labels)
                grads = torch.autograd.grad(loss, x, grad_outputs=None, only_inputs=True, retain_graph=False)[0]
                if self._type == 'linf':
                    scaled_g = torch.sign(grads.data)
                if reduce_loss:
                    x.data -= self.alpha * scaled_g
                else:
                    x.data += self.alpha * scaled_g

                x = torch.clamp(x, self.min_val, self.max_val)
                x = project(x, original_images, self.epsilon, self._type)

        self.model.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        return x.detach()
