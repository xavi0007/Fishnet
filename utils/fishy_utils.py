# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:52:09 2021

@author: yipji
"""

import torch
import torch.nn as nn

# pred = net(next(iter(vis_loader))['images'][:1].cuda())
# labs = next(iter(vis_loader))['mask_classes'].cuda()

def f_score(pr, gt, beta=1, eps=1e-7, threshold=0.5, activation='none'):

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()


    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score

def predict_acc(model, dataloader):
    acc = []
    samps = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, data in enumerate(dataloader):
        #evaluate the model on the test set
        inputs = data['images']
        labels = data['mask_classes']
        inputs, labels = inputs.to(device), labels.to(device)    
        
        if not len(inputs)==len(labels):
            raise print("target and label size mismatch")
        
        
        for j in inputs:
            for k in labels:
                acc.append(f_score(j,k))
        
        samps += len(inputs)

    return sum(acc)/samps


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1., 
                           eps=self.eps, threshold=None, 
                           activation=self.activation)


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid', lambda_dice=1.0, lambda_bce=1.0):
        super().__init__(eps, activation)
        if activation == None:
            self.bce = nn.BCELoss(reduction='mean')
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.lambda_dice=lambda_dice
        self.lambda_bce=lambda_bce

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return (self.lambda_dice*dice) + (self.lambda_bce* bce)    


import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()

class MultiClass_FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(MultiClass_FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        #weight is alpha param to counter balance class weights
        self.weight = weight

    def forward(self, input, target):
        #CE instead of BCE because multi class
        cross_entropy_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        loss = torch.exp(-cross_entropy_loss)
        focal_loss = ((1 - loss) ** self.gamma * cross_entropy_loss).mean()
        return focal_loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss    
    