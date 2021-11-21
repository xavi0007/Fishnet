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

