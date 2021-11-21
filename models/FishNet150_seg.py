# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:18:42 2021

@author: yipji
"""
import FishNet.models.net_factory as nf
import torch
import torch.nn as nn 
import numpy as np

class FishNet150_seg(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.net = nf.fishnet150()
        self.n_classes = n_classes
    
        checkpoint = torch.load("./FishNet/checkpoints/fishnet150_ckpt.tar")
        # best_prec1 = checkpoint['best_prec1']
        state_dict = checkpoint['state_dict']
    
        from collections import OrderedDict
        
        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        
        self.net.load_state_dict(new_state_dict)
        
        # self.net.fish.fish[9][0][2] = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        self.net.fish.fish[9][4][0] = nn.Conv2d(1056,1024,kernel_size=1)
        self.net.fish.fish[9][4][1] = nn.Flatten()
        self.bypass = nn.Unflatten(dim = 1, unflattened_size = (224,224))
    

        
    def forward(self, x):
        x = self.net(x)
        x = self.bypass(x)
        x = nn.Sigmoid()(x)
        return x

    

if __name__ == '__main__':
    from torchsummary import summary
    net = FishNet150_seg().cuda()
    summary(net, (3,224,224))