# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:18:42 2021

@author: yipji
"""
import FishNet.models.net_factory as nf
import torch
import torch.nn as nn


class FishNet150_fashion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nf.fishnet150()
    
        checkpoint = torch.load("./FishNet/checkpoints/fishnet150_ckpt.tar")
        # best_prec1 = checkpoint['best_prec1']
        state_dict = checkpoint['state_dict']
    
        from collections import OrderedDict
        
        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        
        self.net.load_state_dict(new_state_dict)
        
        self.net.fish.fish[9][4][1] = nn.Sequential(nn.Flatten(), 
                                                    nn.Linear(in_features=1056, out_features= 42, bias=True),
                                                    # nn.Unflatten(dim = 1, unflattened_size=(6,7,1)),
                                                    # nn.Softmax(dim=2),
                                                    ) 
        self.fc = nn.Sequential(nn.Unflatten(dim = 1, unflattened_size=(6,7,1)),
                           nn.Softmax(dim=2),
                           )
    def forward(self, x):
        fish = self.net(x)
        fc = self.fc(fish)
        
        return fc

if __name__ == '__main__':
    from torchsummary import summary
    net = FishNet150_fashion().cuda()
    summary(net, (3,224,224))