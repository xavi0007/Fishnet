# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 00:49:06 2021

@author: yipji
"""

import os
import argparse
from haven import haven_utils as hu
import models.network_factory as nf
import pprint
import torch
import torch.nn as nn
from utils import exp_configs
import utils as ut
import models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision import transforms


from dataloaders import get_dataset


from models.FishNet150_count import FishNet150_count
from models.FishNet150_cls import FishNet150_cls
from models.FishNet201_cls import FishNet201_cls

import pandas as pd

import time

import torch.optim as optim
import torch.optim.lr_scheduler as lr

from utils.fishy_utils import predict_acc, BCEDiceLoss, CrossEntropyLoss2d, MultiClass_FocalLoss

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict_acc(net,loader):
    
    total_acc = 0
    total_batch = 0
    for i in loader:                
        total_acc += sum((net(i['images'].to(device)).squeeze()>0.5) == i['labels'].to(device))
        total_batch += len(i['labels'])
    
    final_acc = total_acc/total_batch
    
    return final_acc

if __name__ == "__main__":

    
    # Dataset
    # Load val set and train set

    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'data/DeepFish')

    test_set = get_dataset(dataset_name='fish_clf', split="test",
                                   transform="resize_normalize",
                                   datadir=data_dir)

    
    # Load train loader, val loader, and vis loader

    test_loader = DataLoader(test_set, shuffle=False, batch_size=64)



    ##FISHNET MOEL##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FishNet150_cls().to(device)
    net.load_state_dict(torch.load('./models/Run6_fishnet150-clf_25ep_state.pth'))
    
    with torch.no_grad():
        test = predict_acc(net, test_loader)
    
    print(test)