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

import wrappers

from models.FishNet150_count import FishNet150_count

import pandas as pd

import time
import os
import torch.optim as optim
import torch.optim.lr_scheduler as lr

from utils.fishy_utils import predict_acc, BCEDiceLoss, CrossEntropyLoss2d, MultiClass_FocalLoss

import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict_mae(net,loader):
    
    total_mae = 0
    total_batch = 0
    for i in loader:
        # images = i['images'].to(device)
        # labels = i['counts'].to(device)
        # prediction = net(images).round().squeeze()
        mae = torch.mean(abs(i['counts'].to(device) - net(i['images'].to(device)).round().squeeze()))
        total_mae += mae
        total_batch += 1
    
    final_mae = total_mae/total_batch
    
    return final_mae

def main(model_path):

    
    # Dataset
    # Load val set and train set
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'data')
    test_set = get_dataset(dataset_name="fish_reg", split="test",
                                   transform="resize_normalize",
                                   datadir=data_dir)

    
    # Load train loader, val loader, and vis loader

    test_loader = DataLoader(test_set, shuffle=False, batch_size=32)


    ##FISHNET MOEL##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FishNet150_count().to(device)
    net.load_state_dict(torch.load(model_path))
    
    with torch.no_grad():
        test = predict_mae(net, test_loader)
    
    print(test)

if __name__ == "__main__":
    main()

   