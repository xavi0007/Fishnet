
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
import torchvision.models

def resnet50():
    net = torchvision.models.resnet50(pretrained= True)
    for param in net.parameters():
        param.requires_grad = False
    for param in net.conv1.parameters():
        param.requires_grad = True
    for param in net.bn1.parameters():
        param.requires_grad = True 
    for param in net.layer1.parameters():
        param.requires_grad = True 
    for param in net.layer2.parameters():
        param.requires_grad = True 
    for param in net.layer4.parameters():
        param.requires_grad = True
    net.fc = nn.Sequential(
                        nn.Linear(in_features=2048, out_features=1, bias=True),
                        # nn.Linear(in_features=768, out_features=512, bias=True),
                        # nn.Linear(in_features=512, out_features=1, bias=True),
                        nn.ReLU(),
                        )    
    return net
    
    
def predict_acc(net,loader):
    
    total_acc = 0
    total_batch = 0
    for i in loader:                
        total_acc += sum((net(i['images'].to(device)).squeeze()>0.5) == i['labels'].to(device))
        total_batch += len(i['labels'])
    
    final_acc = total_acc/total_batch
    
    return final_acc


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir',
                        type=str, default='C:/Users/yipji/Offline Documents/Big Datasets/DeepFish/DeepFish/')
    parser.add_argument("-e", "--exp_config", default='clf')
    parser.add_argument("-uc", "--use_cuda", type=int, default=0)
    args = parser.parse_args()


    exp_dict = exp_configs.EXP_GROUPS[args.exp_config][0]
    
    # Dataset
    test_set = get_dataset(dataset_name=exp_dict["dataset"], split="test",
                                   transform="resize_normalize",
                                   datadir=args.datadir)

    
    # Load train loader, val loader, and vis loader
   
    test_loader = DataLoader(test_set, shuffle=False, batch_size=exp_dict["batch_size"])


    ##FISHNET MOEL##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet50()
    net.to(device)
    net.load_state_dict(torch.load('./models/Run7_resnet50-clf_25ep_state.pth'))
        
    with torch.no_grad():
        test = predict_acc(net,test_loader)
        
    print(test)
