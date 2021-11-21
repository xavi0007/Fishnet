import os




import torch
from torch.utils.data import DataLoader
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

def main(model_path):


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
    net.load_state_dict(torch.load(model_path))
    
    with torch.no_grad():
        test = predict_acc(net, test_loader)
    
    print(test)


if __name__ == "__main__":
    main(os.getcwd())