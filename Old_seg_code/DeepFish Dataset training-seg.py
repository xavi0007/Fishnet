# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:05:13 2021

@author: yipji
"""




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

# datadir = r'C:/Users/yipji/Offline Documents/Big Datasets/DeepFish/DeepFish'

# train_dataset = get_dataset("tiny_fish_clf", "train", transform = "resize_normalize", datadir = datadir) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir',
                        type=str, default='C:/Users/yipji/Offline Documents/Big Datasets/DeepFish/DeepFish/')
    parser.add_argument("-e", "--exp_config", default='seg')
    parser.add_argument("-uc", "--use_cuda", type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda else 'cpu')

    exp_dict = exp_configs.EXP_GROUPS[args.exp_config][0]


    # Dataset
    # Load val set and train set
    val_set = get_dataset(dataset_name=exp_dict["dataset"], split="val",
                                   transform=exp_dict.get("transform"),
                                   datadir=args.datadir)
    train_set = get_dataset(dataset_name=exp_dict["dataset"],
                                     split="train", 
                                     transform=exp_dict.get("transform"),
                                     datadir=args.datadir)
    
    # Load train loader, val loader, and vis loader
    train_loader = DataLoader(train_set, 
                            sampler=RandomSampler(train_set,
                            replacement=True, num_samples=max(min(500, 
                                                            len(train_set)), 
                                                            len(val_set))),
                            batch_size=exp_dict["batch_size"])

    val_loader = DataLoader(val_set, shuffle=False, batch_size=exp_dict["batch_size"])
    vis_loader = DataLoader(val_set, sampler=ut.SubsetSampler(train_set,
                                                     indices=[0, 1, 2]),
                            batch_size=1)

    # #visualize mask
    # plt.imshow(transforms.ToPILImage()(next(iter(val_loader))['mask_classes']))
    
    # Create model, opt, wrapper
    # model_original = models.get_model(exp_dict["model"], exp_dict=exp_dict).cuda()
    
    ##FISHNET MOEL##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FishNet150_seg()
    net.to(device)
    
    
    #Setting up the Run#
    opt = torch.optim.Adam(net.parameters(), 
                        lr=1e-3, weight_decay=0.0001)
    model_original = models.get_model(exp_dict["model"], exp_dict=exp_dict).cuda()
    model = wrappers.get_wrapper(exp_dict["wrapper"], model=model_original, opt=opt).to(device)
    # model = wrappers.get_wrapper(exp_dict["wrapper"], model=net, opt=opt).to(device)
    
    score_list_v = []
    score_list_t = []
    
    for epoch in range(exp_dict["max_epoch"]):
        score_dict_v = {"epoch": epoch}
        score_dict_t = {"epoch": epoch}
        # visualize
        # model.vis_on_loader(vis_loader, savedir="./visualize")
        
        # train
        score_dict_t.update(model.train_on_loader(train_loader))
    
        # validate
        score_dict_v.update(model.val_on_loader(val_loader))
    
        # Add score_dict to score_list
        score_list_v += [score_dict_v]
        score_list_t += [score_dict_t]
    
        #Report
        print(score_dict_t)
        # Report and save
        # print(pd.DataFrame(score_list_v).tail())
        # print(pd.DataFrame(score_list_t).tail())
        # hu.save_pkl(score_list_path, score_list)
        # hu.torch_save(model_path, model.state_dict())
        # hu.torch_save(opt_path, opt.state_dict())
        # print("Saved in %s" % savedir)
        
    print(pd.DataFrame(score_list_t).tail())        
    # if args.exp_config == 'loc':
    #     batch = torch.utils.data.dataloader.default_collate([train_set[3]])
    # else:
    #     batch = torch.utils.data.dataloader.default_collate([train_set[0]])

    # for e in range(100):
    #     score_dict = model.train_on_batch(batch)
    #     print(e, score_dict)

        # model.vis_on_batch(batch, f'single_image_{args.exp_config}.png')
        #hu.save_image(fname, model.vis_on_batch(batch, view_support=True)[0])

        # validate on batch
        # val_dict = model.val_on_batch(batch)
        # pprint.pprint(val_dict)
        
#%%
import matplotlib.pyplot as plt

plt.figure()
f, ax = plt.subplots(3,1, figsize=(10,10))

ax[0].imshow(model.predict_on_batch(next(iter(val_loader))).squeeze()[0])      
ax[1].imshow(next(iter(train_loader))['images'][0].squeeze().permute(1,2,0))
ax[2].imshow(next(iter(train_loader))['mask_classes'][0].squeeze())  