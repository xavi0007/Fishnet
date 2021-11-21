# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 09:58:20 2021

@author: yipji
"""

import matplotlib.pyplot as plt
from random import randint

from torch.utils.data import Dataset

import torch
import torch.nn.functional as F

import pandas as pd
from torchvision.io import read_image

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

# img_lib = list(Path(projdir).glob('img/*'))
# lab_lib = list(Path(projdir).glob('split/*'))

class FashionDataset(Dataset):
    '''
    annotatatios accpepts a filepath to the .txt file provided by the dataset
    img_dir accepts a filepath to the directory containing all the images
    img_list accepts a filebath to the .txt file provided by the dataset, 
    img_list should contain the list of filenames of images to be selected from the img_dir
    '''
    def __init__(self, img_dir, img_list, annotations, one_hot = False, transform=None, target_transform=None):

        self.img_labels = pd.read_csv(annotations, 
                                      sep=' ',
                                      header=None, 
                                      # names = ['cat1','cat2','cat3','cat4','cat5','cat6'],
                                      dtype='float')
        self.img_dir = img_dir
        self.img_list = open(img_list,'r').readlines()
        self.transform = transform
        self.target_transform = target_transform
        self.one_hot = one_hot
        # self.cat_sel = str(cats)
        
        #this code selects the necessary images from img_dir using img_list
        img_sel = []
        for i in range(len(self.img_list)):
            img_sel.append(str(self.img_dir)+self.img_list[i].strip('\n').replace('img/','\\'))
        self.img_sel=img_sel
        
        #this code pre-processes the data levels into an individual multi-classification problem
        # self.lab_ten = torch.as_tensor(self.img_labels[self.cat_sel].cat.codes)
        # F.one_hot(x,num_classes=len(img_lab['cat1'].cat.categories))
        # self.lab_sel = F.one_hot(self.lab_ten.long(), 
        #                          num_classes = len(self.img_labels[self.cat_sel].cat.categories))
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_sel[idx]
        image = read_image(str(img_path))
        label = torch.Tensor(self.img_labels.iloc[idx,:])
        

        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.one_hot:
            label = F.one_hot(label.long(),num_classes=7)    
        
        return image, label
    
    def showimage(self, index=0, random = True):
        if random:
            index = randint(0,len(self.img_labels))
        return plt.imshow(self[index][0].permute(1,2,0).numpy())
         
datadir = r'C:\Users\yipji\Offline Documents\Big Datasets'
projdir = datadir+r'\CE7454 Fashion Dataset\FashionDataset'
   
img_dir = Path(projdir+r'\img')

x_train = Path(projdir+r'\split\train.txt')
y_train = Path(projdir+r'\split\train_attr.txt')

x_test = Path(projdir+r'\split\test.txt')


tsfms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),    
    ])

fashy = FashionDataset(img_dir, x_train, y_train, one_hot=True,transform=tsfms)

train_dataloader = DataLoader(fashy,batch_size=4,shuffle=False)


    




    
    
