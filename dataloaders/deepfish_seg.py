import pandas as pd
import  numpy as np
import os
import torch
from PIL import Image

class DeepFishSeg:
    def __init__(self, split, transform=None,
     datadir="", n_samples=None, habitat=None):

        self.split = split
        self.n_classes = 2
        self.datadir = datadir
        self.transform = transform

        self.img_names, self.labels, self.mask_names = get_seg_data(self.datadir, 
        split, habitat=habitat)

        if n_samples:
           self.img_names = self.img_names[:n_samples] 
           self.mask_names = self.mask_names[:n_samples] 
           self.labels = self.labels[:n_samples]
        self.path = self.datadir #+ "/images/"


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]
        image_pil = Image.open(self.path + "/images/"+ name + ".jpg")
       
        image = self.transform(image_pil)

        mask_classes = Image.open(self.path + "/masks/"+ self.mask_names[index] + ".png").convert('L')
        
        mask_classes = torch.from_numpy(np.array(mask_classes)).float() / 255.
        batch = {"images": image,
                 "labels": self.labels[index],
                 "mask_classes": mask_classes,
                 "meta": {"index": index,
                          "image_id": index,
                          "split": self.split}}

        return batch

def get_seg_data(datadir, split,  habitat=None ):
    df = pd.read_csv(os.path.join(datadir,  '%s.csv' % split))
    df = slice_df_reg(df, habitat)
    img_names = np.array(df['ID'])
    mask_names = np.array(df['ID'])
    labels = np.array(df['labels'])
    return img_names, labels, mask_names

def slice_df(df, habitat):
    if habitat is None:
        return df
    return df[df['ID'].apply(lambda x: True if x.split("/")[0] == habitat else False)]

def slice_df_reg(df, habitat):
    if habitat is None:
        return df
    return df[df['ID'].apply(lambda x: True if x.split("/")[1].split("_")[0] 
                        == habitat else False)]

def get_dataset(dataset_name, split, transform=None, 
                datadir=None, habitat=None):
    
    if dataset_name == "fish_seg":
        dataset = DeepFishSeg(split,
                              transform=transform,
                              datadir=datadir+ "/Segmentation/", 
                              habitat=habitat)

    return dataset
    