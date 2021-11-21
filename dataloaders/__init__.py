from datasets import fish_clf, fish_reg, fish_loc
from torchvision import transforms
import cv2, os
import pandas as pd
import numpy as np

import pandas as pd
import  numpy as np
import os

def get_dataset(dataset_name, split, transform=None, 
                datadir=None, habitat=None):

    transform = get_transformer(transform, split)

    if dataset_name == "fish_clf":
        dataset = fish_clf.FishClf(split,
                               transform=transform,
                               datadir=datadir + "/Classification/", 
                               habitat=habitat)

    elif dataset_name == "fish_reg":
        dataset = fish_reg.FishReg(split,
                               transform=transform,
                              datadir=datadir + "/Localization/", 
                              habitat=habitat)

    elif dataset_name == "fish_loc":
        dataset = fish_loc.FishLoc(split,
                               transform=transform,
                               datadir=datadir+ "/Localization/", 
                               habitat=habitat)

    return dataset                        


def get_transformer(transform, split):
    if transform == "resize_normalize":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalize_transform = transforms.Normalize(mean=mean, std=std)

        return transforms.Compose(
            [transforms.Resize((224,224)),
             transforms.ToTensor(),
             normalize_transform])

    if transform == "rgb_normalize":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalize_transform = transforms.Normalize(mean=mean, std=std)

        return transforms.Compose(
            [
             transforms.ToTensor(),
             normalize_transform])