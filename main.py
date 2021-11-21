import argparse
import time
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.data_aug import ColorAugmentation
import os
from torch.autograd.variable import Variable
import models

parser = argparse.ArgumentParser(description='CE7454 assignment fishnet')


#class, seg, count, ablation
parser.add_argument('--task', '-t', default='ablation')

parser.add_argument('--path', default=os.get_cwd)

def main():
    args = parser.parse_args()
    if args.task == 'segmentation':
        import semantic_segmentation as seg
        seg.main()
    if args.task == 'test_seg':
        import semantic_segmentation as seg
        seg.test()
    if args.task == 'ablation':
        import ablation_cifar100 as ab
        ab.main(test_concat = False)
    if args.task == 'fashion':
        import clf_fashion as fashion
        fashion.main()
    if args.task == 'counting':
        import counting_test as counting
        counting.main(args.path)
    if args.task == 'deepfish':
        import clf_test as classify
        classify.main(args.path)
    if args.task == 'cifar':
        import ablation_cifar100 as ab
        ab.main(test_concat = True)
    

    
    


if __name__ == "__main__":
    # check_model_test()
    # %% python3 run > out.txt
    main()
    # test()