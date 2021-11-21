from __future__ import division

import os
import numpy as np

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data

import torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import crop

# from dataloaders.cityscapes import CityScapes
from dataloaders.deepfish_seg import get_dataset
from torch.utils.data import DataLoader

from models.network_factory import fishnet150
from loss_function.loss_funcs import MultiClass_FocalLoss
from utils.tools import AverageMeter, SegMonitor, confusion, show_image, plot_graph

# from skimage.segmentation import mark_boundaries
# from skimage import data, io, segmentation, color
# from skimage.measure import label

import matplotlib.pyplot as plt
import seaborn as sns

from torchsummary import summary

use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_size = (224, 224)
root_dir = "/Users/xavier/Programming/FishNet"

use_mode = 'train'  # train
class_names = [
   
]

hyperparam_config = {
    'lr': 1e-3,
    'mom': 0.9,
    'scheduler_type': 'step',  # plateau
    'epochs': 50,
    'step_size': 10,
    'patience': 100,
    'gamma': 0.5,
    'weight_decay': 1e-5,
    'dropout': 0.7,
    'loss_function': 'crossentropy',  # focal
    'optimizer': 'SGD',
    'model': 'fishnet150',  # fishnet150, fishnet99
    'feature_extract': False,
    'interpolate' : 'bilinear' #bilinear
}

gpu_config = {
    'num_gpus': 8,
    'imgs_per_gpu': 8,
    'workers_per_gpu': 1,
    'train': [0, 1],
    'test': [0, 1]
}

batch_size = gpu_config['num_gpus'] * gpu_config['imgs_per_gpu']
num_workers = gpu_config['num_gpus'] * gpu_config['workers_per_gpu']



def calc_accuracy(pred, label):
    
    preds = torch.argmax(pred, 0)
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (pred == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    
    return acc


def calc_pixel_accuracy(pred, label):
    cm = confusion(pred, label)
    tn , fp = cm[0]
    fn, tp = cm[1]
    print('tn: {}, fp: {}, fn: {}, tp:{}'.format(tn, fp, fn ,tp))
    acc = (tp + tn) / (tp+tn+fp+fn)

    return acc
    

def train(train_loader, model, criterion, optimizer, epoch):
    train_loss = AverageMeter()
    # train_acc = AverageMeter()
    # seg_monitor = SegMonitor()
    model.train()

    for i, data in enumerate(train_loader):

        inputs = data["images"]
        # 224x224x3
        # print('input data size {}'.format(inputs.size()))
        labels = data["mask_classes"]
        # 1x1080x1920
        # print('label size {}'.format(labels.size()))
        optimizer.zero_grad()
        logits = model(inputs)

        # acc = calc_pixel_accuracy(logits, labels)
        # seg_monitor.add(acc)
        N = inputs.size(0)
        loss = criterion(logits, labels.to(device))
        loss /= N
        # acc /= N
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), N)
        # train_acc.update(acc, N)
        # print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
        #     epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg*100))
        print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
            epoch, i + 1, len(train_loader), train_loss.avg))

    return train_loss.avg


def validate(data_loader, model, criterion, epoch):
    val_loss = AverageMeter()
    # val_acc = AverageMeter()
    model.eval()

    for i, data in enumerate(data_loader):

        inputs = data["images"]
        # 224x224x3
        # print('input data size {}'.format(inputs.size()))
        labels = data["mask_classes"]
        # 1x1080x1920
        # print('label size {}'.format(labels.size()))

        logits = model(inputs)
        acc = calc_accuracy(logits, labels)
        N = inputs.size(0)
        loss = criterion(logits, labels.to(device))
        # loss /= N
        # acc /= N
        val_loss.update(loss.item(), N)
        # val_acc.update(acc, N)
        # print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg*100))
        print('[epoch %d], [val loss %.5f]' % (epoch, val_loss.avg))

    return val_loss.avg


class FishNet150_seg_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = fishnet150()
        self.n_classes = len(class_names)
        new_state_dict = OrderedDict()
        if hyperparam_config['model'] == 'fishnet150':
            chkp_path = os.path.join(root_dir, 'checkpoints/fishnet150_ckpt.tar')

        checkpoint = torch.load(chkp_path, map_location=torch.device(device))
        state_dict = checkpoint['state_dict']
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)

        self.model.fish.fish[9][4][0] = nn.ConvTranspose2d(
            1056, 1056 // 16, kernel_size=2, stride=2)
        # output 112x112x66
        self.model.fish.fish[9][4][1] = nn.Upsample(
            scale_factor=4, mode=hyperparam_config['interpolate'], align_corners=True)

        self.up_reduce_channels_2 = nn.ConvTranspose2d(
            66, 66 // 11, kernel_size=2, stride=2)
        self.upsample_2 = nn.Upsample(
            scale_factor=4, mode=hyperparam_config['interpolate'], align_corners=True)

        self.up_reduce_channels_3 = nn.ConvTranspose2d(6,  1, kernel_size=2, stride=2)
        

    def forward(self, x):
        x = self.model(x)

        x = self.up_reduce_channels_2(x)
        x = self.upsample_2(x)
        x = self.up_reduce_channels_3(x)

        x = nn.functional.interpolate(
            x, size=(1080, 1920), mode=hyperparam_config['interpolate'], align_corners=True)
        x = torch.squeeze(x)
        x = nn.Sigmoid()(x)
        # print('final logit size {}'.format(x.size()))
        return x


def test():
    path = os.path.join(root_dir, "fishnet_image_seg.pth")
    model = FishNet150_seg_model()
    model.load_state_dict(torch.load(
        path, map_location=torch.device(device)),strict=False)
    model.eval()
    data_transforms_test = T.Compose([
        T.Resize(256),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = get_dataset(dataset_name="fish_seg",
                               split="test",
                               transform=data_transforms_test,
                               datadir=os.path.join(root_dir, 'data/DeepFish'))
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=False
    )
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            visualise(model, data, i)


def visualise(model, data, index):
    plt.figure()
    sns.set_style("white")
    f, ax = plt.subplots(3,1, figsize=(10,20))
    images = data['images']
    gt = data['mask_classes']
    logits = model(images).to(device)
    ax[0].imshow(logits.squeeze().cpu().detach().numpy())
    ax[1].imshow(gt[0].squeeze())
    ax[2].imshow(images[0].squeeze().permute(1,2,0))
    
    file_name = str(index) +'_test'+'.png'
    plt.savefig(os.path.join(root_dir,'test_viz',file_name))
    
    
    # cm = confusion(logits.float(),gt.float() )
    # print(cm)
    # ConfusionMatrixDisplay.from_predictions(gt.float(), logits.float())
    # file_name_cm = str(index) +'_cm'+'.png'
    # plt.savefig(os.path.join(root_dir,'test_viz',file_name_cm))


def main():
    model = FishNet150_seg_model()
    model = model.to(device)

    # for finetuning or feature_extracting
    if hyperparam_config['feature_extract'] == True:
        parameters_to_be_updated = []
        for name, parameters in model.named_parameters():
            if parameters.requires_grad == True:
                parameters_to_be_updated.append(parameters)
    else:
        parameters_to_be_updated = model.parameters()

    if use_mode == 'test':
        model.eval()
    else:  # trainval
        epochs = hyperparam_config['epochs']

    # change optimizer
    if hyperparam_config['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            parameters_to_be_updated, lr=hyperparam_config['lr'], momentum=hyperparam_config['mom'], weight_decay=hyperparam_config['weight_decay'])
    elif hyperparam_config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            parameters_to_be_updated, lr=hyperparam_config['lr'], weight_decay=hyperparam_config['weight_decay'])

    if hyperparam_config['loss_function'] == 'crossentropy':
        criterion = nn.BCELoss()
    elif hyperparam_config['loss_function'] == 'focal':
        criterion = MultiClass_FocalLoss()

    if hyperparam_config['scheduler_type'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=hyperparam_config['step_size'], gamma=hyperparam_config['gamma'])
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, 'min', patience=hyperparam_config['patience'], min_lr=1e-10)

    data_transforms_train = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms_val = T.Compose([
        T.Resize(256),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = get_dataset(dataset_name="fish_seg",
                                split="train",
                                transform=data_transforms_train,
                                datadir=os.path.join(root_dir, 'data/DeepFish'))

    val_dataset = get_dataset(dataset_name="fish_seg", split="val",
                              transform=data_transforms_val,
                              datadir=os.path.join(root_dir, 'data/DeepFish'))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=False
    )

    # # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])

    val_loss = []
    train_loss = []


    for epoch in range(epochs):
        print('epoch {0} of {1}'.format(epoch+1, hyperparam_config['epochs']))
        t_loss = train(train_loader, model, criterion, optimizer, epoch)
        v_loss = validate(val_loader, model, criterion, epoch)
        scheduler.step()
        val_loss.append(v_loss)
        train_loss.append(t_loss)

    model_name = 'fishnet' + '_' + 'image_seg' + '.pth'
    torch.save(model.state_dict(), os.path.join(root_dir, model_name))
    plot_graph(train_loss, val_loss, hyperparam_config, root_dir)


def check_model_test():
    from torchsummary import summary
    model = FishNet150_seg_model().to(device)
    summary(model, (3, 224, 224))


if __name__ == "__main__":
    # check_model_test()
    # %% python3 run > out.txt
    main()
    # test()
