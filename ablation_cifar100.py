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
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from utils.data_aug import ColorAugmentation
import os
from torch.autograd.variable import Variable
import models_ablation
from models_ablation.net_factory import fishnet150


# screen -r fishnet
# conda activate pytorchenv 
# cd DLAssignment2/FishNet
# python main.py --config "cfgs/fishnet150.yaml" 



best_prec1 = 0

USE_GPU = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


config = {
    'arch': "fishnet150",
    'model_name': 'fishnet',
    'workers': 1,
    'batch_size': 32,
    'epochs': 100,
    'start_epoch': 0,
    'policy': "step",
    'base_lr': 1e-3,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'print_freq': 10,
    'save_path': "checkpoints/fishnet150_bs256",
    'input_size' : 244,
    'image_size': 256,
    'resume': False,
    'evaluate': False,
 
}

def main(test_concat = True):
    global best_prec1, USE_GPU
    

    # create models
    if config['input_size'] != 224 or config['image_size'] != 256:
        image_size = config['image_size'] 
        input_size = config['input_size']
    else:
        image_size = 256
        input_size = 224
    print("Input image size: {}, test size: {}".format(image_size, input_size))


    if test_concat == True:
        model = models_ablation.fishnet_wo_concat()
    else:
        model = fishnet150()

    # if USE_GPU:
    #     model = model.cuda()
    #     
    if USE_GPU:
        model = model.to(device)
        # model = torch.nn.DataParallel(model)

    # count_params(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), config['base_lr'], momentum=config['momentum'],
                                weight_decay=config['weight_decay'])

    # optionally resume from a checkpoint
    if config["resume"]:
        if os.path.isfile(config["save_path"]):
            print("=> loading checkpoint '{}'".format(config["save_path"]))
            checkpoint = torch.load(config["save_path"])
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(config["save_path"]))

    cudnn.benchmark = True

    # Data loading code

    normalize = transforms.Normalize(mean=[0.5074,0.4867,0.4411],
                                     std=[0.2011,0.1987,0.2025])                    
    img_size = config['input_size']

    ratio = 224.0 / float(img_size)

    train_transform =transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ColorAugmentation(),
            normalize,
        ])

    test_transform =transforms.Compose([
        transforms.Resize(int(256 * ratio)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = CIFAR100(download=True,root="./data",transform=train_transform)
    val_dataset = CIFAR100(root="./data",train=False,transform=test_transform)

    train_sampler = None
    val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=(train_sampler is None),
        num_workers=config['workers'], pin_memory=(train_sampler is None), sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                                             num_workers=config['workers'], pin_memory=True, sampler=val_sampler)

    if config['evaluate']:
        validate(val_loader, model, criterion)
        return

    for epoch in range(config['start_epoch'], config['epochs']):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if not os.path.exists(config['save_path']):
            os.mkdir(config['save_path'])
        save_name = '{}/{}_{}_best.pth.tar'.format(config['save_path'], config['model_name'], epoch) if is_best else\
            '{}/{}_{}.pth.tar'.format(config['save_path'], config['model_name'], epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': config['arch'],
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, filename=save_name)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # #  pytorch 0.4.0 compatible
        # if '0.4.' in torch.__version__:
        #     if USE_GPU:
        #         input_var = torch.cuda.FloatTensor(input.cuda())
        #         target_var = torch.cuda.LongTensor(target.cuda())
        #     else:
        #         input_var = torch.FloatTensor(input)
        #         target_var = torch.LongTensor(target)
        # else:  # pytorch 0.3.1 or less compatible
        #     if USE_GPU:
        #         input = input.cuda()
        #         target = target.cuda()
        #     input_var = Variable(input)
        #     target_var = Variable(target)

        # compute output

   
        input_var = input.to(device)
        target_var = target.to(device)

        output = model(input_var)

        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))

        # measure accuracy and record loss
        reduced_prec1 = prec1.clone()
        reduced_prec5 = prec5.clone()

        top1.update(reduced_prec1[0])
        top5.update(reduced_prec5[0])

        reduced_loss = loss.data.clone()
        losses.update(reduced_loss)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #  check whether the network is well connected
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            with open('logs/{}_{}.log'.format(time_stp, config['arch']), 'a+') as flog:
                line = 'Epoch: [{0}][{1}/{2}]\t ' \
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                        batch_time=batch_time, loss=losses, top1=top1, top5=top5)
                print(line)
                flog.write('{}\n'.format(line))


def validate(val_loader, model, criterion):
    global time_stp
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # #  pytorch 0.4.0 compatible
        # if '0.4.' in torch.__version__:
        #     with torch.no_grad():
        #         if USE_GPU:
        #             input_var = torch.cuda.FloatTensor(input.cuda())
        #             target_var = torch.cuda.LongTensor(target.cuda())
        #         else:
        #             input_var = torch.FloatTensor(input)
        #             target_var = torch.LongTensor(target)
        # else:  # pytorch 0.3.1 or less compatible
        #     if USE_GPU:
        #         input = input.cuda()
        #         target = target.cuda()
        #     input_var = Variable(input, volatile=True)
        #     target_var = Variable(target, volatile=True)

        if USE_GPU:
            input_var = input.to(device)
            target_var = target.to(device)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            line = 'Test: [{0}/{1}]\t' \
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time,
                                                                   loss=losses, top1=top1, top5=top5)

            with open('logs/{}_{}.log'.format(time_stp, config['arch']), 'a+') as flog:
                flog.write('{}\n'.format(line))
                print(line)

    return top1.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = config['base_lr'] * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # 原文这里有跑不出来的bug view correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    time_stp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    main()
