# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 14:38:35 2021

@author: yipji
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def single_acc(pred, target):
    '''
    This function accepts tensors and converts them into numpy arrays for prediction
    Dimensions should be torch.Size([number of samples])
    If feeding in a list, use torch.cat
    '''

    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    result = pred-target
    
    wrongs = np.count_nonzero(result)
    
    accuracy = 1-(wrongs/len(result))
    
    return accuracy

def multi_acc(pred,target,avg = False):
    '''
    This function accepts two dimenstional tensors
    First dimension is number of tensors
    Second dimension is number of classes
    
    '''
    if isinstance(pred, list):
        pred = torch.round(torch.cat(pred))
    if isinstance(target, list):
        target = torch.cat(target)
    
    numclasses = pred.shape[1]
    accuracy = []
    for i in range(numclasses):
        accuracy.append(single_acc(pred[:,i],target[:,i]))
    
    if avg:
        accuracy=np.mean(accuracy)
    
    return accuracy

def predict_acc(model, dataloader):
    acc = 0
    samps = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, (inputs, targets) in enumerate(dataloader):
        #evaluate the model on the test set
        inputs, targets = inputs.to(device), targets.to(device)    
        
        if not len(inputs)==len(targets):
            raise print("target and label size mismatch")
        
        t_yhat = model(inputs).argmax(dim=2).flatten(1,2)
        t_gt = targets.argmax(dim=2)
        n = len(inputs)
        batch_acc = multi_acc(t_yhat, t_gt, avg=True)
        samps += n
        acc += n*batch_acc
    return acc/samps

def predict(model, dataloader):
    
    yhat = []
        
    for i, (inputs, targets ) in enumerate(dataloader):
        #evaluate the model on the test set
        inputs = inputs.to("cpu")
        model = model.to("cpu")
                
        yhat += (model(inputs).argmax(dim=2).flatten(1,2))

    return yhat


