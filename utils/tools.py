import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import sampler

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialize()

    def initialize(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def show_image(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    return inp
    # plt.imshow(inp)

    # if title is not None:
    #     plt.title(title)
    # plt.pause(1000)  # pause a bit so that plots are updated


def plot_graph(train_loss, val_loss, hyperparam_config, root_dir):

    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Val')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    graph_name = hyperparam_config['model'] + '_' + str(hyperparam_config['optimizer']) + str(hyperparam_config['lr']) + '_WD:' + str(
        hyperparam_config['weight_decay']) + '_' + hyperparam_config['loss_function'] + '_' + 'loss.png'
    plt.savefig(os.path.join(root_dir, graph_name))
    plt.close()

class SegMonitor:
    def __init__(self):
        self.cf = None

    def add(self, cf):
        if self.cf is None:
            self.cf = cf 
        else:
            self.cf += cf

    def get_avg_score(self):
        # return -1 
        Inter = np.diag(self.cf)
        G = self.cf.sum(axis=1)
        P = self.cf.sum(axis=0)
        union = G + P - Inter

        nz = union != 0
        mIoU = Inter[nz] / union[nz]
        mIoU = np.mean(mIoU)

        return {"val_seg":1. - mIoU}

def confusion(prediction, truth):
    confusion_vector = prediction / truth

    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()
    cm = np.array([[tn,fp],[fn,tp]])
    return cm

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
