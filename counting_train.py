
import argparse
import torch
import torch.nn as nn
from utils import exp_configs
import utils as ut
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision import transforms

from dataloaders import get_dataset


from models.FishNet150_count import FishNet150_count

import pandas as pd

import time

import torch.optim as optim
import torch.optim.lr_scheduler as lr

from utils.fishy_utils import predict_acc, BCEDiceLoss, CrossEntropyLoss2d, MultiClass_FocalLoss

import matplotlib.pyplot as plt

# datadir = r'C:/Users/yipji/Offline Documents/Big Datasets/DeepFish/DeepFish'

# train_dataset = get_dataset("tiny_fish_clf", "train", transform = "resize_normalize", datadir = datadir) 

###Accuracy Metric
    # test_data = next(iter(val_loader))
    # images = test_data['images']
    # labels = test_data['counts']
    # net.cpu()
    # prediction = net(images.cpu()).round().squeeze()
    # accuracy = sum(prediction == labels)/len(labels)
    

use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict_acc(net,loader):
    
    total_acc = 0
    total_batch = 0
    for i in loader:
        images = i['images'].to(device)
        labels = i['counts'].to(device)
        prediction = net(images).round().squeeze()
        accuracy = sum(prediction == labels)/len(labels)
        total_acc += accuracy
        total_batch += 1
    
    final_acc = total_acc/total_batch
    
    return final_acc

def predict_mae(net,loader):
    
    total_mae = 0
    total_batch = 0
    for i in loader:
        # images = i['images'].to(device)
        # labels = i['counts'].to(device)
        # prediction = net(images).round().squeeze()
        mae = torch.mean(abs(i['counts'].to(device) - net(i['images'].to(device)).round().squeeze()))
        total_mae += mae
        total_batch += 1
    
    final_mae = total_mae/total_batch
    
    return final_mae

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir',
                        type=str, default='C:/Users/yipji/Offline Documents/Big Datasets/DeepFish/DeepFish/')
    parser.add_argument("-e", "--exp_config", default='fish_reg')
    parser.add_argument("-uc", "--use_cuda", type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda else 'cpu')

    exp_dict = exp_configs.EXP_GROUPS[args.exp_config][0]
    
    # Dataset
    # Load val set and train set
    val_set = get_dataset(dataset_name=exp_dict["dataset"], split="val",
                                   transform=exp_dict.get("val_transform"),
                                   datadir=args.datadir)
    train_set = get_dataset(dataset_name=exp_dict["dataset"],
                                     split="train", 
                                     transform=exp_dict.get("train_transform"), #overwrite the existing dataset's transform with my own, which has additional image augmentations
                                     datadir=args.datadir)
    test_set = get_dataset(dataset_name=exp_dict["dataset"], split="test",
                                   transform="resize_normalize",
                                   datadir=args.datadir)

    
    # Load train loader, val loader, and vis loader
    train_loader = DataLoader(train_set, 
                            sampler=RandomSampler(train_set,
                            replacement=True, num_samples=max(min(500, 
                                                            len(train_set)), 
                                                            len(val_set))),
                            batch_size=exp_dict["batch_size"])
    test_loader = DataLoader(test_set, shuffle=False, batch_size=exp_dict["batch_size"])
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
    net = FishNet150_count()
    net.to(device)
    
    
    #Setting up the Run#
    # opt = torch.optim.Adam(net.parameters(), 
    #                     lr=1e-3, weight_decay=0.0001)
    # model_original = models.get_model(exp_dict["model"], exp_dict=exp_dict).cuda()
    # model = wrappers.get_wrapper(exp_dict["wrapper"], model=model_original, opt=opt).to(device)
    # model = wrappers.get_wrapper(exp_dict["wrapper"], model=net, opt=opt).to(device)
    
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss().to(device) 
    # criterion = BCEDiceLoss().to(device)
    # criterion = MultiClass_FocalLoss().to(device)
    # criterion = FocalLoss().to(device) 
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # scheduler = lr.ReduceLROnPlateau(optimizer, 'min', patience = 2)
    # scheduler1 = lr.ExponentialLR(optimizer, gamma=0.9, verbose = True)
    scheduler2 = lr.MultiStepLR(optimizer,milestones=[15,25],gamma=0.1, verbose = True)
    
    #meta info
    run_number = 16
    model_name = 'fishnet-regL1'
    total_epochs = exp_dict["max_epoch"]
    
    #setting up counters#
    train_avg_acc_by_epoch = []
    val_avg_acc_by_epoch = []
    tic0 = time.perf_counter()

    score_list_v = []
    score_list_t = []
    
    total_epochs = exp_dict["max_epoch"]
    
    for epoch in range(total_epochs):  # loop over the dataset multiple times
        
        ###Epoch start counter
        tic = time.perf_counter()
        
        running_loss = 0.0
    
        for i, data in enumerate(train_loader, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['images']
            labels = data['counts']
            inputs, labels = inputs.to(device), labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            # print(loss.item())
            if i % 5 == 4:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
            
            # scheduler.step(round(loss.item(),6))
        # scheduler1.step()
        # scheduler2.step()
        
        ###Validation start counter
        tic2 = time.perf_counter()
        
        with torch.no_grad():
            ep_train_acc = predict_mae(net,train_loader)
            ep_val_acc = predict_mae(net,val_loader)
        train_avg_acc_by_epoch.append(ep_train_acc)    
        val_avg_acc_by_epoch.append(ep_val_acc)
        
    #     #Epoch End Counter
        toc = time.perf_counter()
        print(f'train mae: {ep_train_acc:0.4f}')
        print(f'val mae: {ep_val_acc:0.4f}')
        print(f"Epoch {epoch+1} of {total_epochs} Ended. Total epoch took {toc-tic:0.4f}s. Validation took {toc-tic2:04f}s")
    
    final_val = ep_val_acc
    torch.save(net.state_dict(), f'./models/Run{run_number}_{model_name}_{total_epochs:0.0f}ep_state.pth')
    plt.plot(train_avg_acc_by_epoch)
    plt.plot(val_avg_acc_by_epoch)
    print(f'Finished Training. Total time {toc-tic0:0.4f}s, Final Validation MAE {final_val:0.4f}')
    # print(f'Finished Training. Total time {toc-tic0:0.4f}s')
    
    
    torch.save(train_avg_acc_by_epoch, f'./Run{run_number}_{model_name}_{total_epochs:0.0f}ep_train_acc.pt')
    torch.save(val_avg_acc_by_epoch, f'./Run{run_number}_{model_name}_{total_epochs:0.0f}ep_vat_acc.pt')
    
    
    # ###Saving Results###
    # results = predict(net,test_dataloader)
    # np.savetxt(f'Run{run_number}_{model_name}_{total_epochs:0.0f}ep_PREDICTION.txt',torch.stack(results).numpy(),delimiter=" ",fmt = "%0.0f")
        
#%%
import matplotlib.pyplot as plt

plt.figure()
f, ax = plt.subplots(2,1, figsize=(10,10))

ax[0].imshow(net(next(iter(train_loader))['images'][:1].cuda()).squeeze().cpu().detach().numpy())     
ax[1].imshow(next(iter(train_loader))['images'][0].squeeze().permute(1,2,0))  

#%%
import matplotlib.pyplot as plt

plt.figure()
f, ax = plt.subplots(2,1, figsize=(10,10))

ax[0].imshow(net(next(iter(vis_loader))['images'][:1].cuda()).squeeze().cpu().detach().numpy())     
ax[1].imshow(next(iter(vis_loader))['images'][0].squeeze().permute(1,2,0))  
