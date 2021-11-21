
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


## Dataloading for Fashion Dataset ###
from dataloaders import fashion_loader as FashionDataset
from utils.fashion_utils import single_acc, multi_acc, imshow, predict_acc, predict


from torch.utils.data import DataLoader
from torchvision import transforms

from torchsummary import summary
import matplotlib.pyplot as plt
import time

# from RandoCNN import Network


datadir = r'C:\Users\yipji\Offline Documents\Big Datasets'
projdir = datadir+r'\CE7454 Fashion Dataset\FashionDataset'
   
img_dir = Path(projdir+r'\img')

x_train = Path(projdir+r'\split\train.txt')
y_train = Path(projdir+r'\split\train_attr.txt')

x_val = Path(projdir+r'\split\val.txt')
y_val = Path(projdir+r'\split\val_attr.txt')
    
x_test = Path(projdir+r'\split\test.txt')

torch.manual_seed(0)

tsfms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    # transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomErasing(p=0.6),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1)),
    ])

tsfms0 = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1)),    
    ])

train_dataset = FashionDataset(img_dir, x_train, y_train, one_hot = True, transform=tsfms)
val_dataset = FashionDataset(img_dir, x_val, y_val, one_hot = True, transform=tsfms)
test_dataset = FashionDataset(img_dir, x_test, y_val, one_hot = True, transform=tsfms0) #using y_val as placeholder since y_test is not available
val0_dataset = FashionDataset(img_dir, x_val, y_val, one_hot = True, transform=tsfms0)

train_dataloader = DataLoader(train_dataset,batch_size=25,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=25,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=25,shuffle=False)
val0_dataloader = DataLoader(val0_dataset,batch_size=25,shuffle=True)

# get some random training images
dataiter = iter(train_dataloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images[0:5]))


#### Setting up model ####
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torch.optim.lr_scheduler as lr

from models.FishNet150_fashion import FishNet150_fashion

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FishNet150_fashion()
    net.to(device)


    #### TRAINING SECTION ####

    #Settings#
    criterion = nn.BCELoss().to(device) #adam learning rate about 0.001 is good
    # criterion = FocalLoss(alpha = 10, gamma = 5).to(device) #adam learning rate about 0.000001 is good
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # scheduler1 = lr.ExponentialLR(optimizer, gamma=0.9, verbose = True)
    scheduler2 = lr.MultiStepLR(optimizer,milestones=[15,20],gamma=0.1, verbose = True)


    run_number = 3
    model_name = "FishNet_fashion"
    total_epochs = 25

    #setting up counters#
    train_avg_acc_by_epoch = []
    val_avg_acc_by_epoch = []
    tic0 = time.perf_counter()

    for epoch in range(5):  # loop over the dataset multiple times
        
        ###Epoch start counter
        tic = time.perf_counter()
        
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
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
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        
        # scheduler1.step()
        scheduler2.step()
        
        ###Validation start counter
        tic2 = time.perf_counter()
        
        with torch.no_grad():
            ep_train_acc = predict_acc(net,train_dataloader)
            ep_val_acc = predict_acc(net,val_dataloader)
        train_avg_acc_by_epoch.append(ep_train_acc)    
        val_avg_acc_by_epoch.append(ep_val_acc)
        
        #Epoch End Counter
        toc = time.perf_counter()
        print(f'train acc: {ep_train_acc:0.4f}')
        print(f'val acc: {ep_val_acc:0.4f}')
        print(f"Epoch {epoch+1} of {total_epochs} Ended. Total epoch took {toc-tic:0.4f}s. Validation took {toc-tic2:04f}s")

    total_epochs += 5
    final_val = predict_acc(net.to(device), val0_dataloader)
    # final_val = ep_val_acc    
    torch.save(net.state_dict(), f'./Run{run_number}_{model_name}_{total_epochs:0.0f}ep_state.pth')
    plt.plot(train_avg_acc_by_epoch)
    plt.plot(val_avg_acc_by_epoch)
    print(f'Finished Training. Total time {toc-tic0:0.4f}s, Final Validation Acc {final_val:0.4f}')


    torch.save(train_avg_acc_by_epoch, f'./Run{run_number}_{model_name}_{total_epochs:0.0f}ep_train_acc.pt')
    torch.save(val_avg_acc_by_epoch, f'./Run{run_number}_{model_name}_{total_epochs:0.0f}ep_vat_acc.pt')


    ###Saving Results###
    results = predict(net,test_dataloader)
    np.savetxt(f'Run{run_number}_{model_name}_{total_epochs:0.0f}ep_PREDICTION.txt',torch.stack(results).numpy(),delimiter=" ",fmt = "%0.0f")