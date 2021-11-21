
import models.network_factory as nf
import torch
import torch.nn as nn
import os

class FishNet150_count(nn.Module):
    def __init__(self, n_classes=1, pretrained = True, trainable = True, softmax = False):
        super().__init__()
        self.softmax = softmax
        self.n_classes = n_classes
        self.net = nf.fishnet150()
        self.linear0 = nn.Linear(in_features = 1000, out_features = 512)
        self.linear1 = nn.Linear(in_features = 512, out_features = 256)
        self.linear2 = nn.Linear(in_features = 256, out_features = self.n_classes)
        self.relu = nn.ReLU()
        if pretrained:
            cwd = os.getcwd()
            checkpoint_path = os.path.join(cwd, 'checkpoints/fishnet150_ckpt.tar')
            checkpoint = torch.load(checkpoint_path)
            # best_prec1 = checkpoint['best_prec1']
            state_dict = checkpoint['state_dict']
        
            from collections import OrderedDict
            
            new_state_dict = OrderedDict()
            
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            
            self.net.load_state_dict(new_state_dict)
        
        if not trainable:
            for param in self.net.parameters():
                param.requires_grad = False
            # for param in self.net.fish.fish[1].parameters():
            #     param.requires_grad = True
            for param in self.net.fish.fish[2].parameters():
                param.requires_grad = True
            for param in self.net.fish.fish[3].parameters():
                param.requires_grad = True
            for param in self.net.fish.fish[4].parameters():
                param.requires_grad = True
            # for param in self.net.fish.fish[5].parameters():
            #     param.requires_grad = True
        # self.net.fish.fish[9][4][1] = nn.Sequential(nn.Flatten(), 
        #                                             nn.Linear(in_features=1056, out_features= self.n_classes, bias=True),
        #                                             )
            
    def forward(self, x):
        x = self.net(x)
        x = self.linear0(x)
        x = self.linear1(x)
        x = self.linear2(x)
        if self.softmax:
            x = nn.Softmax(x)
        else:
            x = self.relu(x)
        return x
    
    
if __name__ == '__main__':
    from torchsummary import summary
    net = FishNet150_count(trainable=False).cuda()
    summary(net, (3,224,224))