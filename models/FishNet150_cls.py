
import models.network_factory as nf
import torch
import torch.nn as nn
import os

class FishNet150_cls(nn.Module):
    def __init__(self, n_classes=1, pretrained = True):
        super().__init__()
        self.n_classes = n_classes
        self.net = nf.fishnet150()
        self.last_linear = nn.Linear(1000, self.n_classes)
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
        
        # self.net.fish.fish[9][4][1] = nn.Sequential(nn.Flatten(), 
        #                                             nn.Linear(in_features=1056, out_features= self.n_classes, bias=True),
        #                                             nn.Softmax()
        #                                             )
        
        self.fc = nn.Sequential(nn.Linear(in_features=1000, out_features= self.n_classes, bias=True),
                                nn.Softmax(dim=1)
                                )
    def logits(self, features):
        # x = self.avgpool_1a(features)
        x = features.view(features.size(0), -1)
        x = self.last_linear(x)
        return x
    
    def forward(self, x):
        x = self.net(x)
        x = self.logits(x)
        return x
    
    
if __name__ == '__main__':
    from torchsummary import summary
    net = FishNet150_cls().cuda()
    summary(net, (3,224,224))