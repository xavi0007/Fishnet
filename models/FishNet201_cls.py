
import models.network_factory as nf
import torch
import torch.nn as nn


class FishNet201_cls(nn.Module):
    def __init__(self, n_classes=1, pretrained = True):
        super().__init__()
        self.n_classes = n_classes
        self.net = nf.fishnet201()
        self.avgpool_1a = nn.AvgPool2d(5, count_include_pad=False)
        self.last_linear = nn.Linear(1000, self.n_classes)
        self.softmax = nn.Softmax(dim=0)
        self.fc = nn.Sequential(nn.Linear(in_features=1000, out_features= self.n_classes, bias=True),
                                
                                )
    def logits(self, features):
        # x = self.avgpool_1a(features)
        x = features.view(features.size(0), -1)
        x = self.last_linear(x)
        return x
    
    def forward(self, x):
        x = self.net(x)
        # x = self.fc(x)
        x = self.logits(x)
        # x = self.softmax(x)
        return x
    
    
if __name__ == '__main__':
    from torchsummary import summary
    net = FishNet201_cls().cuda()
    summary(net, (3,224,224))