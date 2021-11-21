import torch
from torch import nn
from torch.nn import functional as F

class FishNet_wo_Concat(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()

        # input image resolution: 3x224x224
        # stem: channel 3->64
        inplanes = 64
        self.conv1 = self.conv_stride2_bn_relu(3, 64 // 2, stride=2) # 112x112
        self.conv2 = self.conv_bn_relu(64 // 2, 64 // 2)
        self.conv3 = self.conv_bn_relu(64//2, 64)
        self.pool1 = nn.MaxPool2d(3, padding=1, stride=2)        
        # stem end resolution: 64x112x112
        
        # layer1 (64*56*56) (32*56*56)(128*56*56)(64*56*56)(32*56*56)(128*56*56)
        self.conv4 = self.bottleneck_conv_bn_relu(64) # （128*56*56)
        
        #self.pool=
        self.conv5 = self.bottleneck_conv_bn_relu(128) # （256*28*28)
        
        #self.pool=
        self.conv6 = self.bottleneck_conv_bn_relu(256) # （512*14*14)
        
        #self.pool= 
        self.conv7 = self.bottleneck_conv_bn_relu(512) # （1024*7*7)
        
        self.conv8 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
                             nn.BatchNorm2d(512),
                             nn.ReLU(inplace=True)) # （512*7*7)
        
        self.upsample = nn.Upsample(scale_factor=2) # （512*14*14)
        self.conv9 = self.bottleneck_body_conv_bn_relu(512) # （256*14*14)
        
        
        #self.up（256*28*28)
        self.conv10 = self.bottleneck_body_conv_bn_relu(256) # （128*28*28)
        
        #self.up（128*56*256)
        self.conv11 = self.bottleneck_body_conv_bn_relu(128) # （64*56*56)
        
        
        
        #self.pool=   
        self.conv12 = self.bottleneck_conv_bn_relu(64) # （128*28*28)
        
        #self.pool=  
        self.conv13 = self.bottleneck_conv_bn_relu(128) # （256*14*14)
    
        #self.pool=  
        self.conv14 = self.bottleneck_conv_bn_relu(256) # （512*7*7)
        
        
        self.avgpool=nn.AvgPool2d(7)
        self.conv15=nn.Conv2d(512, 10, kernel_size=1, bias=False)
        self.fc = nn.Linear(10, 10)
    
    def conv_bn_relu(self, in_ch, out_ch, stride=1):
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False),
                             nn.BatchNorm2d(out_ch),
                             nn.ReLU(inplace=True))
    
    def conv_stride2_bn_relu(self, in_ch, out_ch, stride=2):
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False),
                             nn.BatchNorm2d(out_ch),
                             nn.ReLU(inplace=True))   
    
    
    def bottleneck_conv_bn_relu(self, in_ch):
        return nn.Sequential(
                             nn.Conv2d(in_ch, in_ch//2, kernel_size=1, stride=1, bias=False),
                             nn.BatchNorm2d(in_ch//2),
                             nn.ReLU(inplace=True),
                             
                             nn.Conv2d(in_ch//2, in_ch//2, kernel_size=3,  padding=1, stride=1, bias=False),
                             nn.BatchNorm2d(in_ch//2),
                             nn.ReLU(inplace=True),
                            
                             nn.Conv2d(in_ch//2, in_ch*2, kernel_size=1, stride=1, bias=False),
                             nn.BatchNorm2d(in_ch*2),
                             nn.ReLU(inplace=True)
                            )

    def bottleneck_body_conv_bn_relu(self, in_ch):
        return nn.Sequential(
                             nn.Conv2d(in_ch, in_ch//2, kernel_size=1, stride=1, bias=False),
                             nn.BatchNorm2d(in_ch//2),
                             nn.ReLU(inplace=True),
                             
                             nn.Conv2d(in_ch//2, in_ch//8, kernel_size=1,  stride=1, bias=False),
                             nn.BatchNorm2d(in_ch//8),
                             nn.ReLU(inplace=True),
            
                             nn.Conv2d(in_ch//8, in_ch//8, kernel_size=3,  padding=1, stride=1, bias=False),
                             nn.BatchNorm2d(in_ch//8),
                             nn.ReLU(inplace=True),
                            
                             nn.Conv2d(in_ch//8, in_ch//2, kernel_size=1, stride=1, bias=False),
                             nn.BatchNorm2d(in_ch//2),
                             nn.ReLU(inplace=True)
                            )

    
    def forward(self, x):
        #stem
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
  
        
        #Tail1
        x = self.conv4(x)
        
        #Tail2
        x = self.pool1(x)
        x = self.conv5(x)
        
        #Tail3
        x = self.pool1(x)
        x = self.conv6(x)
        
        #Tail4
        x = self.pool1(x)
        x = self.conv7(x)
        
        # 
        x = self.conv8(x)
        
        #body 1
        x = self.upsample(x)
        x = self.conv9(x)
        
        #body 2
        x = self.upsample(x)
        x = self.conv10(x)
        
        #body 3
        x = self.upsample(x)
        x = self.conv11(x)
        
        # head1
        x = self.pool1(x)
        x = self.conv12(x)
        
        # head2 
        x = self.pool1(x)
        x = self.conv13(x)    
        
        # head3
        x = self.pool1(x)
        x = self.conv14(x)        

        # end
        x = self.avgpool(x)
        x = self.conv15(x) 
        out = x.view(x.size(0), -1)
        
        return out 