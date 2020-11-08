
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def conv3x3x3(inplane, outplane, stride=1,padding=0,kernel_size=3):
    """
    Simple 3x3x3 convolutional block. 
    """
    return nn.Conv3d(inplane,outplane,kernel_size=kernel_size,stride=stride,padding=padding,bias=True)

class BasicBlock(nn.Module):
    """
    Basic convolutional block used for all non-dense blocks in my network. Specifically, this is the downsampling layer and bottleneck layers. 
    Since the bottleneck layer is better suited for dropout, I include the optional choice for either here, with a preset dropout val of 0.5.
    """
    def __init__(self,inplane,outplane,stride = 1,padding = 1, kernel_size = 3):
        super(BasicBlock, self).__init__()
        self.padding = padding
        self.conv1 = conv3x3x3(inplane,outplane,padding=padding,stride=stride,kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm3d(outplane)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


def crop_tensor(x):
    x = x.narrow(2,1,x.shape[2]-3).narrow(3,1,x.shape[3]-3).narrow(4,1,x.shape[4]-3).contiguous()
    return x


class DM2Gal(nn.Module):
    """
    """
    def _make_layer(self,block,inplanes,outplanes,nblocks,stride=1, padding=0,kernel_size=3): 
        layers = []
        for i in range(0,nblocks):
            layers.append(block(inplanes,outplanes,stride,padding,kernel_size))
            inplanes = outplanes  #how channelling is dealt with. Neat!
        return nn.Sequential(*layers)
    
    def __init__(self, block,in_ch=2,ch=256,nblocks = 4):
        super(DM2Gal,self).__init__()
        self.ch = ch
        
        self.layer1 = self._make_layer(block, in_ch, ch, nblocks=nblocks,stride=1,padding=0,kernel_size=7) 
        self.layer2 = self._make_layer(block, ch, 2*ch, nblocks=1,stride=2,kernel_size=3) 
        self.layer3 = self._make_layer(block, 2*ch, 2*ch, nblocks=nblocks,stride=1,padding=0,kernel_size=5)
        self.layer4 = self._make_layer(block, 2*ch, 2*2*ch, nblocks=1,stride=2,kernel_size=3)
        self.layer5 = self._make_layer(block, 2*2*ch, 2*2*ch, nblocks=nblocks,stride=1,padding=0,kernel_size=3)
        self.layer6 = self._make_layer(block, 2*2*ch, 2*2*2*ch, nblocks=1,stride=2,kernel_size=3)
        self.fc1 = nn.Linear(2*2*2*ch*1*1*1, 256) 
     #   self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(256,1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
    
        x = x.view(-1,2*2*2*self.ch*1*1*1) #reshape 
        x = F.relu(self.fc1(x))
      #  x = self.dropout(x)
        x = self.fc2(x)

        return x
    

    
