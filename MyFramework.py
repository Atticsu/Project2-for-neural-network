import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Stem(nn.Module):  # stem 层
    def __init__(self):
        super(Stem, self).__init__()
        
        # First set of layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)  # (299x299x3) -> (149x149x32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)           # (149x149x32) -> (147x147x32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # (147x147x32) -> (147x147x64)
        
        # Second set of layers
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)    # (147x147x64) -> (73x73x64)
        self.conv4 = nn.Conv2d(64, 96, kernel_size=3, stride=2)  # (147x147x64) -> (73x73x96)
        
        # Third set of layers
        self.conv5 = nn.Conv2d(160, 64, kernel_size=1)           # (73x73x160) -> (73x73x64)
        self.conv6 = nn.Conv2d(64, 96, kernel_size=3)            # (73x73x64) -> (71x71x96)
        self.conv7 = nn.Conv2d(160, 64, kernel_size=1)           # (73x73x160) -> (73x73x64)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)) # (73x73x64) -> (73x73x64)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)) # (73x73x64) -> (73x73x64)
        self.conv10 = nn.Conv2d(64, 96, kernel_size=3)           # (73x73x64) -> (71x71x96)
        
        # Fourth set of layers
        self.conv11 = nn.Conv2d(192, 192, kernel_size=3, stride=2) # (71x71x192) -> (35x35x192)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)      # (71x71x192) -> (35x35x192)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  
        # # Second set of layers
        batch1 = self.maxpool1(x)
        batch2 = self.conv4(x)
        x1 = torch.cat((batch1, batch2), dim=1)
        batch3 = self.conv5(x1)
        batch3 = self.conv6(batch3)
        batch4 = self.conv7(x1)
        batch4 = self.conv8(batch4)
        batch4 = self.conv9(batch4)
        batch4 = self.conv10(batch4)
        x2 = torch.cat((batch3, batch4), dim=1)
        batch5 = self.conv11(x2)
        batch6 = self.maxpool2(x2)
        output = torch.cat((batch5,batch6),dim =1)
        return output


class InceptionA(nn.Module):
    def __init__(self, input_channels=384):
        super(InceptionA, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.conv3 =  nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv3_1x1_proj = nn.Conv2d(input_channels, 96, kernel_size=1)
    
        # average pooling
        self.avg_pooling = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_1x1_proj = nn.Conv2d(input_channels, 96, kernel_size=1)
        

    def forward(self, x):
        #batch1
        batch1 = self.avg_pooling(x)
        batch1 = self.conv_1x1_proj(batch1)
        #batch2
        batch2 = self.conv3_1x1_proj(x)
        #batch3
        batch3 = self.conv1(x)
        batch3 = self.conv2(batch3)

        #batch4
        batch4 = self.conv1(x) 
        batch4 = self.conv2(batch4)
        batch4 = self.conv3(batch4)
        return torch.cat((batch1, batch2, batch3, batch4), dim = 1)
    

#添加Res 层