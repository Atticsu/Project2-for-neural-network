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

class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1x1 Convolution
        self.branch1x1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1, stride=1)
        
        # 1x1 Convolution followed by 3x3 Convolution
        self.branch3x3_1 = nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1, stride=1)
        self.branch3x3_2 = nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, stride=1, padding=1)
        
        # 1x1 Convolution followed by 5x5 Convolution
        self.branch5x5_1 = nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1, stride=1)
        self.branch5x5_2 = nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, stride=1, padding=2)
        
        # 3x3 MaxPool followed by 1x1 Convolution
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1, stride=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class InceptionNet(nn.Module):
    def __init__(self):
        super(InceptionNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x