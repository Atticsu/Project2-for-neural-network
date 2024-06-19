import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
# 模型改进，增加残差链接
# 残差连接时进行残差缩放 
class Inception_ResNet_Module(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj, scale_factor=0.2):
        super(Inception_ResNet_Module, self).__init__()
        
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

        self.output_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj
        if self.output_channels != in_channels:
            self.residual_conv = nn.Conv2d(in_channels, self.output_channels, kernel_size=1, stride=1)
        else:
            self.residual_conv = None
        
        self.scale_factor = scale_factor

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        concatenated = torch.cat(outputs, 1)
        
        # Residual connection
        if self.residual_conv:
            x = self.residual_conv(x)
        
        # Scale the residual
        output = concatenated + self.scale_factor * x
        return output
    
class InceptionResNet(nn.Module):
    def __init__(self):
        super(InceptionResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception3a = Inception_ResNet_Module(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_ResNet_Module(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = Inception_ResNet_Module(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_ResNet_Module(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_ResNet_Module(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_ResNet_Module(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_ResNet_Module(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.inception5a = Inception_ResNet_Module(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_ResNet_Module(832, 384, 192, 384, 48, 128, 128)
        
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