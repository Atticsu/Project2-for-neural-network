import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
        # First set of layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Second set of layers
        x1 = self.maxpool1(x)
        x2 = self.conv4(x)
        x = torch.cat((x1, x2), dim=1)
        
        # Third set of layers
        x1 = self.conv5(x)
        x1 = self.conv6(x1)
        x2 = self.conv7(x)
        x2 = self.conv8(x2)
        x2 = self.conv9(x2)
        x2 = self.conv10(x2)
        x = torch.cat((x1, x2), dim=1)
        
        # Fourth set of layers
        x1 = self.maxpool2(x)
        x2 = self.conv11(x)
        x = torch.cat((x1, x2), dim=1)
        
        return x



class InceptionA(nn.Module):
    def __init__(self, input_channels=384):
        super(InceptionA, self).__init__()
        self.one_by_one_1 = nn.Conv2d(input_channels, 64, kernel_size=1)
        self.three_by_three_1 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.one_by_one_2 = nn.Conv2d(input_channels, 96, kernel_size=1)
        self.three_by_three_2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.one_by_one_3 = nn.Conv2d(input_channels, 64, kernel_size=1)
        self.avg_pooling = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.one_by_one_4 = nn.Conv2d(input_channels, 96, kernel_size=1)

    def forward(self, x):
        batch1 = self.one_by_one_1(x)
        batch1 = self.three_by_three_1(batch1)
        batch2 = self.one_by_one_2(batch2)
        batch2 = self.three_by_three_2(batch2)
        batch2 = self.three_by_three_2(batch2)
        batch3 = self.one_by_one_3(x)
        batch4 = self.avg_pooling(x)
        batch4 = self.one_by_one_4(batch4)
        # 合并所有分支
        return torch.cat((batch1, batch2, batch3, batch4), dim = 1)



class InceptionB(nn.Module):
    def __init__(self, input_channels=1024):
        super(InceptionA, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 192, kernel_size=1, padding=1)
        self.conv2 = nn.Conv2d(192, 224, kernel_size=(1, 7), padding=(0, 3))
        self.conv3 =  nn.Conv2d(224, 256, kernel_size=(1, 7), padding=(0, 3))
        
        self.conv4 = nn.Conv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.conv5 = nn.Conv2d(192, 224, kernel_size=(7, 1), padding=(3, 0))
        self.conv6 = nn.Conv2d(224, 224, kernel_size=(1, 7), padding=(0, 3))
        self.conv7 = nn.Conv2d(224, 256, kernel_size=(7, 1), padding=(3, 0))

        self.conv3_1x1_proj = nn.Conv2d(input_channels, 384, kernel_size=1)
        
        # average pooling
        self.avg_pooling = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_1x1_proj = nn.Conv2d(input_channels, 128, kernel_size=1)
        

    def forward(self, x):
        #batch1
        batch1 = self.avg_pooling(x)
        batch1 = self.conv_1x1_proj(batch1)
        #batch2
        batch2 = self.conv3_1x1_proj(x)
        #batch3
        batch3 = self.conv1(x)
        batch3 = self.conv2(batch3)
        batch3 = self.conv3(batch3)
        #batch4
        batch4 = self.conv1(x) 
        batch4 = self.conv4(batch4)
        batch4 = self.conv5(batch4)
        batch4 = self.conv6(batch4)
        batch4 = self.conv7(batch4)
        return torch.cat((batch1, batch2, batch3, batch4), dim = 1)


class InceptionC(nn.Module):
    def __init__(self, input_channels=1536):
        super(InceptionA, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 384, kernel_size=1, padding=1)
        self.conv2 = nn.Conv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))
        self.conv3 =  nn.Conv2d(224, 256, kernel_size=(3, 1), padding=(1, 0))
        
        self.conv4 = nn.Conv2d(384, 448, kernel_size=(1, 3), padding=(0, 1))
        self.conv5 = nn.Conv2d(448, 512, kernel_size=(3, 1), padding=(1, 0))
        self.conv6 = nn.Conv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))
        self.conv7 =  nn.Conv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))

        self.conv3_1x1_proj = nn.Conv2d(input_channels, 256, kernel_size=1)
        
        # average pooling
        self.avg_pooling = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_1x1_proj = nn.Conv2d(input_channels, 256, kernel_size=1)
        

    def forward(self, x):
        #batch1
        batch1 = self.avg_pooling(x)
        batch1 = self.conv_1x1_proj(batch1)
        #batch2
        batch2 = self.conv3_1x1_proj(x)
        #batch3
        batch3 = self.conv1(x)
        batch3_1 = self.conv2(batch3)
        batch3_2 = self.conv3(batch3)
        batch3 = torch.cat((batch3_1,batch3_2),dim=1)
        #batch4
        batch4 = self.conv1(x) 
        batch4 = self.conv4(batch4)
        batch4 = self.conv5(batch4)
        batch4_1 = self.conv6(batch4)
        batch4_2 = self.conv7(batch4)
        batch4 = torch.cat((batch4_1,batch4_2),dim=1)
        return torch.cat((batch1, batch2, batch3, batch4), dim = 1)    

class ReductionA(nn.Module):
    def __init__(self, input_channels=384):
        super(ReductionA, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(input_channels, 384, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels, 192, kernel_size=1, padding=1)
        self.conv3 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(192, 256, kernel_size=3, padding=1)

    def forward(self, x):
        batch1 = self.maxpool(x)
        batch2 = self.conv1(x)
        batch3 = self.conv2(x)
        batch3 = self.conv3(batch3)
        batch3 = self.conv4(batch3)
        return torch.cat((batch1, batch2, batch3), 1)


class ReductionB(nn.Module):
    def __init__(self, input_channels = 1024):
        super(ReductionB, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(input_channels, 192, kernel_size=1, padding=1)
        self.conv2 = nn.Conv2d(192, 192, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(input_channels, 256, kernel_size=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(1,7), padding=(0,3))
        self.conv5 = nn.Conv2d(256, 320, kernel_size=(7,1), padding=(3,0))
        self.conv6 = nn.Conv2d(320, 320, kernel_size=3, padding=1)

    def forward(self, x):
        batch1 = self.maxpool(x)
        batch2 = self.conv1(x)
        batch2 = self.conv2(batch2)
        batch3 = self.conv3(x)
        batch3 = self.conv4(batch3)
        batch3 = self.conv5(batch3)
        batch3 = self.conv6(batch3)
        return torch.cat((batch1, batch2, batch3), 1)


class InceptionV4(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionV4, self).__init__()
        self.stem = nn.Sequential(
            Stem()
        )
        
        self.inception_a = nn.Sequential(
            InceptionA(384),
            InceptionA(384),
            InceptionA(384),
            InceptionA(384)
        )
        
        self.reduction_a = ReductionA(384)
        
        self.inception_b = nn.Sequential(
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024)
        )
        
        self.reduction_b = ReductionB(1024)
        
        self.inception_c = nn.Sequential(
            InceptionC(1536),
            InceptionC(1536),
            InceptionC(1536)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(1536, num_classes) #缩放到10类

        
    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)

model = InceptionV4(num_classes=10).to(device)

