import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# モデルの定義
class BuildingBlock(nn.Module):
    """
    H(x) = BuilidingBlock(x) + x
    """
    def __init__(self, in_channels, med_channels, out_channels, is_downsample=False):
        super().__init__()

        if is_downsample == True:
            stride = 2
        else:
            stride = 1
        self.m_1 = nn.Conv2d(in_channels, med_channels, kernel_size=3, stride=stride, padding=1)
        self.m_2 = nn.Conv2d(med_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.m_1(x)
        out = F.relu(out)
        out = self.m_2(out)
        
        return out

# RESNET-LIKE Structure
class RESNETLIKE(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 5, 2, 1)
        self.maxpool = nn.MaxPool2d(3, 2, 0, 1)
        self.conv21 = BuildingBlock(in_channels=64, med_channels=64, out_channels=64)
        self.conv22 = BuildingBlock(in_channels=64, med_channels=64, out_channels=64)
        self.conv31 = BuildingBlock(in_channels=64, med_channels=128, out_channels=128, is_downsample=True)
        self.conv32 = BuildingBlock(in_channels=128, med_channels=128, out_channels=128)
        self.conv41 = BuildingBlock(in_channels=128, med_channels=256, out_channels=256, is_downsample=True)
        self.conv42 = BuildingBlock(in_channels=256, med_channels=256, out_channels=256)
        self.conv51 = BuildingBlock(in_channels=256, med_channels=512, out_channels=512, is_downsample=True)
        self.conv52 = BuildingBlock(in_channels=512, med_channels=512, out_channels=512)        

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

        self.conv11_1 = self.conv11(64,128)
        self.conv11_2 = self.conv11(128,256)
        self.conv11_3 = self.conv11(256,512)

        self.bn64  = nn.BatchNorm2d(64)
        self.bn128  = nn.BatchNorm2d(128)
        self.bn256  = nn.BatchNorm2d(256)
        self.bn512  = nn.BatchNorm2d(512)    

    def conv11(self, in_channels, out_channels):
            """
            入力xの調整用
            if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer.
            だそうなので，画像のサイズが1/2になり，フィルタの数が２倍になっていく．
            そのため，画像のサイズを調整し，チャンネル数も調整する必要がある．
            画像のサイズはstrideを２にすることで，1/2に，チャンネル数は1*1のカーネルをin_channelsの２倍用いて，畳み込みをすればよい．
            """
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn64(x)
        x = self.maxpool(x)
        x = F.relu(x)
        x = self.bn64(x)
        
        x = self.conv21(x) + x
        x = F.relu(x)     
        x = self.bn64(x)         
        x = self.conv22(x) + x
        x = F.relu(x)  
        x = self.bn64(x)    
        
        x = self.conv31(x) + self.conv11_1(x)
        x = F.relu(x) 
        x = self.bn128(x)        
        x = self.conv32(x) + x
        x = F.relu(x) 
        x = self.bn128(x)        
        
        x = self.conv41(x) + self.conv11_2(x)
        x = F.relu(x)         
        x = self.bn256(x)
        x = self.conv42(x) + x
        x = F.relu(x) 
        x = self.bn256(x)
        
        x = self.conv51(x) + self.conv11_3(x)
        x = F.relu(x)         
        x = self.bn512(x)        
        x = self.conv52(x) + x
        x = F.relu(x) 

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)        
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)        
        #x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# SIMPLE
class MyModel(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 5, 2, 1)
        self.maxpool = nn.MaxPool2d(3, 2, 0, 1)
        self.conv21 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv22 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv31 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv32 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv41 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv42 = nn.Conv2d(256, 256, 3, 1, 1)    

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.bn64  = nn.BatchNorm2d(64)
        self.bn128  = nn.BatchNorm2d(128)
        self.bn256  = nn.BatchNorm2d(256)
        self.bn512  = nn.BatchNorm2d(512)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = F.relu(x)
        
        x = self.conv21(x)
        x = F.relu(x)        
        x = self.conv22(x)
        x = F.relu(x)  
        
        x = self.conv31(x)
        x = F.relu(x) 
        x = self.conv32(x)
        x = F.relu(x) 
        
        x = self.conv41(x)
        x = F.relu(x)         
        x = self.conv42(x)
        x = F.relu(x) 

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)        
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)        
        #x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# SIMPLE
class MyModel_shallow(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 5, 2, 1)
        self.maxpool = nn.MaxPool2d(3, 2, 0, 1)
        self.conv21 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv22 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv31 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv32 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv41 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv42 = nn.Conv2d(256, 256, 3, 1, 1)    
        self.conv51 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)    
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.bn64  = nn.BatchNorm2d(64)
        self.bn128  = nn.BatchNorm2d(128)
        self.bn256  = nn.BatchNorm2d(256)
        self.bn512  = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn64(x)
        x = self.maxpool(x)
        x = F.relu(x)
        x = self.bn64(x)        
        
        x = self.conv21(x)
        x = F.relu(x)
        x = self.bn64(x)        
        #x = self.conv22(x)
        #x = F.relu(x)  
        
        x = self.conv31(x)
        x = F.relu(x) 
        x = self.bn128(x)        
        #x = self.conv32(x)
        #x = F.relu(x) 
        
        x = self.conv41(x)
        x = F.relu(x)     
        x = self.bn256(x)        
        #x = self.conv42(x)
        #x = F.relu(x) 

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)        
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)        
        #x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# SIMPLE
class MyModel_shallow_more(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 5, 2, 1)
        self.maxpool = nn.MaxPool2d(3, 2, 0, 1)
        self.conv21 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv22 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv31 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv32 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv41 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv42 = nn.Conv2d(256, 256, 3, 1, 1)    
        self.conv51 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)    
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.bn64  = nn.BatchNorm2d(64)
        self.bn128  = nn.BatchNorm2d(128)
        self.bn256  = nn.BatchNorm2d(256)
        self.bn512  = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn64(x)
        #x = self.maxpool(x)
        #x = F.relu(x)
        #x = self.bn64(x)        
        
        x = self.conv21(x)
        x = F.relu(x)
        x = self.bn64(x)        
        #x = self.conv22(x)
        #x = F.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)        
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)        
        #x = self.dropout(x)
        x = self.fc3(x)
        
        return x