## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

def computeSz(sz,k, stride, pad = 0):
    return int(int(sz-k+2*pad)/stride + 1)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.c1=nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        sz=computeSz(224,11,4,2)
        self.pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        sz=computeSz(sz,3,2,0)
        self.rl = nn.ReLU(inplace=True)
        self.c2=nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        sz=computeSz(sz,5,1,2)
        sz=computeSz(sz,3,2,0)
        self.c3=nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        sz=computeSz(sz,3,1,1)
        self.c4=nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        sz=computeSz(sz,3,1,1)
        self.c5=nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        sz=computeSz(sz,3,1,1)
        sz=computeSz(sz,3,2,0)
        in_features=sz*sz*256

        self.drop= nn.Dropout(p=0.5, inplace=False)
        self.fc1 = nn.Linear(in_features=in_features, out_features=4096, bias=True)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc3 = nn.Linear(in_features=4096, out_features=136, bias=True)
       
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        #(0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        #(1): ReLU(inplace=True)
        #(2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        x = self.pool(self.rl(self.c1(x)))
            
        #(3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #(4): ReLU(inplace=True)
        #(5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        x = self.pool(self.rl(self.c2(x)))   
        
        #(6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #(7): ReLU(inplace=True)
        x = self.rl(self.c3(x))
        
        #(8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #(9): ReLU(inplace=True)
        x = self.rl(self.c4(x))
        #(10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #(11): ReLU(inplace=True)
        #(12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        x = self.pool(self.rl(self.c5(x)))
        
        #(0): Dropout(p=0.5, inplace=False)
        x = self.drop(x)
        #(1): Linear(in_features=9216, out_features=4096, bias=True)
        #(2): ReLU(inplace=True)
        x = F.relu(self.fc1(x))
        #(3): Dropout(p=0.5, inplace=False)
        x = self.drop(x)
        #(4): Linear(in_features=4096, out_features=4096, bias=True)
        #(5): ReLU(inplace=True)
        x = F.relu(self.fc2(x))
        #(6): Linear(in_features=4096, out_features=1000, bias=True)
        self.fc3(x)
       
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

class Net2(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 11)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv1_bn = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 7)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 192, 3)  
        
        sz = computeSz(224, 11, 1) # conv 1
        sz = computeSz(sz, 2, 2) # maxpool
        sz = computeSz(sz, 7, 1) # conv2
        sz = computeSz(sz, 2, 2) # maxpool
        sz = computeSz(sz, 5, 1) # conv3
        sz = computeSz(sz, 2, 2) # maxpool
        sz = computeSz(sz, 3, 1) # conv4
        sz = computeSz(sz, 2, 2) # maxpool
        
        self.drop = nn.Dropout(p=0.4)
            
        self.fc1 = nn.Linear(sz*sz*192, 1200)
        
        self.fc2 = nn.Linear(1200, 800)
        
        self.fcf = nn.Linear(800, 136)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_bn(self.pool(F.relu(self.conv1(x))))
        x = self.conv2_bn(self.pool(F.relu(self.conv2(x))))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.fcf(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
class Net1(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 31)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 15)
       
        self.conv3 = nn.Conv2d(64, 128, 7)
        
        self.conv4 = nn.Conv2d(128, 192, 3)  
        
        sz = computeSz(224, 31, 1) # conv 1
        sz = computeSz(sz, 2, 2) # maxpool
        sz = computeSz(sz, 15, 1) # conv2
        sz = computeSz(sz, 2, 2) # maxpool
        sz = computeSz(sz, 7, 1) # conv3
        sz = computeSz(sz, 2, 2) # maxpool
        sz = computeSz(sz, 3, 1) # conv4
        sz = computeSz(sz, 2, 2) # maxpool
        
        self.fc1 = nn.Linear(sz*sz*192, 800)
        
        self.fc1_drop = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(800, 400)
        
        self.fc2_drop = nn.Dropout(p=0.3)
        
        self.fcf = nn.Linear(400, 136)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fcf(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

