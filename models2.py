## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
          ## model details
        
        ## Layer Name Number of Filters Filter Shape
        ##Convolution2d1 32 (4, 4)
        ##Convolution2d2 64 (3, 3)
        ##Convolution2d3 128 (2, 2)
        ##Convolution2d4 256 (1, 1)
        
        #Dropout probability is increased from 0.1 to (...) from with a step size of 0.1.
        
        #####
        #
        # input image width/height, W, 
        # minus the filter size, F, divided by the stride, S, all + 1. 
        # The equation looks like: output_dim = (W-F)/S + 1
        # at the max pool out, the value are rounded to the lower bound (12.5 ->12)
        #####
        
        ## max pooling is always the same 
        #self.pool = nn.MaxPool2d(2, 2)
        
        # 1 - input image channel (grayscale), 32 output channels/feature maps, 4x4 square convolution kernel
        # max pooling + 10% droupout
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p = 0.1)
        )
       
        
        # 2 - 64 output channels/feature maps, 3x3 square convolution kernel
        # 2x2 max pooling with 20% droupout
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p = 0.2)
        )
        
        # 3 - 128 output channels/feature maps, 2x2 square convolution kernel
        # 2x2 max pooling with 30% droupout
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p = 0.3)
        )
        
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 258, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p = 0.4)
        )
        
        # Fully Connected Layers
        # 128 outputs * the 2*2 filtered/pooled map size
        self.fc1 = nn.Linear(258*13*13, 1000) # i dont understand this size of input !!!!
        self.dropout4 = nn.Dropout(p = 0.5)
        
        self.fc2 = nn.Linear(1000, 1000)
        self.dropout5 = nn.Dropout(p = 0.6)
        
        # this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        self.fc3 = nn.Linear(1000, 136)
       

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # x = self.pool(F.relu(self.conv1(x)))
        #print("initinal shape : " + str(x.shape))
        x = self.layer1(x)
        #print("output layer1 : " + str(x.shape))
        x = self.layer2(x)
        #print("output layer2 : " + str(x.shape))
        x = self.layer3(x)
        #print("output layer3 : " + str(x.shape))
        x = self.layer4(x)
        #print("output layer4 : " + str(x.shape))
        
        #print(x.shape) #-> this should be [batch size, last layer output size,
        #flatten
        x = x.view(x.size(0), -1)
        #print("output flatten : " + str(x.shape))
        
        #fully connected
        x = self.dropout4(self.fc1(x))
        x = self.dropout5(self.fc2(x))
        x = self.fc3(x)
       
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
