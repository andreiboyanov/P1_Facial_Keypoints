import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # From here: https://arxiv.org/pdf/1710.00977.pdf 
        # (shapes are changed according our input image size [224, 224])
        #
        #Layer Number Layer Name                                    Layer Shape
        # 1                                 Input1                    (1, 224, 224)
        # 2                                 Convolution2d1            (32, 221, 221)
        # 3                                 Activation1               (32, 221, 221)
        # 4                                 Maxpooling2d1             (32, 110, 110)
        # 5                                 Dropout1                  (32, 110, 110)
        # 6                                 Convolution2d2            (64, 108, 108)
        # 7                                 Activation2               (64, 108, 108)
        # 8                                 Maxpooling2d2             (64, 54, 54)
        # 9                                 Dropout2                  (64, 54, 54)
        # 10                                Convolution2d3            (128, 53, 53)
        # 11                                Activation3               (128, 53, 53)
        # 12                                Maxpooling2d3             (128, 26, 26)
        # 13                                Dropout3                  (128, 26, 26)
        # 14                                Convolution2d4            (256, 26, 26)
        # 15                                Activation4               (256, 26, 26)
        # 16                                Maxpooling2d4             (256, 13, 13)
        # 17                                Dropout4                  (256, 13, 13)
        # 18                                Flatten1                  (43264)
        # 19                                Dense1                    (43264)
        # 20                                Activation5               (43264)
        # 21                                Dropout5                  (43264)
        # 22                                Dense2                    (1000)
        # 23                                Activation6               (1000)
        # 24                                Dropout6                  (1000)
        # 25                                Dense3                    (136)
        # Table I
        # NAIMISHNET LAYER-WISE ARCHITECTURE
        #
        # Layer Name           Number of Filters            Filter Shape
        # Convolution2d1          32                                            (4, 4)
        # Convolution2d2          64                                          (3, 3)
        # Convolution2d3          128                                        (2, 2)
        # Convolution2d4          256                                       (1, 1)
        # Table II
        # FILTER DETAILS OF CONVOLUTION2D LAYERS

        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.convolution_1 = nn.Conv2d(1, 32, 4)
        self.convolution_2 = nn.Conv2d(32, 64, 3)
        self.convolution_3 = nn.Conv2d(64, 128, 2)
        self.convolution_4 = nn.Conv2d(128, 256, 1)
        self.activation_1_5 = F.elu
        self.activation_6 = F.relu
        self.pooling = nn.MaxPool2d(2, 2)
        self.drop_out_1 = nn.Dropout(p=0.1)
        self.drop_out_2 = nn.Dropout(p=0.2)
        self.drop_out_3 = nn.Dropout(p=0.3)
        self.drop_out_4 = nn.Dropout(p=0.4)
        self.drop_out_5 = nn.Dropout(p=0.5)
        self.drop_out_6 = nn.Dropout(p=0.6)
        self.dense_1 = nn.Linear(43264, 1000)
        self.dense_2 = nn.Linear(1000, 1000)
        self.dense_3 = nn.Linear(1000, 136)
        
    def forward(self, x):
        x = self.convolution_1(x)
        x = self.activation_1_5(x)
        x = self.pooling(x)
        x = self.drop_out_1(x)
        x = self.convolution_2(x)
        x = self.activation_1_5(x)
        x = self.pooling(x)
        x = self.drop_out_2(x)
        x = self.convolution_3(x)
        x = self.activation_1_5(x)
        x = self.pooling(x)
        x = self.drop_out_3(x)
        x = self.convolution_4(x)
        x = self.activation_1_5(x)
        x = self.pooling(x)
        x = self.drop_out_4(x)

        x = x.view(x.size(0), -1)
        x = self.dense_1(x)
        x = self.activation_1_5(x)
        x = self.drop_out_5(x)
        x = self.dense_2(x)
        x = self.activation_6(x)
        x = self.dense_3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x