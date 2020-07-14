## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
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
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
         # Convolutional layer-1  (W+2P-F)/S +1 = ((224+0-5)/1)+1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=0)   # input: 224X224X1, output: 220X220X32
        # Max Pooling layer -1  [(W1−F)/S]+1 = ((220-2)/2)+1    # input: 220X220X32, output: 110X110X32           
        
        # Convolutional layer 2  (W+2P-F)/S +1 = ((110+0-3)/1)+1 = 108
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=0)  # input: 110X110X32, output: 108X108X64
        # Max Pooling layer -2 [(W1−F)/S]+1 = ((108-2)/2)+1 =54 # input: 108X108X64, output: 54X54X64
        
        # Convolutional layer 3 (W+2P-F)/S +1 = ((54+0-3)/1)+1 = 52
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=0) # input: 54X54X64, output: 54X54X128
        # Max Pooling layer -3 [(W1−F)/S]+1 = ((54-2)/2)+1 =26 # input: 54X54X128, output: 26X26X128
        
        # Convolutional layer 4 (W+2P-F)/S +1 = ((26+0-3)/1)+1 = 24
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=0) # input: 26X26X128, output: 24X24X256
        # Max Pooling layer -4 [(W1−F)/S]+1 = ((24-2)/2)+1 =12   # input: 24X24X256, output: 12X12X256
        
        # Convolutional layer 5 (W+2P-F)/S +1 = ((12+0-1)/1)+1 = 12
        self.conv5 = nn.Conv2d(256, 512, 1,stride=1, padding=0) # input: 12X12X256, output: 12X12X512
        # Max Pooling layer -5 [(W1−F)/S]+1 = ((11-1)/2)+1 =6   # input: 12X12X512, output: 6X6X512
            
        # We will be using this self.pool in above 5 layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected Linear layer-1 will have inputs:6*6*512(from maxpool5) & outputs:1024 nodes
        self.fc1 = nn.Linear(6*6*512, 1024)
        # Fully connected Linear layer-2 will have inputs:1024 & outputs:136 
        # the output 136 in order to having 2 for each of the 68 keypoint (x, y) pairs
        self.fc2 = nn.Linear(1024, 136) 
        
        # We will have dropout layer (p=0.3) for each full connected layer
        self.dropout = nn.Dropout(0.2)
               

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Adding sequence of layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)                   # Dropout layer
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)                   # Dropout layer
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)                   # Dropout layer
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)                   # Dropout layer
        x = self.pool(F.relu(self.conv5(x)))
        x = self.dropout(x)                   # Dropout layer
        x = x.view(x.size(0), -1)             # flatten image input
        x = F.relu(self.fc1(x))               # Full connected layer-1 (hidden layer 1) 
        x = self.dropout(x)                   # Dropout layer
        x = self.fc2(x)                       # Full connected layer-2 (hidden layer 2)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
