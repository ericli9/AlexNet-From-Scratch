# This file implements the architecture of AlexNet.

import torch
import torch.nn as nn

# input is 224x224x3, apparently this is wrong and should be 227x227x3 instead

# Layers 1-2: (Convolution, local response normalization, and max-pooling)^2

# Layers 3-5: (Convolution^3, max-pooling)


# Layers 6-7: (Fully connected, dropout)^2
# size of each layer is 4096

# Layer 8: (Linear + softmax)
# size is 1000
# add a softmax

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.dropout = nn.Dropout() # default dropout probability is 0.5, following AlexNet paper

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5) # stride should be something


        self.conv3 = nn.Conv2d(256, 384, kernel_size=3)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3)

        # there is max pooling here

        self.fc1 = nn.Linear(13 * 13 * 256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.output = nn.Linear(4096, num_classes)

    def forward(self, x):