# This file implements the AlexNet model.

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.dropout = nn.Dropout() # default dropout probability is 0.5, following AlexNet paper
        self.pool = nn.MaxPool2d(3, 2) # Section 4.4: kernel size is 3, stride is 2
        # alpha is 0.0005 instead of 0.0001 due to a slight discrepancy between the paper's function definition and the PyTorch version
        self.lrn = nn.LocalResponseNorm(5, 0.0001 * 5, 0.75, 2) 

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(6 * 6 * 256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.output = nn.Linear(4096, num_classes)

    def forward(self, x):
        # Input is 224x224x3 in the paper, but this would give 54.25 x 54.25 x 96
        # at the output of the first layer. If the input 227x227x3 instead, you get 55 x 55 x 96
        # Layers 1-2: (Convolution, local response normalization, and max-pooling)^2
        x = self.pool(F.relu(self.lrn(self.conv1(x))))
        x = self.pool(F.relu(self.lrn(self.conv2(x))))

        # Layers 3-5: (Convolution^3, max-pooling)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))

        # Layers 6-7: (Fully connected, dropout)^2
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))

        # Layer 8: (Linear + softmax)
        x = F.softmax(self.output(x))
        return x
