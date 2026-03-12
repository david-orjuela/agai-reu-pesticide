'''
AGAI_REU_2026
David Orjuela
Undergraduate Student under Dr. Chen

Multi-class Classification Problem
Goal: classify the residual into different ranges, e.g. 0 - 100ml; 100 - 200ml; etc.

Input Image: [] x []

(N, C, H, W)
N = len(dataset)
C = 3 (RGB)
H = 
'''

import torch
import torch.nn as nn

class baselineCNN(nn.Module):
    def __init__(self, channels = 3, total_features = 0, num_bins = 5):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)

        # Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)

        # Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(total_features,128) # Need to calculate total features
        self.fc2 = nn.Linear(128, num_bins) # Need to know PPM threshold for ranges of bins, for classification

    def forward(self):
        pass

