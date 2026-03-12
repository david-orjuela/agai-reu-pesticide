'''
AGAI_REU_2026
David Orjuela
Undergraduate Student under Dr. Chen

Multi-class Classification Problem
Goal: classify the residual into different ranges, e.g. 0 - 100ml; 100 - 200ml; etc.
(N, C, H, W)
N = len(dataset)
C = 3 (RGB)
H = 
'''

import torch
import torch.nn as nn

class baselineCNN(nn.Module):
    def __init__(self, channels = 3):
        super().__init__()

        

    def forward(self):
        pass

