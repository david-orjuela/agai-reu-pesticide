'''
AGAI_REU_2026
David Orjuela
Undergraduate Student under Dr. Chen

Multi-class Classification Problem
Goal: classify the residual into different ranges, e.g. low, medium, high ICP-derived residue density

Input Image: [] x []

(N, C, H, W)
N = len(dataset)
C = 3 (RGB)
H = 
'''

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import pipeline

REPO_DIR = ""
def frozen_resnet(num_bins=3):
    model_pretrained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Baseline 1. Frozen Resnet
    for param in model_pretrained.parameters():
        param.requires_grad = False

    num_ftrs = model_pretrained.fc.in_features # 2048
    
    model_pretrained.fc = nn.Linear(num_ftrs, num_bins)

    return model_pretrained

def frozen_dinoV3():
    dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=>)

    return 
