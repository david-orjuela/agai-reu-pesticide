'''
AGAI_REU_2026
David Orjuela
Undergraduate Student under Dr. Chen

Multi-class Classification Problem
Goal: classify the residual into different ranges, e.g. low, medium, high ICP-derived residue density

Input batch: (B, 3, 256, 256)
B = current batch size
Frozen encoder output: (B, 384)
Classification output: (B, 3)
Regression output: (B, 1)
'''

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

REPO_DIR = "/home/davidorjuela/dev/dinov3"
WEIGHTS_DIR = "/home/davidorjuela/dev/agai-reu-pesticide/checkpoints/"
vits16plus_weights = "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"

def frozen_resnet(num_bins=3):
    model_pretrained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Baseline 1. Frozen Resnet
    for param in model_pretrained.parameters():
        param.requires_grad = False

    num_ftrs = model_pretrained.fc.in_features # 2048
    
    model_pretrained.fc = nn.Linear(num_ftrs, num_bins)

    return model_pretrained

class FrozenDINOv3(nn.Module):
    """Normalized CLS embedding -> trainable linear task head."""

    def __init__(self, encoder: nn.Module, num_outputs: int):
        super().__init__()
        self.encoder = encoder.requires_grad_(False)
        self.embed_dim = encoder.embed_dim
        self.head = nn.Linear(self.embed_dim, num_outputs)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("DINO expects a batch with shape (B, 3, H, W).")
        if images.shape[-2] % 16 or images.shape[-1] % 16:
            raise ValueError("DINO image dimensions must be multiples of 16.")
        # no_grad permits the head to save features for backward.
        with torch.no_grad():
            features = self.encoder.forward_features(images)["x_norm_clstoken"]
        return self.head(features)


def frozen_dinov3(num_bins=3, *, repo_dir=REPO_DIR, weights_path=None):
    """Build ViT-S+/16 with three logits, or one output for regression."""
    repo = Path(repo_dir).expanduser()
    weights = (
        Path(weights_path).expanduser()
        if weights_path is not None
        else Path(WEIGHTS_DIR) / vits16plus_weights
    )
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINO repository hubconf.py not found in {repo}")
    if not weights.is_file():
        raise FileNotFoundError(f"DINO weights not found: {weights}")
    encoder = torch.hub.load(
        str(repo), "dinov3_vits16plus", source="local", weights=str(weights)
    )
    if encoder.embed_dim != 384:
        raise ValueError(f"Expected ViT-S+/16 embedding size 384, got {encoder.embed_dim}")
    return FrozenDINOv3(encoder, num_bins)


# Keep existing imports working while using snake_case in new code.
frozen_dinoV3 = frozen_dinov3