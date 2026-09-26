"""1024x1024 RGB -> 256 CNN features -> three logits; no normalization/pooling/dropout."""
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self, num_outputs=3):
        super().__init__()
        layers = []
        in_channels = 3
        # Spatial sizes: 512, 256, 128, 64, 32, 16, 8, 4, 2, 1.
        # Final shape: (B, 256, 1, 1), giving 256 flattened features.
        for channels in (8, 8, 16, 16, 32, 32, 64, 64, 128, 256):
            layers.extend([
                nn.Conv2d(in_channels, channels, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            in_channels = channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, num_outputs),
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Hidden layers are followed by ReLU: preserve forward signal scale.
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Output logits have no ReLU activation.
        nn.init.xavier_uniform_(self.head[-1].weight)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 1024, 1024):
            raise ValueError('Expected (B, 3, 1024, 1024) images.')
        return self.head(self.features(images))


def custom_cnn(num_bins=3):
    return CustomCNN(num_outputs=num_bins)
