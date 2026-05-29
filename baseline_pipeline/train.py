'''
AGAI_REU_2026
David Orjuela
Undergraduate Student under Dr. Chen

Leaf Pre-processing
Goal: Perform the following:
- crop
- resize
- normalize
- augment

Assuming dataset contains images of size (W, H) = (3024, 4032).
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from baseline_model import frozen_resnet
from dataset import agai_correct_v3

EPOCHS = 10

model = frozen_resnet(num_bins=3)
model.to(device=torch.cuda)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

data = csv_to_arr
labels = csv_to_arr

dataset = agai_correct_v3(data, labels)
train_loader = DataLoader(
    dataset=dataset
)

for epoch in range(EPOCHS):
    pass


# Crop and Scale based off max OR average size.


# ONLY IF POOR RESULTS:
# Orient all leaves to be pointing in the same direction
# Considering: segment -> estimate major axis -> infer tip vs base using shape cues -> rotate/flip to canonical direction



