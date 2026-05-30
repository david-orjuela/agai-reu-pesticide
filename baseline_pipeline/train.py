'''
AGAI_REU_2026
David Orjuela
Undergraduate Student under Dr. Chen
Training Orchestration
'''
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from baseline_model import frozen_resnet
from dataset import agai_correct_v3

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD =  [0.229, 0.224, 0.225]

    # Hyperparameters
    LEARNING_RATE = 0.001
    EPOCHS = 10

    model = frozen_resnet(num_bins=3)
    model.to(device=device)
    
    train_transform = transforms.Compose([ # or torch.nn.Sequential?
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)    
    ])

    val_transform = transforms.Compose([ # or torch.nn.Sequential?
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)    
    ])

    dataset = agai_correct_v3("/home/davidorjuela/dev/agai-reu-2026/datasets/agai_correct/v3/master_icp.csv", transform=train_transform) # string path of master_icp.csv

    # consider scikit learn for stratified splits later
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )

    # Does validation receive a toTensor transforms only?
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False,
        generator=torch.Generator().manual_seed(42)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=LEARNING_RATE, momentum=0.9)

    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Val Acc: {accuracy:.2f}%")

        # Save Checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, f"checkpoints/checkpoint_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()


# ONLY IF POOR RESULTS:
# Orient all leaves to be pointing in the same direction
# Considering: segment -> estimate major axis -> infer tip vs base using shape cues -> rotate/flip to canonical direction



