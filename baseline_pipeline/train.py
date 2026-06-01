'''
AGAI_REU_2026
David Orjuela
Undergraduate Student under Dr. Chen
Training Orchestration
'''
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from collections import Counter

from baseline_model import frozen_resnet
from dataset import agai_correct_v3

# Helper for printing class counts
def print_class_counts(name, subset):
    labels = [subset.dataset[i][1] for i in subset.indices]
    counts = Counter(labels)

    label_names = {
        0: "low",
        1: "medium",
        2: "high",
    }

    print(f"\n{name} class counts:")
    for label_id in [0, 1, 2]:
        print(f"  {label_names[label_id]} ({label_id}): {counts[label_id]}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently on {device}")
    
    torch.manual_seed(42)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD =  [0.229, 0.224, 0.225]

    # Hyperparameters
    LEARNING_RATE = 0.001
    EPOCHS = 30
    DEBUG = False

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
    
    train_dataset = agai_correct_v3("/home/david/dev/agai-reu-2026/datasets/agai_correct/v3/master_icp.csv", transform=train_transform)
    val_dataset = agai_correct_v3("/home/david/dev/agai-reu-2026/datasets/agai_correct/v3/master_icp.csv", transform=val_transform)

    all_idx = list(range(len(train_dataset)))
    labels = [train_dataset[i][1] for i in all_idx]

    train_idx, val_idx = model_selection.train_test_split(
        all_idx,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    
    print_class_counts("Train", train_subset)
    print_class_counts("Validation", val_subset)
    print()

    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=4,
        shuffle=False,
        generator=torch.Generator().manual_seed(42)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=LEARNING_RATE, momentum=0.9)

    best_val_loss = float("inf")
    best_epoch = float("inf")

    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            _, train_predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (train_predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * train_correct / train_total
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        val_running_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                val_loss = criterion(outputs, labels)

                val_running_loss += val_loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        accuracy = 100 * correct / total

        print(
        f"Epoch [{epoch+1}/{EPOCHS}] - "
        f"Train Loss: {epoch_loss:.4f} - "
        f"Train Acc: {train_accuracy:.2f}% - "
        f"Val Loss: {val_epoch_loss:.4f} - "
        f"Val Acc: {accuracy:.2f}%"
    )

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss

            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": epoch_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_epoch_loss,
                "val_accuracy": accuracy,
            }, "checkpoints/best_checkpoint.pt")

            if DEBUG: print(f"Saved new best checkpoint with val loss {best_val_loss:.4f}")
            else: best_epoch = epoch + 1
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    print("\nFinal validation confusion matrix:")
    print("Rows = true labels, columns = predicted labels")
    print("Labels: 0=low, 1=medium, 2=high")
    print(cm)

    if not DEBUG: print(f"Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}")

if __name__ == "__main__":
    main()

# ONLY IF POOR RESULTS:
# Orient all leaves to be pointing in the same direction
# Considering: segment -> estimate major axis -> infer tip vs base using shape cues -> rotate/flip to canonical direction