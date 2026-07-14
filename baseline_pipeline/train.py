'''
AGAI_REU_2026
David Orjuela
Undergraduate Student under Dr. Chen
Training Orchestration
'''
import os
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

from scipy.stats import pearsonr
from matplotlib import pyplot as plt

from collections import Counter

from baseline_pipeline.baseline_model import frozen_resnet, frozen_dinoV3
from dataset import agai_correct_v3

# Helper for printing class counts using stratification bins
def print_class_counts_from_indices(name, indices, stratify_labels):
    labels = [stratify_labels[i] for i in indices]
    counts = Counter(labels)

    label_names = {
        0: "low",
        1: "medium",
        2: "high",
    }

    print(f"\n{name} stratification-bin counts:")
    for label_id in [0, 1, 2]:
        print(f"  {label_names[label_id]} ({label_id}): {counts[label_id]}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently on {device}")

torch.manual_seed(42)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD =  [0.229, 0.224, 0.225]

CSV_PATH = "/home/david/dev/agai-reu-pesticide/datasets/agai_correct/v3/master_icp.csv"

# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 30
DEBUG = False

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

train_dataset = agai_correct_v3(CSV_PATH, transform=train_transform)
val_dataset = agai_correct_v3(CSV_PATH, transform=val_transform)

all_idx = list(range(len(train_dataset)))

# For Bin Classification:
# labels = [train_dataset[i][1] for i in all_idx]

# For ICP Residue Regression:
# Dataset now returns mg_cm2 as target, so we cannot use train_dataset[i][1]
# for stratification because that is continuous. Instead, use residue_bin
# from the CSV only for StratifiedKFold splitting.
label_to_int = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

stratify_labels = [
    label_to_int[train_dataset.icp_data["residue_bin"].iloc[i]]
    for i in all_idx
]

def run_experiment(train_idx, val_idx, fold):
    # model = frozen_resnet(num_bins=3).to(device=device)  # Bin Classification
    # model = frozen_resnet(num_bins=1).to(device=device)    # ICP Residue Regression
    model = frozen_dinoV3(num_bins=1).to(device=device)    # DINOv3
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    print_class_counts_from_indices("Train", train_idx, stratify_labels)
    print_class_counts_from_indices("Validation", val_idx, stratify_labels)
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
        shuffle=False
    )

    # criterion = nn.CrossEntropyLoss()  # Classification loss (bins)
    criterion = nn.SmoothL1Loss()       # ICP Residue Regression loss

    optimizer = torch.optim.SGD(
        model.fc.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9
    )

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_val_rmse = float("inf")
    best_val_r2 = float("-inf")
    
    best_epoch = -1

    best_preds_np = None
    best_targets_np = None
    best_corr = None
    best_baseline_mae = None

    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss = 0.0

        # train_correct = 0  # Bin Classification
        # train_total = 0    # Bin Classification

        train_preds = []
        train_targets = []

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.float().to(device)

            optimizer.zero_grad()

            # outputs = model(images)              # Bin Classification
            outputs = model(images).squeeze(1)     # ICP Residue Regression

            # _, train_predicted = torch.max(outputs, 1)  # Bin Classification
            # train_total += targets.size(0)              # Bin Classification
            # train_correct += (train_predicted == targets).sum().item()

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            train_preds.extend(outputs.detach().cpu().tolist())
            train_targets.extend(targets.detach().cpu().tolist())

        epoch_loss = running_loss / len(train_loader.dataset)

        # train_accuracy = 100 * train_correct / train_total  # Bin Classification
        train_preds_np = np.array(train_preds)
        train_targets_np = np.array(train_targets)
        train_mae = np.mean(np.abs(train_preds_np - train_targets_np))
        train_rmse = np.sqrt(np.mean((train_preds_np - train_targets_np) ** 2))

        # Validation
        model.eval()
        val_running_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.float().to(device)

                # outputs = model(images)          # Bin Classification
                outputs = model(images).squeeze(1) # ICP Residue Regression

                val_loss = criterion(outputs, targets)
                val_running_loss += val_loss.item() * images.size(0)

                all_preds.extend(outputs.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        val_epoch_loss = val_running_loss / len(val_loader.dataset)

        val_preds_np = np.array(all_preds)
        val_targets_np = np.array(all_targets)

        val_mae = np.mean(np.abs(val_preds_np - val_targets_np))
        val_rmse = np.sqrt(np.mean((val_preds_np - val_targets_np) ** 2))
        val_r2 = r2_score(val_targets_np, val_preds_np)

        corr, _ = pearsonr(val_targets_np, val_preds_np)
        
        train_mean = np.mean(train_targets_np) # Instead of "oracle" val mean
        baseline_pred = np.full_like(val_targets_np, train_mean)

        baseline_mae = np.mean(
            np.abs(val_targets_np - baseline_pred)
        )

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] - "
            f"Train Loss: {epoch_loss:.4f} - "
            f"Train MAE: {train_mae:.4f} - "
            f"Train RMSE: {train_rmse:.4f} - "
            f"Val Loss: {val_epoch_loss:.4f} - "
            f"Val MAE: {val_mae:.4f} - "
            f"Val RMSE: {val_rmse:.4f} - "
            f"Val R²: {val_r2:.4f}"
        )

        print(f"Target mean: {np.mean(val_targets_np):.4f}")
        print(f"Target std:  {np.std(val_targets_np):.4f}")

        print(f"Pred mean:   {np.mean(val_preds_np):.4f}")
        print(f"Pred std:    {np.std(val_preds_np):.4f}")
        
        print(f"Pearson r: {corr:.4f}")
        print(f"Baseline MAE: {baseline_mae:.4f}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            best_val_r2 = val_r2
            
            best_epoch = epoch + 1

            best_preds_np = val_preds_np.copy()
            best_targets_np = val_targets_np.copy()
            best_corr = corr
            best_baseline_mae = baseline_mae

            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": epoch_loss,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "val_loss": val_epoch_loss,
                "val_mae": best_val_mae,
                "val_rmse": best_val_rmse,
                "val_r2": best_val_r2,
            }, f"checkpoints/best_checkpoint_fold_{fold}.pt")

            if DEBUG:
                print(f"Saved new best checkpoint with val loss {best_val_loss:.4f}")

    print(f"\nBest Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    print(f"Best Val MAE: {best_val_mae:.4f}")
    print(f"Best Val RMSE: {best_val_rmse:.4f}")
    print(f"Best Val R²: {best_val_r2:.4f}")

    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.scatter(best_targets_np, best_preds_np)

    min_val = min(best_targets_np.min(), best_preds_np.min())
    max_val = max(best_targets_np.max(), best_preds_np.max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Ground Truth mg/cm²")
    plt.ylabel("Predicted mg/cm²")
    plt.title(f"Fold {fold + 1}: Prediction vs Ground Truth")
    plt.tight_layout()
    plt.savefig(f"plots/fold_{fold + 1}_pred_vs_gt.png", dpi=300)
    plt.close()

    return {
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "best_val_rmse": best_val_rmse,
        "best_val_r2": best_val_r2,
        "best_corr": best_corr,
        "best_baseline_mae": best_baseline_mae,
    }


def main():
    val_losses = []
    val_maes = []
    val_rmses = []
    val_r2s = []
    val_corrs = []

    baseline_maes = []

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_idx, stratify_labels)):
        print(f"\n========== Fold {fold + 1} ==========")

        result = run_experiment(train_idx, val_idx, fold)

        val_losses.append(result["best_val_loss"])
        val_maes.append(result["best_val_mae"])
        val_rmses.append(result["best_val_rmse"])
        val_r2s.append(result["best_val_r2"])
        val_corrs.append(result["best_corr"])
        
        baseline_maes.append(result["best_baseline_mae"])

    print("\n========== Cross-Validation Summary ==========")
    print(f"Val Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"Val MAE:  {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")
    print(f"Val RMSE: {np.mean(val_rmses):.4f} ± {np.std(val_rmses):.4f}")
    print(f"Val R²: {np.mean(val_r2s):.4f} ± {np.std(val_r2s):.4f}")
    print(f"Val Pearson r: {np.mean(val_corrs):.4f} ± {np.std(val_corrs):.4f}")
    print(f"Baseline MAE:  {np.mean(baseline_maes):.4f} ± {np.std(baseline_maes):.4f}")

if __name__ == "__main__":
    main()

# ONLY IF POOR RESULTS:
# Orient all leaves to be pointing in the same direction
# Considering: segment -> estimate major axis -> infer tip vs base using shape cues -> rotate/flip to canonical direction