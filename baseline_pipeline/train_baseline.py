"""Train a simple CNN to classify leaf residue bins: low, medium, high."""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


PROJECT_ROOT = Path("/home/davidorjuela/dev/agai-reu-pesticide")
CSV_PATH = PROJECT_ROOT / "anthony/master_icp.csv"
OUTPUT_DIR = PROJECT_ROOT / "anthony/cnn_results"

IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "residue_bin"
SAMPLE_ID_COLUMN = "sample_id"

CLASS_NAMES = ("low", "medium", "high")
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10
SEED = 42
NUM_WORKERS = 4


def set_seed(seed):
    """Make training reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LeafDataset(Dataset):
    """Load images and residue-bin targets from CSV rows."""

    def __init__(self, dataframe, transform):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        with Image.open(row[IMAGE_COLUMN]) as image:
            image = self.transform(image.convert("RGB"))
        return image, CLASS_TO_INDEX[row[LABEL_COLUMN]], index


class SimpleCNN(nn.Module):
    """Use four convolution layers followed by a small classifier."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        feature_size = IMAGE_SIZE // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feature_size * feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, len(CLASS_NAMES)),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_metadata():
    """Load the CSV and validate paths, labels, and IDs."""
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV does not exist: {CSV_PATH}")

    data = pd.read_csv(CSV_PATH)
    required = [IMAGE_COLUMN, LABEL_COLUMN, SAMPLE_ID_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    if data[required].isna().any().any():
        raise ValueError("CSV contains missing image paths, labels, or sample IDs.")

    data[LABEL_COLUMN] = data[LABEL_COLUMN].astype(str).str.strip().str.lower()
    invalid = sorted(set(data[LABEL_COLUMN]) - set(CLASS_NAMES))
    if invalid:
        raise ValueError(f"Unexpected residue_bin labels: {invalid}")

    missing_images = [path for path in data[IMAGE_COLUMN] if not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} images are missing. First paths:\n" + "\n".join(missing_images[:5])
        )
    if data[IMAGE_COLUMN].duplicated().any():
        raise ValueError("Duplicate image paths found in the CSV.")
    return data


def split_metadata(data):
    """Stratify whole sample_id groups into an 80/10/10 split."""
    labels_per_sample = data.groupby(SAMPLE_ID_COLUMN)[LABEL_COLUMN].nunique()
    inconsistent = labels_per_sample[labels_per_sample != 1]
    if len(inconsistent):
        raise ValueError(f"{len(inconsistent)} sample IDs map to multiple residue bins.")

    groups = data[[SAMPLE_ID_COLUMN, LABEL_COLUMN]].drop_duplicates().reset_index(drop=True)

    train_groups, heldout = train_test_split(
        groups,
        test_size=VALIDATION_FRACTION + TEST_FRACTION,
        random_state=SEED,
        stratify=groups[LABEL_COLUMN],
    )
    validation_groups, test_groups = train_test_split(
        heldout,
        test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
        random_state=SEED,
        stratify=heldout[LABEL_COLUMN],
    )

    train_ids = set(train_groups[SAMPLE_ID_COLUMN])
    validation_ids = set(validation_groups[SAMPLE_ID_COLUMN])
    test_ids = set(test_groups[SAMPLE_ID_COLUMN])

    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise RuntimeError("sample_id leakage detected between splits.")

    train = data[data[SAMPLE_ID_COLUMN].isin(train_ids)].reset_index(drop=True)
    validation = data[data[SAMPLE_ID_COLUMN].isin(validation_ids)].reset_index(drop=True)
    test = data[data[SAMPLE_ID_COLUMN].isin(test_ids)].reset_index(drop=True)

    if len(train) + len(validation) + len(test) != len(data):
        raise RuntimeError("Split does not cover the dataset exactly once.")

    for name, split in (("train", train), ("validation", validation), ("test", test)):
        if set(split[LABEL_COLUMN]) != set(CLASS_NAMES):
            missing = set(CLASS_NAMES) - set(split[LABEL_COLUMN])
            raise RuntimeError(f"{name} split is missing classes: {sorted(missing)}")

    return train, validation, test


def print_split_distribution(name, data):
    """Print image and sample_id counts for each class."""
    image_counts = data[LABEL_COLUMN].value_counts().reindex(CLASS_NAMES, fill_value=0)
    group_counts = (
        data[[SAMPLE_ID_COLUMN, LABEL_COLUMN]]
        .drop_duplicates()[LABEL_COLUMN]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )
    print(f"{name}:")
    for label in CLASS_NAMES:
        pct = 100 * group_counts[label] / group_counts.sum()
        print(
            f"  {label}: {group_counts[label]} sample_ids ({pct:.1f}%) | "
            f"{image_counts[label]} images"
        )


def make_transforms():
    """Build train and evaluation image transforms."""
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    evaluation = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        normalize,
    ])
    return train, evaluation


def make_loader(data, transform, shuffle):
    """Create a deterministic DataLoader."""
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        LeafDataset(data, transform), batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), generator=generator,
    )


def class_weights(train_data):
    """Compute inverse-frequency weights from training labels."""
    counts = train_data[LABEL_COLUMN].value_counts().reindex(CLASS_NAMES).to_numpy(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"Training split contains an empty class: {counts}")
    return torch.tensor(len(train_data) / (len(CLASS_NAMES) * counts), dtype=torch.float32)


def metrics_from_predictions(targets, predictions):
    """Compute accuracy, macro-F1, and UAR."""
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "uar": recall_score(targets, predictions, average="macro", zero_division=0),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one training or validation epoch."""
    training = optimizer is not None
    model.train(training)
    total_loss, targets, predictions = 0.0, [], []

    for images, labels, _ in tqdm(loader, leave=False, desc="train" if training else "eval"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(images)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(1).detach().cpu().tolist())

    return {"loss": total_loss / len(loader.dataset), **metrics_from_predictions(targets, predictions)}


def predict(model, loader, device):
    """Return test targets, predictions, probabilities, and row indices."""
    model.eval()
    targets, predictions, probabilities, indices = [], [], [], []
    with torch.no_grad():
        for images, labels, batch_indices in tqdm(loader, leave=False, desc="test"):
            probs = torch.softmax(model(images.to(device, non_blocking=True)), dim=1).cpu()
            targets.extend(labels.tolist())
            predictions.extend(probs.argmax(1).tolist())
            probabilities.extend(probs.tolist())
            indices.extend(batch_indices.tolist())
    return tuple(map(np.asarray, (targets, predictions, probabilities, indices)))


def save_confusion_matrix(targets, predictions):
    """Save confusion matrix data and figure."""
    matrix = confusion_matrix(targets, predictions, labels=range(len(CLASS_NAMES)))
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix)
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            ax.text(column, row, matrix[row, column], ha="center", va="center")
    ax.set(
        xticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES,
        yticks=range(len(CLASS_NAMES)), yticklabels=CLASS_NAMES,
        xlabel="Predicted residue bin", ylabel="True residue bin", title="Test confusion matrix",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def main():
    """Train the CNN, select the best validation checkpoint, and test it."""
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_metadata()
    train_data, validation_data, test_data = split_metadata(data)
    train_transform, eval_transform = make_transforms()
    train_loader = make_loader(train_data, train_transform, True)
    validation_loader = make_loader(validation_data, eval_transform, False)
    test_loader = make_loader(test_data, eval_transform, False)

    weights = class_weights(train_data).to(device)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"Device: {device}")
    print(f"Total: {len(data)} images | {data[SAMPLE_ID_COLUMN].nunique()} sample_ids")
    print(f"Train: {len(train_data)} images | {train_data[SAMPLE_ID_COLUMN].nunique()} sample_ids")
    print(f"Validation: {len(validation_data)} images | {validation_data[SAMPLE_ID_COLUMN].nunique()} sample_ids")
    print(f"Test: {len(test_data)} images | {test_data[SAMPLE_ID_COLUMN].nunique()} sample_ids")
    print("\nClass distributions:")
    print_split_distribution("Overall", data)
    print_split_distribution("Train", train_data)
    print_split_distribution("Validation", validation_data)
    print_split_distribution("Test", test_data)

    print("\nTraining class weights:")
    for name, weight in zip(CLASS_NAMES, weights.cpu().tolist()):
        print(f"  {name}: {weight:.4f}")

    history = []
    best_f1, best_epoch, stale_epochs = -1.0, 0, 0
    checkpoint_path = OUTPUT_DIR / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, validation_loader, criterion, device)
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"validation_{k}": v for k, v in val_metrics.items()},
        })

        print(
            f"Epoch {epoch:02d} | train loss {train_metrics['loss']:.4f} | "
            f"train acc {train_metrics['accuracy']:.4f} | train F1 {train_metrics['macro_f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | val acc {val_metrics['accuracy']:.4f} | "
            f"val F1 {val_metrics['macro_f1']:.4f} | val UAR {val_metrics['uar']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, stale_epochs = val_metrics["macro_f1"], epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "image_size": IMAGE_SIZE,
                "epoch": epoch,
                "validation_macro_f1": best_f1,
            }, checkpoint_path)
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print(f"Early stopping after epoch {epoch}.")
            break

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    targets, predictions, probabilities, row_indices = predict(model, test_loader, device)
    test_metrics = metrics_from_predictions(targets, predictions)
    report = classification_report(
        targets, predictions, labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )

    metrics = {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "classification_report": report,
        "class_names": list(CLASS_NAMES),
        "train_size": len(train_data),
        "validation_size": len(validation_data),
        "test_size": len(test_data),
        "seed": SEED,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    prediction_rows = test_data.iloc[row_indices].reset_index(drop=True).copy()
    prediction_rows["target_index"] = targets
    prediction_rows["prediction_index"] = predictions
    prediction_rows["prediction_label"] = [CLASS_NAMES[i] for i in predictions]
    for i, name in enumerate(CLASS_NAMES):
        prediction_rows[f"probability_{name}"] = probabilities[:, i]
    prediction_rows.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    save_confusion_matrix(targets, predictions)
    train_data.to_csv(OUTPUT_DIR / "train_split.csv", index=False)
    validation_data.to_csv(OUTPUT_DIR / "validation_split.csv", index=False)
    test_data.to_csv(OUTPUT_DIR / "test_split.csv", index=False)

    print(f"\nBest checkpoint: epoch {best_epoch} | validation macro-F1 {best_f1:.4f}")
    print("Test results")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f} ({100 * test_metrics['accuracy']:.2f}%)")
    print(f"  Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  UAR: {test_metrics['uar']:.4f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


# Workflow
# 1. Load anthony/master_icp.csv and validate image_path, residue_bin, and sample_id.
# 2. Stratify whole sample_id groups into 80% train, 10% validation, and 10% test while preserving label proportions.
# 3. Verify no sample_id appears in more than one split and every split contains all three labels.
# 4. Resize images to 128x128, normalize them, and horizontally flip training images at random.
# 5. Compute inverse-frequency class weights from the training split.
# 6. Train the scratch CNN with weighted cross-entropy and AdamW.
# 7. Evaluate every epoch on validation data and keep the checkpoint with the best validation macro-F1.
# 8. Stop early when validation macro-F1 does not improve for PATIENCE epochs.
# 9. Reload the best checkpoint and evaluate once on the held-out test set.
# 10. Report accuracy, macro-F1, UAR, per-class metrics, predictions, and a confusion matrix.
# 11. Save the exact train/validation/test CSV splits and all results under anthony/cnn_results.