"""Fine-tune PlantNet ResNet-18 with multi-scale global views and overlapping local tiles."""

import json
import random
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

PROJECT_ROOT = Path("/home/davidorjuela/dev/agai-reu-pesticide")
CSV_PATH = PROJECT_ROOT / "anthony/master_icp.csv"
OUTPUT_DIR = PROJECT_ROOT / "anthony/plantnet_resnet18_multiscale_tiles_results_ablation"

PRETRAINED_DIR = PROJECT_ROOT / "anthony/pretrained"
PLANTNET_WEIGHTS_PATH = PRETRAINED_DIR / "plantnet_resnet18.pth"
PLANTNET_WEIGHTS_URL = (
    "https://huggingface.co/cpoisson/plantnet300k-resnet18/"
    "resolve/main/plantnet_resnet18.pth"
)

IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "residue_bin"
SAMPLE_ID_COLUMN = "sample_id"

CLASS_NAMES = ("low", "medium", "high")
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = 448
SMALL_IMAGE_SIZE = 224
TILE_IMAGE_SIZE = 224
BATCH_SIZE = 8

HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 40
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8

VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10
SEED = 42
NUM_WORKERS = 4

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed):
    """Make training reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResizePadSquare:
    """Resize the full image to fit inside a square and pad without cropping."""

    def __init__(self, size, fill=(124, 116, 104)):
        self.size = size
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        pad_left = (self.size - new_width) // 2
        pad_top = (self.size - new_height) // 2
        pad_right = self.size - new_width - pad_left
        pad_bottom = self.size - new_height - pad_top
        return ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
        )


def detail_safe_augment(image):
    """Apply flips and right-angle rotation without interpolation."""
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    rotation = random.randrange(4)
    if rotation == 1:
        image = image.transpose(Image.Transpose.ROTATE_90)
    elif rotation == 2:
        image = image.transpose(Image.Transpose.ROTATE_180)
    elif rotation == 3:
        image = image.transpose(Image.Transpose.ROTATE_270)
    return image


class LeafDataset(Dataset):
    """Load two global scales and a 3x3 grid of overlapping local tiles."""

    def __init__(
        self,
        dataframe,
        large_transform,
        small_transform,
        tile_transform,
        training,
    ):
        self.data = dataframe.reset_index(drop=True)
        self.large_transform = large_transform
        self.small_transform = small_transform
        self.tile_transform = tile_transform
        self.training = training

    def __len__(self):
        return len(self.data)

    @staticmethod
    def overlapping_tiles(image):
        width, height = image.size
        tile_width = max(1, (width + 1) // 2)
        tile_height = max(1, (height + 1) // 2)
        x_positions = [0, (width - tile_width) // 2, width - tile_width]
        y_positions = [0, (height - tile_height) // 2, height - tile_height]
        return [
            image.crop((x, y, x + tile_width, y + tile_height))
            for y in y_positions
            for x in x_positions
        ]

    def __getitem__(self, index):
        row = self.data.iloc[index]
        with Image.open(row[IMAGE_COLUMN]) as image:
            image = image.convert("RGB")
            if self.training:
                image = detail_safe_augment(image)
            global_large = self.large_transform(image)
            global_small = self.small_transform(image)
            tiles = torch.stack([
                self.tile_transform(tile)
                for tile in self.overlapping_tiles(image)
            ])
        return (
            global_large,
            global_small,
            tiles,
            CLASS_TO_INDEX[row[LABEL_COLUMN]],
            index,
        )


def download_plantnet_weights():
    """Download the PlantNet-300K ResNet-18 checkpoint when it is missing."""
    if PLANTNET_WEIGHTS_PATH.is_file():
        return

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PlantNet ResNet-18 weights to:\n  {PLANTNET_WEIGHTS_PATH}")

    try:
        urllib.request.urlretrieve(
            PLANTNET_WEIGHTS_URL,
            PLANTNET_WEIGHTS_PATH,
        )
    except Exception:
        PLANTNET_WEIGHTS_PATH.unlink(missing_ok=True)
        raise


def extract_state_dict(checkpoint):
    """Extract a model state dictionary from official or raw checkpoints."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Unexpected PlantNet checkpoint format.")

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


class MultiScaleTilePlantNet(nn.Module):
    """Apply one shared PlantNet backbone to global scales and local tiles."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    @property
    def fc(self):
        return self.backbone.fc

    def forward(self, global_large, global_small, tiles):
        large_logits = self.backbone(global_large)
        small_logits = self.backbone(global_small)
        batch_size, views, channels, height, width = tiles.shape
        tile_logits = self.backbone(
            tiles.reshape(batch_size * views, channels, height, width)
        ).reshape(batch_size, views, -1)
        tile_logits = tile_logits.mean(dim=1)
        return (large_logits + small_logits + tile_logits) / 3.0


def build_plantnet_model():
    """Load PlantNet-300K ResNet-18 weights and wrap it for multi-scale views."""
    download_plantnet_weights()
    backbone = resnet18(weights=None, num_classes=1081)
    checkpoint = torch.load(
        PLANTNET_WEIGHTS_PATH,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = extract_state_dict(checkpoint)
    backbone.load_state_dict(state_dict, strict=True)
    input_features = backbone.fc.in_features
    backbone.fc = nn.Linear(input_features, len(CLASS_NAMES))
    return MultiScaleTilePlantNet(backbone)


def load_metadata():
    """Load the CSV and validate paths, labels, and sample IDs."""
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

    missing_images = [
        path for path in data[IMAGE_COLUMN]
        if not Path(path).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} images are missing. First paths:\n"
            + "\n".join(missing_images[:5])
        )

    if data[IMAGE_COLUMN].duplicated().any():
        raise ValueError("Duplicate image paths found in the CSV.")

    return data


def split_metadata(data):
    """Stratify whole sample_id groups into an 80/10/10 split."""
    labels_per_sample = data.groupby(SAMPLE_ID_COLUMN)[LABEL_COLUMN].nunique()
    inconsistent = labels_per_sample[labels_per_sample != 1]

    if len(inconsistent):
        raise ValueError(
            f"{len(inconsistent)} sample IDs map to multiple residue bins."
        )

    groups = (
        data[[SAMPLE_ID_COLUMN, LABEL_COLUMN]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

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

    for name, split in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        if set(split[LABEL_COLUMN]) != set(CLASS_NAMES):
            missing = set(CLASS_NAMES) - set(split[LABEL_COLUMN])
            raise RuntimeError(
                f"{name} split is missing classes: {sorted(missing)}"
            )

    return train, validation, test


def print_split_distribution(name, data):
    """Print image and sample_id counts for each class."""
    image_counts = (
        data[LABEL_COLUMN]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )
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
    """Build global multi-scale and local-tile transforms."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    large_transform = transforms.Compose([
        ResizePadSquare(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])
    small_transform = transforms.Compose([
        ResizePadSquare(SMALL_IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])
    tile_transform = transforms.Compose([
        ResizePadSquare(TILE_IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])
    return large_transform, small_transform, tile_transform


def make_loader(
    data,
    large_transform,
    small_transform,
    tile_transform,
    shuffle,
):
    """Create a deterministic multi-scale tile DataLoader."""
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        LeafDataset(
            data,
            large_transform,
            small_transform,
            tile_transform,
            training=shuffle,
        ),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def class_weights(train_data):
    """Compute inverse-frequency class weights from training labels."""
    counts = (
        train_data[LABEL_COLUMN]
        .value_counts()
        .reindex(CLASS_NAMES)
        .to_numpy(np.float32)
    )

    if np.any(counts == 0):
        raise ValueError(f"Training split contains an empty class: {counts}")

    return torch.tensor(
        len(train_data) / (len(CLASS_NAMES) * counts),
        dtype=torch.float32,
    )


def metrics_from_predictions(targets, predictions):
    """Compute accuracy, macro-F1, and UAR."""
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "uar": recall_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


def set_head_only(model, enabled):
    """Freeze or unfreeze the PlantNet ResNet-18 backbone."""
    for parameter in model.parameters():
        parameter.requires_grad = not enabled

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    head_only=False,
):
    """Run one multi-scale tile training or validation epoch."""
    training = optimizer is not None
    if training and head_only:
        model.eval()
        model.fc.train()
    else:
        model.train(training)
    total_loss = 0.0
    targets = []
    predictions = []
    for global_large, global_small, tiles, labels, _ in tqdm(
        loader,
        leave=False,
        desc="train" if training else "eval",
    ):
        global_large = global_large.to(device, non_blocking=True)
        global_small = global_small.to(device, non_blocking=True)
        tiles = tiles.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(global_large, global_small, tiles)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(1).detach().cpu().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        **metrics_from_predictions(targets, predictions),
    }


def train_stage(
    model,
    train_loader,
    validation_loader,
    criterion,
    device,
    optimizer,
    epochs,
    stage_name,
    checkpoint_path,
    best_f1,
    best_epoch,
    epoch_offset,
    head_only=False,
):
    """Train one fine-tuning stage and keep the best validation checkpoint."""
    history = []
    stale_epochs = 0

    for local_epoch in range(1, epochs + 1):
        epoch = epoch_offset + local_epoch

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            head_only=head_only,
        )
        val_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        history.append({
            "stage": stage_name,
            "epoch": epoch,
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },
            **{
                f"validation_{key}": value
                for key, value in val_metrics.items()
            },
        })

        print(
            f"{stage_name} | Epoch {local_epoch:02d}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"train acc {train_metrics['accuracy']:.4f} | "
            f"train F1 {train_metrics['macro_f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val acc {val_metrics['accuracy']:.4f} | "
            f"val F1 {val_metrics['macro_f1']:.4f} | "
            f"val UAR {val_metrics['uar']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "stage": stage_name,
                    "epoch": epoch,
                    "validation_macro_f1": best_f1,
                    "backbone": "plantnet300k_resnet18_multiscale_tiles",
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print(f"{stage_name}: early stopping after epoch {local_epoch}.")
            break

    return history, best_f1, best_epoch


def predict(model, loader, device):
    """Return test targets, predictions, probabilities, and row indices."""
    model.eval()
    targets = []
    predictions = []
    probabilities = []
    indices = []
    with torch.no_grad():
        for global_large, global_small, tiles, labels, batch_indices in tqdm(
            loader,
            leave=False,
            desc="test",
        ):
            global_large = global_large.to(device, non_blocking=True)
            global_small = global_small.to(device, non_blocking=True)
            tiles = tiles.to(device, non_blocking=True)
            probs = torch.softmax(
                model(global_large, global_small, tiles),
                dim=1,
            ).cpu()
            targets.extend(labels.tolist())
            predictions.extend(probs.argmax(1).tolist())
            probabilities.extend(probs.tolist())
            indices.extend(batch_indices.tolist())
    return tuple(
        map(
            np.asarray,
            (targets, predictions, probabilities, indices),
        )
    )


def save_confusion_matrix(targets, predictions):
    """Save confusion matrix data and figure."""
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=range(len(CLASS_NAMES)),
    )

    pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix)

    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            ax.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
            )

    ax.set(
        xticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticks=range(len(CLASS_NAMES)),
        yticklabels=CLASS_NAMES,
        xlabel="Predicted residue bin",
        ylabel="True residue bin",
        title="PlantNet ResNet-18 test confusion matrix",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def main():
    """Fine-tune PlantNet ResNet-18 and evaluate its best validation checkpoint."""
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_metadata()
    train_data, validation_data, test_data = split_metadata(data)

    large_transform, small_transform, tile_transform = make_transforms()
    train_loader = make_loader(
        train_data,
        large_transform,
        small_transform,
        tile_transform,
        True,
    )
    validation_loader = make_loader(
        validation_data,
        large_transform,
        small_transform,
        tile_transform,
        False,
    )
    test_loader = make_loader(
        test_data,
        large_transform,
        small_transform,
        tile_transform,
        False,
    )

    weights = class_weights(train_data).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = build_plantnet_model().to(device)

    print(f"Device: {device}")
    print("Backbone: PlantNet-300K pretrained ResNet-18")
    print(
        f"Input: global {IMAGE_SIZE} + global {SMALL_IMAGE_SIZE} + "
        f"9 overlapping {TILE_IMAGE_SIZE} tiles"
    )
    print(
        f"Total: {len(data)} images | "
        f"{data[SAMPLE_ID_COLUMN].nunique()} sample_ids"
    )
    print(
        f"Train: {len(train_data)} images | "
        f"{train_data[SAMPLE_ID_COLUMN].nunique()} sample_ids"
    )
    print(
        f"Validation: {len(validation_data)} images | "
        f"{validation_data[SAMPLE_ID_COLUMN].nunique()} sample_ids"
    )
    print(
        f"Test: {len(test_data)} images | "
        f"{test_data[SAMPLE_ID_COLUMN].nunique()} sample_ids"
    )

    print("\nClass distributions:")
    print_split_distribution("Overall", data)
    print_split_distribution("Train", train_data)
    print_split_distribution("Validation", validation_data)
    print_split_distribution("Test", test_data)

    print("\nTraining class weights:")
    for name, weight in zip(CLASS_NAMES, weights.cpu().tolist()):
        print(f"  {name}: {weight:.4f}")

    checkpoint_path = OUTPUT_DIR / "best_model.pt"
    history = []
    best_f1 = -1.0
    best_epoch = 0

    print("\nStage 1: train classification head")
    set_head_only(model, True)
    head_optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=HEAD_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    stage_history, best_f1, best_epoch = train_stage(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        device=device,
        optimizer=head_optimizer,
        epochs=HEAD_EPOCHS,
        stage_name="head",
        checkpoint_path=checkpoint_path,
        best_f1=best_f1,
        best_epoch=best_epoch,
        epoch_offset=0,
        head_only=True,
    )
    history.extend(stage_history)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nStage 2: fine-tune full PlantNet backbone")
    set_head_only(model, False)
    fine_tune_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINE_TUNE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    stage_history, best_f1, best_epoch = train_stage(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        device=device,
        optimizer=fine_tune_optimizer,
        epochs=FINE_TUNE_EPOCHS,
        stage_name="full",
        checkpoint_path=checkpoint_path,
        best_f1=best_f1,
        best_epoch=best_epoch,
        epoch_offset=HEAD_EPOCHS,
        head_only=False,
    )
    history.extend(stage_history)

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "history.csv",
        index=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    targets, predictions, probabilities, row_indices = predict(
        model,
        test_loader,
        device,
    )

    test_metrics = metrics_from_predictions(targets, predictions)
    report = classification_report(
        targets,
        predictions,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "backbone": "plantnet300k_resnet18_multiscale_tiles",
        "pretrained_weights": str(PLANTNET_WEIGHTS_PATH),
        "best_epoch": best_epoch,
        "best_stage": checkpoint["stage"],
        "best_validation_macro_f1": best_f1,
        **{
            f"test_{key}": value
            for key, value in test_metrics.items()
        },
        "classification_report": report,
        "class_names": list(CLASS_NAMES),
        "train_size": len(train_data),
        "validation_size": len(validation_data),
        "test_size": len(test_data),
        "seed": SEED,
    }

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    prediction_rows = (
        test_data.iloc[row_indices]
        .reset_index(drop=True)
        .copy()
    )
    prediction_rows["target_index"] = targets
    prediction_rows["prediction_index"] = predictions
    prediction_rows["prediction_label"] = [
        CLASS_NAMES[index]
        for index in predictions
    ]

    for i, name in enumerate(CLASS_NAMES):
        prediction_rows[f"probability_{name}"] = probabilities[:, i]

    prediction_rows.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    save_confusion_matrix(targets, predictions)

    train_data.to_csv(OUTPUT_DIR / "train_split.csv", index=False)
    validation_data.to_csv(
        OUTPUT_DIR / "validation_split.csv",
        index=False,
    )
    test_data.to_csv(OUTPUT_DIR / "test_split.csv", index=False)

    print(
        f"\nBest checkpoint: stage {checkpoint['stage']} | "
        f"epoch {best_epoch} | validation macro-F1 {best_f1:.4f}"
    )
    print("Test results")
    print(
        f"  Accuracy: {test_metrics['accuracy']:.4f} "
        f"({100 * test_metrics['accuracy']:.2f}%)"
    )
    print(f"  Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  UAR: {test_metrics['uar']:.4f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

# Workflow
# 1. Load and validate anthony/master_icp.csv.
# 2. Keep whole sample_id groups in stratified 80/10/10 train/validation/test splits.
# 3. Load a ResNet-18 pretrained on PlantNet-300K's 1081 plant species.
# 4. Replace its 1081-class head with a 3-class low/medium/high residue head.
# 5. Use 448/224 global scales plus nine overlapping 224x224 local tiles.
# 6. Train only the new classification head first while the backbone is frozen.
# 7. Reload the best checkpoint, unfreeze the full network, and fine-tune at a lower LR.
# 8. Keep the checkpoint with the best validation macro-F1 across both stages.
# 9. Reload that checkpoint and evaluate once on the held-out test split.
# 10. Save metrics, predictions, confusion matrix, history, and exact split CSVs.
