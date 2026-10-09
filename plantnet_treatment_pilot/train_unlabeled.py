"""Matched PlantNet pilot: optional treatment pretraining, then ICP training.

Run both matched experiments (default): python train_unlabeled.py
Run only A: python train_unlabeled.py --pretrain off
Run only B: python train_unlabeled.py --pretrain on

Edit the paths below for your machine. Splits are stratified by image, 80/10/10;
no tree grouping or cross-validation. Unknown tree IDs are saved as "unknown".
Treatment labels are nominal application concentrations, not ICP measurements.
"""

import argparse
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
from pillow_heif import register_heif_opener

register_heif_opener()

PROJECT_ROOT = Path("/home/davidorjuela/dev/agai-reu-pesticide")
CSV_PATH = PROJECT_ROOT / "anthony/master_icp.csv"
OUTPUT_DIR = PROJECT_ROOT / "plantnet_treatment_pilot/results"
TREATMENT_CSV_PATH = (
    PROJECT_ROOT / "datasets/kocide_application_2025/nominal_treatment.csv"
)
# "both" runs A and B together and writes a direct comparison.
PRETRAINING = "both"
TREATMENT_LABEL_COLUMN = "nominal_treatment"
TREATMENT_CLASS_NAMES = ("400", "600", "800", "1000")
TREATMENT_EPOCHS = 5
TREATMENT_LEARNING_RATE = 1e-4
ICP_HEAD_SEED = 43
TREE_ID_COLUMN = "tree_id"

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
        label_column=LABEL_COLUMN,
        class_names=CLASS_NAMES,
    ):
        self.data = dataframe.reset_index(drop=True)
        self.large_transform = large_transform
        self.small_transform = small_transform
        self.tile_transform = tile_transform
        self.training = training
        self.label_column = label_column
        self.class_names = tuple(class_names)
        self.class_to_index = {name: i for i, name in enumerate(self.class_names)}

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
            self.class_to_index[row[self.label_column]],
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


def build_plantnet_model(num_outputs=len(CLASS_NAMES)):
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
    backbone.fc = nn.Linear(input_features, num_outputs)
    return MultiScaleTilePlantNet(backbone)


def load_metadata(csv_path, label_column, class_names):
    """Validate a task CSV; sample IDs are optional and trees may be unknown."""
    data = pd.read_csv(csv_path)
    required = [IMAGE_COLUMN, label_column]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns: {missing}")
    if data[required].isna().any().any():
        raise ValueError(f"{csv_path}: missing image paths or labels.")

    if label_column == TREATMENT_LABEL_COLUMN:
        # Accept both integer and floating-point CSV representations of ppm.
        values = pd.to_numeric(data[label_column], errors="raise")
        if not values.isin([int(name) for name in class_names]).all():
            raise ValueError(f"Unexpected treatment labels: {values.unique()}")
        data[label_column] = values.astype(int).astype(str)
    else:
        data[label_column] = data[label_column].astype(str).str.strip().str.lower()
    invalid = sorted(set(data[label_column]) - set(class_names))
    if invalid:
        raise ValueError(f"Unexpected {label_column} labels: {invalid}")

    # Relative paths are relative to the CSV, never the launch directory.
    def resolve_image(value):
        path = Path(str(value)).expanduser()
        return str((path if path.is_absolute() else csv_path.parent / path).resolve())

    data[IMAGE_COLUMN] = data[IMAGE_COLUMN].map(resolve_image)
    missing_images = [path for path in data[IMAGE_COLUMN] if not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} images are missing. First paths:\n"
            + "\n".join(missing_images[:5])
        )
    if data[IMAGE_COLUMN].duplicated().any():
        raise ValueError(f"{csv_path}: duplicate image paths.")
    if TREE_ID_COLUMN not in data:
        data[TREE_ID_COLUMN] = "unknown"
    else:
        data[TREE_ID_COLUMN] = data[TREE_ID_COLUMN].fillna("unknown")
    return data


def split_metadata(data, label_column, class_names):
    """One fixed stratified image-level 80/10/10 split; no group inference."""
    try:
        train, heldout = train_test_split(
            data,
            test_size=VALIDATION_FRACTION + TEST_FRACTION,
            random_state=SEED,
            stratify=data[label_column],
        )
        validation, test = train_test_split(
            heldout,
            test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
            random_state=SEED,
            stratify=heldout[label_column],
        )
    except ValueError as exc:
        raise ValueError(
            f"Cannot create stratified 80/10/10 splits for {label_column}; "
            f"class counts: {data[label_column].value_counts().to_dict()}. {exc}"
        ) from exc
    splits = tuple(frame.reset_index(drop=True) for frame in (train, validation, test))
    for name, frame in zip(("train", "validation", "test"), splits):
        if set(frame[label_column]) != set(class_names):
            raise ValueError(f"{label_column}: {name} split is missing classes.")
        print(f"{label_column} {name}: {len(frame)} images | "
              f"{frame[label_column].value_counts().to_dict()}")
    return splits


def save_splits(splits, output_dir):
    """Save image lists, task labels, available IDs, and explicit split names."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, frame in zip(("train", "validation", "test"), splits):
        frame = frame.assign(split=name)
        frame.to_csv(output_dir / f"{name}_split.csv", index=False)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(
        output_dir / "dataset_manifest.csv", index=False,
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
    label_column=LABEL_COLUMN,
    class_names=CLASS_NAMES,
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
            label_column=label_column,
            class_names=class_names,
        ),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def class_weights(train_data, label_column, class_names):
    """Compute inverse-frequency weights using only this task's training split."""
    counts = train_data[label_column].value_counts().reindex(
        class_names, fill_value=0,
    ).to_numpy(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"Training split contains an empty class: {counts}")
    return torch.tensor(
        len(train_data) / (len(class_names) * counts), dtype=torch.float32,
    )


def metrics_from_predictions(targets, predictions, class_names):
    """Use all task classes for accuracy, macro-F1, UAR, and per-class recall."""
    labels = list(range(len(class_names)))
    recalls = recall_score(
        targets, predictions, labels=labels, average=None, zero_division=0,
    )
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(
            targets, predictions, labels=labels, average="macro", zero_division=0,
        ),
        "uar": float(np.mean(recalls)),
        **{f"recall_{name}": float(value) for name, value in zip(class_names, recalls)},
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
        **metrics_from_predictions(targets, predictions, loader.dataset.class_names),
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
                    "class_names": train_loader.dataset.class_names,
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


def save_confusion_matrix(targets, predictions, class_names, output_dir):
    """Save confusion matrix data and figure."""
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=range(len(class_names)),
    )

    pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    ).to_csv(output_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix)

    for row in range(len(class_names)):
        for column in range(len(class_names)):
            ax.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
            )

    ax.set(
        xticks=range(len(class_names)),
        xticklabels=class_names,
        yticks=range(len(class_names)),
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="PlantNet ResNet-18 test confusion matrix",
    )

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def train_task(model, splits, label_column, class_names, output_dir, device, treatment=False):
    """Train one task with fresh loaders, optimizers, and checkpoint selection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_splits(splits, output_dir)
    # Reset augmentation RNG and loader generators at each task, including ICP B.
    set_seed(SEED)
    transforms_by_view = make_transforms()
    loaders = [
        make_loader(frame, *transforms_by_view, index == 0, label_column, class_names)
        for index, frame in enumerate(splits)
    ]
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(splits[0], label_column, class_names).to(device),
    )
    checkpoint_path = output_dir / ("best_treatment.pt" if treatment else "best_icp.pt")
    # Never carry treatment validation scores into the ICP checkpoint selection.
    best_f1, best_epoch = -1.0, 0
    history = []
    phases = (
        [("treatment_full", TREATMENT_EPOCHS, TREATMENT_LEARNING_RATE, False)]
        if treatment else [
            ("icp_head", HEAD_EPOCHS, HEAD_LEARNING_RATE, True),
            ("icp_full", FINE_TUNE_EPOCHS, FINE_TUNE_LEARNING_RATE, False),
        ]
    )
    for phase_index, (phase, epochs, learning_rate, head_only) in enumerate(phases):
        if phase_index:
            # Anthony's full fine-tuning starts from the best head checkpoint.
            model.load_state_dict(torch.load(
                checkpoint_path, map_location=device, weights_only=False,
            )["model_state_dict"])
        set_head_only(model, head_only)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=learning_rate, weight_decay=WEIGHT_DECAY,
        )
        stage_history, best_f1, best_epoch = train_stage(
            model=model,
            train_loader=loaders[0],
            validation_loader=loaders[1],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            epochs=epochs,
            stage_name=phase,
            checkpoint_path=checkpoint_path,
            best_f1=best_f1,
            best_epoch=best_epoch,
            epoch_offset=len(history),
            head_only=head_only,
        )
        history.extend(stage_history)
        del optimizer
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Evaluate the selected checkpoint once on this task's held-out test split.
    targets, predictions, probabilities, row_indices = predict(model, loaders[2], device)
    test_metrics = metrics_from_predictions(targets, predictions, class_names)
    report = classification_report(
        targets, predictions, labels=range(len(class_names)),
        target_names=list(class_names), output_dict=True, zero_division=0,
    )
    metrics = {
        "task": "treatment" if treatment else "icp",
        "backbone": "plantnet300k_resnet18_multiscale_tiles",
        "pretrained_weights": str(PLANTNET_WEIGHTS_PATH),
        "best_epoch": best_epoch,
        "best_stage": checkpoint["stage"],
        "best_validation_macro_f1": best_f1,
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "classification_report": report,
        "class_names": list(class_names),
        "class_to_index": {name: i for i, name in enumerate(class_names)},
        "train_size": len(splits[0]),
        "validation_size": len(splits[1]),
        "test_size": len(splits[2]),
        "seed": SEED,
        "split_unit": "image",
        "tree_separated": False,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    prediction_rows = splits[2].iloc[row_indices].reset_index(drop=True).copy()
    prediction_rows["target_index"] = targets
    prediction_rows["prediction_index"] = predictions
    prediction_rows["prediction_label"] = [class_names[i] for i in predictions]
    for index, name in enumerate(class_names):
        prediction_rows[f"probability_{name}"] = probabilities[:, index]
    prediction_rows.to_csv(output_dir / "test_predictions.csv", index=False)
    save_confusion_matrix(targets, predictions, class_names, output_dir)
    print(f"\n{output_dir.name}: selected {checkpoint['stage']}, epoch {best_epoch}")
    print(f"Validation macro-F1: {best_f1:.4f} | Test: {test_metrics}")
    return metrics


def run_experiment(pretrain, icp_splits, treatment_splits, head_state, device):
    """Use original PlantNet weights for each run and an identical new ICP head."""
    run_name = "B_treatment_pretrained" if pretrain else "A_matched_baseline"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = build_plantnet_model(
        num_outputs=len(TREATMENT_CLASS_NAMES) if pretrain else len(CLASS_NAMES),
    ).to(device)
    if pretrain:
        # Full backbone + four-class head learn; train_task reloads the best model.
        train_task(
            model, treatment_splits, TREATMENT_LABEL_COLUMN, TREATMENT_CLASS_NAMES,
            run_dir / "treatment", device, treatment=True,
        )
    # Discard any prior task head. Transfer the selected backbone (including BN
    # buffers), then start ICP with exactly the same new head in A and B.
    model.backbone.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.fc.load_state_dict(head_state)
    model.to(device)
    metrics = train_task(
        model, icp_splits, LABEL_COLUMN, CLASS_NAMES, run_dir / "icp", device,
    )
    del model
    return {"run": run_name, "treatment_pretraining": pretrain, **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrain", choices=("off", "on", "both"), default=PRETRAINING,
        help="off: matched baseline A; on: treatment-pretrained B; both: compare A/B",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; pretraining: {args.pretrain}")
    print("Pilot: image-level 80/10/10 splits; unknown tree relationships are not controlled.")
    icp_data = load_metadata(CSV_PATH, LABEL_COLUMN, CLASS_NAMES)
    # Split ONCE and pass these exact same frames to A and B.
    icp_splits = split_metadata(icp_data, LABEL_COLUMN, CLASS_NAMES)
    save_splits(icp_splits, OUTPUT_DIR / "shared_icp_splits")
    treatment_splits = None
    if args.pretrain in ("on", "both"):
        treatment_data = load_metadata(
            TREATMENT_CSV_PATH, TREATMENT_LABEL_COLUMN, TREATMENT_CLASS_NAMES,
        )
        # Exclude every ICP image before any treatment split. This detects shared
        # resolved paths, not renamed copies or unrecorded shared leaves/trees.
        overlap = treatment_data[IMAGE_COLUMN].isin(icp_data[IMAGE_COLUMN])
        treatment_data.loc[overlap].to_csv(
            OUTPUT_DIR / "treatment_excluded_icp_overlap.csv", index=False,
        )
        print(f"Excluded {int(overlap.sum())} treatment images also present in ICP.")
        treatment_data = treatment_data.loc[~overlap].reset_index(drop=True)
        treatment_splits = split_metadata(
            treatment_data, TREATMENT_LABEL_COLUMN, TREATMENT_CLASS_NAMES,
        )

    # ResNet-18 has 512 features. Save and reuse the exact tensors, not just a seed.
    set_seed(ICP_HEAD_SEED)
    head_state = nn.Linear(512, len(CLASS_NAMES)).state_dict()
    torch.save(head_state, OUTPUT_DIR / "shared_icp_initial_head.pt")
    config = {
        "pretrain": args.pretrain,
        "icp_csv": str(CSV_PATH),
        "treatment_csv": str(TREATMENT_CSV_PATH),
        "plantnet_weights": str(PLANTNET_WEIGHTS_PATH),
        "seed": SEED,
        "icp_head_seed": ICP_HEAD_SEED,
        "split_unit": "image",
        "split_fractions": [0.8, VALIDATION_FRACTION, TEST_FRACTION],
        "tree_separated": False,
        "icp_classes": CLASS_NAMES,
        "treatment_classes": TREATMENT_CLASS_NAMES,
        "treatment_epochs": TREATMENT_EPOCHS,
        "treatment_learning_rate": TREATMENT_LEARNING_RATE,
        "head_epochs": HEAD_EPOCHS,
        "fine_tune_epochs": FINE_TUNE_EPOCHS,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "fine_tune_learning_rate": FINE_TUNE_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience_per_phase": PATIENCE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "image_sizes": [IMAGE_SIZE, SMALL_IMAGE_SIZE, TILE_IMAGE_SIZE],
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "augmentation": "random flips and right-angle rotation",
        "loss": "training-split inverse-frequency weighted cross-entropy",
        "optimizer": "AdamW; fresh per task and phase",
        "checkpoint_rule": "strictly higher validation macro-F1; ties keep earlier checkpoint",
        "success_criterion": "ICP test performance B versus A on the shared split",
        "historical_accuracy_target": 0.61,
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2))
    modes = [False, True] if args.pretrain == "both" else [args.pretrain == "on"]
    results = [
        run_experiment(pretrain, icp_splits, treatment_splits, head_state, device)
        for pretrain in modes
    ]
    if len(results) == 2:
        fields = ["test_accuracy", "test_macro_f1"] + [
            f"test_recall_{name}" for name in CLASS_NAMES
        ]
        rows = [{"run": result["run"], **{key: result[key] for key in fields}}
                for result in results]
        rows.append({"run": "B_minus_A", **{
            key: results[1][key] - results[0][key] for key in fields
        }})
        comparison = pd.DataFrame(rows)
        comparison.to_csv(OUTPUT_DIR / "comparison.csv", index=False)
        (OUTPUT_DIR / "comparison.json").write_text(json.dumps({
            "results": results,
            "B_minus_A": rows[-1],
            "historical_accuracy_target": 0.61,
            "interpretation": "Use matched ICP B versus A; treatment accuracy alone "
                              "is not success. This is an image-split pilot, not a "
                              "tree-separated or cross-validated estimate.",
        }, indent=2))
        print("\nICP comparison (fractions; multiply by 100 for percent/percentage points):")
        print(comparison.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
