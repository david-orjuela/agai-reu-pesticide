"""
AGAI_REU_2026
David Orjuela
Undergraduate Student under Dr. Chen
Training orchestration for pesticide-residue experiments.

Change CONFIG.task, CONFIG.backbone, and CONFIG.random_seeds to select an
experiment. Training, evaluation, plotting, and checkpoint logic adapts
automatically.
"""

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score, r2_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from baseline_pipeline.baseline_model import frozen_resnet
from baseline_pipeline.dataset import agai_correct_v3


class Task(str, Enum):
    BIN_CLASSIFICATION = "bin_classification"
    ICP_REGRESSION = "icp_regression"


class Backbone(str, Enum):
    RESNET50 = "resnet50"
    DINOV3 = "dinov3"


@dataclass(frozen=True)
class ExperimentConfig:
    # These are the primary experiment toggles.
    task: Task = Task.ICP_REGRESSION
    backbone: Backbone = Backbone.RESNET50

    csv_path: str = (
        "/home/davidorjuela/dev/agai-reu-pesticide/"
        "datasets/agai_correct/batch_2_v1/master_icp.csv"
    )
    residue_bin_column: str = "residue_bin"
    sample_id_column: Optional[str] = None
    class_names: Tuple[str, ...] = ("low", "medium", "high")

    learning_rate: float = 0.001
    momentum: float = 0.9
    batch_size: int = 4
    epochs: int = 30
    folds: int = 4
    # Keep (42,) for one run. For the stability experiment, use several seeds,
    # for example: (42, 7, 21, 84, 123).
    random_seeds: Tuple[int, ...] = (42,)
    num_workers: int = 0

    checkpoint_dir: Path = Path("checkpoints")
    plot_dir: Path = Path("plots")
    results_dir: Path = Path("results")
    debug: bool = False

    @property
    def num_outputs(self) -> int:
        if self.task == Task.BIN_CLASSIFICATION:
            return len(self.class_names)
        return 1


# To switch experiments, change only these settings. DINOv3 is wired into the
# model factory below and will work once frozen_dinoV3 is implemented.
CONFIG = ExperimentConfig(
    task=Task.ICP_REGRESSION,
    backbone=Backbone.RESNET50,
    random_seeds=(42,)
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ExperimentDataset(Dataset):
    """Expose either CSV bin labels or the base dataset's regression target."""

    def __init__(
        self,
        base_dataset: Dataset,
        task: Task,
        class_targets: Sequence[int],
        sample_ids: Sequence[str],
    ) -> None:
        self.base_dataset = base_dataset
        self.task = task
        self.class_targets = class_targets
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, regression_target = self.base_dataset[index]

        if self.task == Task.BIN_CLASSIFICATION:
            target = torch.tensor(self.class_targets[index], dtype=torch.long)
        else:
            target = torch.as_tensor(regression_target, dtype=torch.float32)

        return image, target


def make_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def read_class_targets(base_dataset, config: ExperimentConfig) -> List[int]:
    """Read residue bins once for classification targets and CV stratification."""

    label_to_index = {
        label.strip().lower(): index
        for index, label in enumerate(config.class_names)
    }
    bin_series = base_dataset.icp_data[config.residue_bin_column]
    class_targets = []

    for row_index, label in enumerate(bin_series):
        normalized_label = str(label).strip().lower()
        if normalized_label not in label_to_index:
            expected = ", ".join(config.class_names)
            raise ValueError(
                f"Unexpected residue bin {label!r} at row {row_index}. "
                f"Expected one of: {expected}."
            )
        class_targets.append(label_to_index[normalized_label])

    if len(class_targets) != len(base_dataset):
        raise ValueError(
            "The number of residue-bin labels does not match the dataset length: "
            f"{len(class_targets)} labels for {len(base_dataset)} samples."
        )

    return class_targets


def read_sample_ids(base_dataset, config: ExperimentConfig) -> List[str]:
    """Use a configured/recognized metadata column, or fall back to row indices."""

    metadata = base_dataset.icp_data
    columns_by_name = {
        str(column).strip().lower(): column for column in metadata.columns
    }

    if config.sample_id_column is not None:
        requested = config.sample_id_column.strip().lower()
        if requested not in columns_by_name:
            raise ValueError(
                f"Sample ID column {config.sample_id_column!r} was not found. "
                f"Available columns: {list(metadata.columns)}"
            )
        id_column = columns_by_name[requested]
    else:
        id_column = None
        candidates = (
            "sample_id",
            "sample id",
            "sample",
            "file_name",
            "file name",
            "filename",
            "image_name",
            "image name",
            "image_path",
            "image path",
            "path",
        )
        for candidate in candidates:
            if candidate in columns_by_name:
                id_column = columns_by_name[candidate]
                break

    if id_column is None:
        print("No sample ID column recognized; using dataset row indices as IDs.")
        return [str(index) for index in range(len(base_dataset))]

    return [str(value) for value in metadata[id_column].tolist()]


def build_datasets(config: ExperimentConfig):
    train_transform, val_transform = make_transforms()
    train_base = agai_correct_v3(config.csv_path, transform=train_transform)
    val_base = agai_correct_v3(config.csv_path, transform=val_transform)
    class_targets = read_class_targets(train_base, config)
    sample_ids = read_sample_ids(train_base, config)

    train_dataset = ExperimentDataset(
        train_base, config.task, class_targets, sample_ids
    )
    val_dataset = ExperimentDataset(
        val_base, config.task, class_targets, sample_ids
    )
    return train_dataset, val_dataset, class_targets


def build_resnet(num_outputs: int) -> nn.Module:
    return frozen_resnet(num_bins=num_outputs)


def build_dinov3(num_outputs: int) -> nn.Module:
    """Load DINOv3 only when selected, so unfinished code does not block ResNet."""

    try:
        from baseline_pipeline.baseline_model import frozen_dinoV3
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            "Backbone.DINOV3 is selected, but frozen_dinoV3 is not implemented "
            "in baseline_pipeline.baseline_model yet."
        ) from exc

    return frozen_dinoV3(num_bins=num_outputs)


MODEL_BUILDERS = {
    Backbone.RESNET50: build_resnet,
    Backbone.DINOV3: build_dinov3,
}


def build_model(config: ExperimentConfig, device: torch.device) -> nn.Module:
    try:
        model_builder = MODEL_BUILDERS[config.backbone]
    except KeyError as exc:
        raise ValueError(f"Unsupported backbone: {config.backbone}") from exc
    return model_builder(config.num_outputs).to(device)


def build_criterion(config: ExperimentConfig) -> nn.Module:
    if config.task == Task.BIN_CLASSIFICATION:
        return nn.CrossEntropyLoss()
    if config.task == Task.ICP_REGRESSION:
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported task: {config.task}")


def trainable_parameters(model: nn.Module):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The selected model has no trainable parameters.")
    return parameters


def prepare_batch(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    task: Task,
):
    outputs = model(images)

    if task == Task.BIN_CLASSIFICATION:
        targets = targets.long().reshape(-1)
        loss = criterion(outputs, targets)
        predictions = outputs.argmax(dim=1)
        probabilities = torch.softmax(outputs, dim=1)
    elif task == Task.ICP_REGRESSION:
        targets = targets.float().reshape(-1)
        predictions = outputs.reshape(-1)
        loss = criterion(predictions, targets)
        probabilities = None
    else:
        raise ValueError(f"Unsupported task: {task}")

    return loss, predictions, targets, probabilities


def metric_label(class_name: str) -> str:
    return class_name.strip().lower().replace(" ", "_")


def compute_metrics(
    config: ExperimentConfig,
    loss: float,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    metrics = {"loss": loss}

    if config.task == Task.BIN_CLASSIFICATION:
        labels = list(range(len(config.class_names)))
        metrics["accuracy"] = float(np.mean(predictions == targets))
        metrics["macro_f1"] = float(
            f1_score(
                targets,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        recalls = recall_score(
            targets,
            predictions,
            labels=labels,
            average=None,
            zero_division=0,
        )
        for class_name, recall in zip(config.class_names, recalls):
            metrics[f"recall_{metric_label(class_name)}"] = float(recall)
        return metrics

    metrics["mae"] = float(np.mean(np.abs(predictions - targets)))
    metrics["rmse"] = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    metrics["r2"] = float(r2_score(targets, predictions))
    metrics["target_mean"] = float(np.mean(targets))
    metrics["target_std"] = float(np.std(targets))
    metrics["prediction_mean"] = float(np.mean(predictions))
    metrics["prediction_std"] = float(np.std(predictions))

    if len(targets) < 2 or np.std(targets) == 0 or np.std(predictions) == 0:
        metrics["pearson_r"] = float("nan")
    else:
        metrics["pearson_r"] = float(pearsonr(targets, predictions)[0])

    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
):
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.set_grad_enabled(is_training):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            if is_training:
                optimizer.zero_grad()

            loss, predictions, prepared_targets, probabilities = prepare_batch(
                model, images, targets, criterion, config.task
            )

            if is_training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            all_predictions.extend(predictions.detach().cpu().tolist())
            all_targets.extend(prepared_targets.detach().cpu().tolist())
            if probabilities is not None:
                all_probabilities.extend(probabilities.detach().cpu().tolist())

    predictions_np = np.asarray(all_predictions)
    targets_np = np.asarray(all_targets)
    probabilities_np = (
        np.asarray(all_probabilities) if all_probabilities else None
    )
    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(config, epoch_loss, predictions_np, targets_np)
    return metrics, predictions_np, targets_np, probabilities_np


def print_class_counts(
    name: str,
    indices: Sequence[int],
    stratify_labels: Sequence[int],
    class_names: Sequence[str],
) -> None:
    counts = Counter(stratify_labels[index] for index in indices)
    print(f"\n{name} stratification-bin counts:")
    for label_id, label_name in enumerate(class_names):
        print(f"  {label_name} ({label_id}): {counts[label_id]}")


def print_epoch_metrics(
    epoch: int,
    config: ExperimentConfig,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
) -> None:
    prefix = f"Epoch [{epoch + 1}/{config.epochs}]"

    if config.task == Task.BIN_CLASSIFICATION:
        print(
            f"{prefix} - "
            f"Train Loss: {train_metrics['loss']:.4f} - "
            f"Train Accuracy: {train_metrics['accuracy']:.2%} - "
            f"Val Loss: {val_metrics['loss']:.4f} - "
            f"Val Accuracy: {val_metrics['accuracy']:.2%} - "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} - "
            f"Majority Baseline: {baseline_metrics['baseline_accuracy']:.2%}"
        )
        return

    print(
        f"{prefix} - "
        f"Train Loss: {train_metrics['loss']:.4f} - "
        f"Train MAE: {train_metrics['mae']:.4f} - "
        f"Train RMSE: {train_metrics['rmse']:.4f} - "
        f"Val Loss: {val_metrics['loss']:.4f} - "
        f"Val MAE: {val_metrics['mae']:.4f} - "
        f"Val RMSE: {val_metrics['rmse']:.4f} - "
        f"Val R²: {val_metrics['r2']:.4f} - "
        f"Pearson r: {val_metrics['pearson_r']:.4f} - "
        f"Baseline MAE: {baseline_metrics['baseline_mae']:.4f}"
    )
    print(
        f"Target mean/std: {val_metrics['target_mean']:.4f} / "
        f"{val_metrics['target_std']:.4f} | "
        f"Prediction mean/std: {val_metrics['prediction_mean']:.4f} / "
        f"{val_metrics['prediction_std']:.4f}"
    )


def compute_baseline_metrics(
    task: Task,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
) -> Dict[str, float]:
    if task == Task.BIN_CLASSIFICATION:
        class_counts = np.bincount(train_targets.astype(int))
        # np.argmax deterministically selects the lowest class ID when tied.
        majority_class = int(np.argmax(class_counts))
        baseline_predictions = np.full_like(val_targets, majority_class)
        return {
            "baseline_accuracy": float(np.mean(baseline_predictions == val_targets))
        }

    train_mean = np.mean(train_targets)
    baseline_predictions = np.full_like(val_targets, train_mean)
    return {
        "baseline_mae": float(np.mean(np.abs(val_targets - baseline_predictions)))
    }


def serializable_config(config: ExperimentConfig) -> Dict[str, object]:
    values = asdict(config)
    values["task"] = config.task.value
    values["backbone"] = config.backbone.value
    values["checkpoint_dir"] = str(config.checkpoint_dir)
    values["plot_dir"] = str(config.plot_dir)
    values["results_dir"] = str(config.results_dir)
    return values


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    fold: int,
    seed: int,
    val_indices: Sequence[int],
    val_predictions: np.ndarray,
    val_targets: np.ndarray,
    val_probabilities: Optional[np.ndarray],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    config: ExperimentConfig,
) -> Path:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.checkpoint_dir / (
        f"{config.task.value}_{config.backbone.value}_"
        f"seed_{seed}_fold_{fold + 1}.pt"
    )
    torch.save(
        {
            "epoch": epoch,
            "fold": fold,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_indices": np.asarray(val_indices),
            "val_predictions": val_predictions,
            "val_targets": val_targets,
            "val_probabilities": val_probabilities,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": serializable_config(config),
        },
        checkpoint_path,
    )
    return checkpoint_path


def save_plots(
    predictions: np.ndarray,
    targets: np.ndarray,
    fold: Optional[int],
    seed: int,
    config: ExperimentConfig,
) -> List[Path]:
    config.plot_dir.mkdir(parents=True, exist_ok=True)
    scope = f"seed_{seed}_all_folds" if fold is None else f"seed_{seed}_fold_{fold + 1}"
    prefix = (
        f"{config.task.value}_{config.backbone.value}_{scope}"
    )
    saved_paths = []

    if config.task == Task.BIN_CLASSIFICATION:
        plot_path = config.plot_dir / f"{prefix}_confusion_matrix.png"
        labels = list(range(len(config.class_names)))
        matrix = confusion_matrix(targets, predictions, labels=labels)

        plt.figure()
        plt.imshow(matrix, cmap="Blues")
        plt.colorbar()
        plt.xticks(labels, config.class_names, rotation=30)
        plt.yticks(labels, config.class_names)
        plt.xlabel("Predicted bin")
        plt.ylabel("Ground-truth bin")

        for row in labels:
            for column in labels:
                plt.text(column, row, matrix[row, column], ha="center", va="center")
        plt.title(f"Confusion Matrix: {scope.replace('_', ' ')}")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        saved_paths.append(plot_path)
    else:
        prediction_plot_path = config.plot_dir / f"{prefix}_pred_vs_gt.png"
        residual_plot_path = config.plot_dir / f"{prefix}_residuals.png"

        plt.figure()
        plt.scatter(targets, predictions)
        min_value = min(targets.min(), predictions.min())
        max_value = max(targets.max(), predictions.max())
        plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
        plt.xlabel("Ground Truth mg/cm²")
        plt.ylabel("Predicted mg/cm²")
        plt.title(f"Prediction vs Ground Truth: {scope.replace('_', ' ')}")
        plt.tight_layout()
        plt.savefig(prediction_plot_path, dpi=300)
        plt.close()
        saved_paths.append(prediction_plot_path)

        residuals = targets - predictions
        plt.figure()
        plt.scatter(predictions, residuals)
        plt.axhline(0.0, linestyle="--", color="black")
        plt.xlabel("Predicted mg/cm²")
        plt.ylabel("Residual (ground truth - prediction) mg/cm²")
        plt.title(f"Residual Plot: {scope.replace('_', ' ')}")
        plt.tight_layout()
        plt.savefig(residual_plot_path, dpi=300)
        plt.close()
        saved_paths.append(residual_plot_path)

    return saved_paths


def build_oof_rows(
    val_indices: Sequence[int],
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: Optional[np.ndarray],
    best_epoch: int,
    fold: int,
    seed: int,
    val_dataset: ExperimentDataset,
    config: ExperimentConfig,
) -> List[Dict[str, object]]:
    """Create one out-of-fold record for every validation sample."""

    rows = []
    for position, dataset_index in enumerate(val_indices):
        dataset_index = int(dataset_index)
        row = {
            "task": config.task.value,
            "backbone": config.backbone.value,
            "seed": seed,
            "fold": fold + 1,
            "best_epoch": best_epoch,
            "dataset_index": dataset_index,
            "sample_id": val_dataset.sample_ids[dataset_index],
        }

        if config.task == Task.BIN_CLASSIFICATION:
            target = int(targets[position])
            prediction = int(predictions[position])
            row.update(
                {
                    "target": target,
                    "target_label": config.class_names[target],
                    "prediction": prediction,
                    "prediction_label": config.class_names[prediction],
                }
            )
            if probabilities is None:
                raise ValueError("Classification probabilities were not recorded.")
            for class_id, class_name in enumerate(config.class_names):
                row[f"probability_{metric_label(class_name)}"] = float(
                    probabilities[position, class_id]
                )
        else:
            row["target"] = float(targets[position])
            row["prediction"] = float(predictions[position])
            row["residual"] = float(targets[position] - predictions[position])

        rows.append(row)

    return rows


def save_oof_predictions(
    rows: Sequence[Dict[str, object]], config: ExperimentConfig
) -> Path:
    """Save raw out-of-fold predictions so metrics can be recomputed later."""

    if not rows:
        raise ValueError("No out-of-fold predictions were provided.")

    config.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.results_dir / (
        f"oof_{config.task.value}_{config.backbone.value}.csv"
    )
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def save_seed_plots(
    rows: Sequence[Dict[str, object]], seed: int, config: ExperimentConfig
) -> List[Path]:
    """Save aggregate out-of-fold plots using all four validation folds."""

    predictions = np.asarray([row["prediction"] for row in rows])
    targets = np.asarray([row["target"] for row in rows])
    return save_plots(predictions, targets, None, seed, config)


def run_experiment(
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    fold: int,
    seed: int,
    train_dataset: ExperimentDataset,
    val_dataset: ExperimentDataset,
    stratify_labels: Sequence[int],
    config: ExperimentConfig,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    set_random_seed(seed + fold)
    model = build_model(config, device)
    criterion = build_criterion(config)
    optimizer = torch.optim.SGD(
        trainable_parameters(model),
        lr=config.learning_rate,
        momentum=config.momentum,
    )

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    print_class_counts("Train", train_idx, stratify_labels, config.class_names)
    print_class_counts("Validation", val_idx, stratify_labels, config.class_names)
    print()

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + fold),
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_metrics = None
    best_predictions = None
    best_targets = None
    best_probabilities = None
    best_baseline_metrics = None

    for epoch in range(config.epochs):
        train_metrics, _, train_targets, _ = run_epoch(
            model, train_loader, criterion, config, device, optimizer
        )
        val_metrics, val_predictions, val_targets, val_probabilities = run_epoch(
            model, val_loader, criterion, config, device
        )
        baseline_metrics = compute_baseline_metrics(
            config.task, train_targets, val_targets
        )
        print_epoch_metrics(
            epoch, config, train_metrics, val_metrics, baseline_metrics
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_val_metrics = val_metrics.copy()
            best_predictions = val_predictions.copy()
            best_targets = val_targets.copy()
            best_probabilities = (
                None if val_probabilities is None else val_probabilities.copy()
            )
            best_baseline_metrics = baseline_metrics.copy()
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                epoch,
                fold,
                seed,
                val_idx,
                best_predictions,
                best_targets,
                best_probabilities,
                train_metrics,
                val_metrics,
                config,
            )
            if config.debug:
                print(f"Saved new best checkpoint to {checkpoint_path}")

    result = {
        "best_epoch": float(best_epoch),
        **best_val_metrics,
        **best_baseline_metrics,
    }

    print(f"\nBest epoch: {best_epoch}")
    if config.task == Task.BIN_CLASSIFICATION:
        print(f"Best Val Loss: {result['loss']:.4f}")
        print(f"Best Val Accuracy: {result['accuracy']:.2%}")
        print(f"Best Val Macro-F1: {result['macro_f1']:.4f}")
        for class_name in config.class_names:
            recall_key = f"recall_{metric_label(class_name)}"
            print(f"Recall ({class_name}): {result[recall_key]:.2%}")
        print(f"Majority Baseline Accuracy: {result['baseline_accuracy']:.2%}")
    else:
        print(f"Best Val Loss: {result['loss']:.4f}")
        print(f"Best Val MAE: {result['mae']:.4f}")
        print(f"Best Val RMSE: {result['rmse']:.4f}")
        print(f"Best Val R²: {result['r2']:.4f}")
        print(f"Best Val Pearson r: {result['pearson_r']:.4f}")
        print(f"Baseline MAE: {result['baseline_mae']:.4f}")

    save_plots(best_predictions, best_targets, fold, seed, config)
    oof_rows = build_oof_rows(
        val_idx,
        best_predictions,
        best_targets,
        best_probabilities,
        best_epoch,
        fold,
        seed,
        val_dataset,
        config,
    )
    return result, oof_rows


def summarize_results(
    results: Sequence[Dict[str, float]],
    config: ExperimentConfig,
    heading: str = "Cross-Validation Summary",
) -> None:
    if config.task == Task.BIN_CLASSIFICATION:
        metric_names = ["loss", "accuracy", "macro_f1"]
        metric_names.extend(
            f"recall_{metric_label(class_name)}"
            for class_name in config.class_names
        )
        metric_names.append("baseline_accuracy")
    else:
        metric_names = ("loss", "mae", "rmse", "r2", "pearson_r", "baseline_mae")

    print(f"\n========== {heading} ==========")
    print(f"Task: {config.task.value}")
    print(f"Backbone: {config.backbone.value}")

    for metric_name in metric_names:
        values = np.asarray([result[metric_name] for result in results], dtype=float)
        print(
            f"{metric_name}: {np.nanmean(values):.4f} "
            f"± {np.nanstd(values):.4f}"
        )


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(config: ExperimentConfig = CONFIG) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently on {device}")
    print(f"Task: {config.task.value} | Backbone: {config.backbone.value}")

    if not config.random_seeds:
        raise ValueError("CONFIG.random_seeds must contain at least one seed.")

    train_dataset, val_dataset, stratify_labels = build_datasets(config)
    all_indices = np.arange(len(train_dataset))

    all_results = []
    all_oof_rows = []

    for seed in config.random_seeds:
        set_random_seed(seed)
        print(f"\n================ Seed {seed} ================")
        splitter = StratifiedKFold(
            n_splits=config.folds,
            shuffle=True,
            random_state=seed,
        )

        seed_results = []
        seed_oof_rows = []
        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(all_indices, stratify_labels)
        ):
            print(f"\n========== Fold {fold + 1}/{config.folds} ==========")
            result, oof_rows = run_experiment(
                train_idx,
                val_idx,
                fold,
                seed,
                train_dataset,
                val_dataset,
                stratify_labels,
                config,
                device,
            )
            seed_results.append(result)
            seed_oof_rows.extend(oof_rows)

        summarize_results(seed_results, config, heading=f"Seed {seed} Summary")
        save_seed_plots(seed_oof_rows, seed, config)
        all_results.extend(seed_results)
        all_oof_rows.extend(seed_oof_rows)

    if len(config.random_seeds) > 1:
        summarize_results(all_results, config, heading="All-Seeds Summary")

    output_path = save_oof_predictions(all_oof_rows, config)
    print(f"\nSaved out-of-fold predictions to {output_path}")


if __name__ == "__main__":
    main()


# ONLY IF POOR RESULTS:
# Orient all leaves to point in the same direction.
# Consider: segment -> estimate major axis -> infer tip/base -> rotate/flip.