"""
AGAI_REU_2026
David Orjuela
Undergraduate Student under Dr. Chen
Training orchestration for pesticide-residue experiments.

Change CONFIG.task and CONFIG.backbone to select an experiment. The rest of
the training, evaluation, plotting, and checkpoint logic adapts automatically.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, r2_score
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
    class_names: Tuple[str, ...] = ("low", "medium", "high")

    learning_rate: float = 0.001
    momentum: float = 0.9
    batch_size: int = 4
    epochs: int = 30
    folds: int = 4
    random_seed: int = 42
    num_workers: int = 0

    checkpoint_dir: Path = Path("checkpoints")
    plot_dir: Path = Path("plots")
    debug: bool = False

    @property
    def num_outputs(self) -> int:
        if self.task == Task.BIN_CLASSIFICATION:
            return len(self.class_names)
        return 1


# To switch experiments, change only these settings. DINOv3 is wired into the
# model factory below and will work once frozen_dinoV3 is implemented.
CONFIG = ExperimentConfig(
    task=Task.BIN_CLASSIFICATION,
    backbone=Backbone.RESNET50,
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
    ) -> None:
        self.base_dataset = base_dataset
        self.task = task
        self.class_targets = class_targets

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


def build_datasets(config: ExperimentConfig):
    train_transform, val_transform = make_transforms()
    train_base = agai_correct_v3(config.csv_path, transform=train_transform)
    val_base = agai_correct_v3(config.csv_path, transform=val_transform)
    class_targets = read_class_targets(train_base, config)

    train_dataset = ExperimentDataset(train_base, config.task, class_targets)
    val_dataset = ExperimentDataset(val_base, config.task, class_targets)
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
    elif task == Task.ICP_REGRESSION:
        targets = targets.float().reshape(-1)
        predictions = outputs.reshape(-1)
        loss = criterion(predictions, targets)
    else:
        raise ValueError(f"Unsupported task: {task}")

    return loss, predictions, targets


def compute_metrics(
    task: Task,
    loss: float,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    metrics = {"loss": loss}

    if task == Task.BIN_CLASSIFICATION:
        metrics["accuracy"] = float(np.mean(predictions == targets))
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
        metrics["pearson_r"] = float(pearsonr(targets, predictions).statistic)

    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    task: Task,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
):
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.set_grad_enabled(is_training):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            if is_training:
                optimizer.zero_grad()

            loss, predictions, prepared_targets = prepare_batch(
                model, images, targets, criterion, task
            )

            if is_training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            all_predictions.extend(predictions.detach().cpu().tolist())
            all_targets.extend(prepared_targets.detach().cpu().tolist())

    predictions_np = np.asarray(all_predictions)
    targets_np = np.asarray(all_targets)
    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(task, epoch_loss, predictions_np, targets_np)
    return metrics, predictions_np, targets_np


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
        majority_class = Counter(train_targets.tolist()).most_common(1)[0][0]
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
    return values


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    fold: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    config: ExperimentConfig,
) -> Path:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.checkpoint_dir / (
        f"{config.task.value}_{config.backbone.value}_fold_{fold + 1}.pt"
    )
    torch.save(
        {
            "epoch": epoch,
            "fold": fold,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": serializable_config(config),
        },
        checkpoint_path,
    )
    return checkpoint_path


def save_plot(
    predictions: np.ndarray,
    targets: np.ndarray,
    fold: int,
    config: ExperimentConfig,
) -> Path:
    config.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = config.plot_dir / (
        f"{config.task.value}_{config.backbone.value}_fold_{fold + 1}.png"
    )

    plt.figure()
    if config.task == Task.BIN_CLASSIFICATION:
        labels = list(range(len(config.class_names)))
        matrix = confusion_matrix(targets, predictions, labels=labels)
        plt.imshow(matrix, cmap="Blues")
        plt.colorbar()
        plt.xticks(labels, config.class_names, rotation=30)
        plt.yticks(labels, config.class_names)
        plt.xlabel("Predicted bin")
        plt.ylabel("Ground-truth bin")

        for row in labels:
            for column in labels:
                plt.text(column, row, matrix[row, column], ha="center", va="center")
    else:
        plt.scatter(targets, predictions)
        min_value = min(targets.min(), predictions.min())
        max_value = max(targets.max(), predictions.max())
        plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
        plt.xlabel("Ground Truth mg/cm²")
        plt.ylabel("Predicted mg/cm²")

    plt.title(f"Fold {fold + 1}: {config.task.value} ({config.backbone.value})")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


def run_experiment(
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    fold: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    stratify_labels: Sequence[int],
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
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
        generator=torch.Generator().manual_seed(config.random_seed + fold),
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
    best_baseline_metrics = None

    for epoch in range(config.epochs):
        train_metrics, _, train_targets = run_epoch(
            model, train_loader, criterion, config.task, device, optimizer
        )
        val_metrics, val_predictions, val_targets = run_epoch(
            model, val_loader, criterion, config.task, device
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
            best_baseline_metrics = baseline_metrics.copy()
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                epoch,
                fold,
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
        print(f"Majority Baseline Accuracy: {result['baseline_accuracy']:.2%}")
    else:
        print(f"Best Val Loss: {result['loss']:.4f}")
        print(f"Best Val MAE: {result['mae']:.4f}")
        print(f"Best Val RMSE: {result['rmse']:.4f}")
        print(f"Best Val R²: {result['r2']:.4f}")
        print(f"Best Val Pearson r: {result['pearson_r']:.4f}")
        print(f"Baseline MAE: {result['baseline_mae']:.4f}")

    save_plot(best_predictions, best_targets, fold, config)
    return result


def summarize_results(results: Sequence[Dict[str, float]], config: ExperimentConfig) -> None:
    if config.task == Task.BIN_CLASSIFICATION:
        metric_names = ("loss", "accuracy", "baseline_accuracy")
    else:
        metric_names = ("loss", "mae", "rmse", "r2", "pearson_r", "baseline_mae")

    print("\n========== Cross-Validation Summary ==========")
    print(f"Task: {config.task.value}")
    print(f"Backbone: {config.backbone.value}")

    for metric_name in metric_names:
        values = np.asarray([result[metric_name] for result in results], dtype=float)
        print(
            f"{metric_name}: {np.nanmean(values):.4f} "
            f"± {np.nanstd(values):.4f}"
        )


def main(config: ExperimentConfig = CONFIG) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently on {device}")
    print(f"Task: {config.task.value} | Backbone: {config.backbone.value}")

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    train_dataset, val_dataset, stratify_labels = build_datasets(config)
    all_indices = np.arange(len(train_dataset))
    splitter = StratifiedKFold(
        n_splits=config.folds,
        shuffle=True,
        random_state=config.random_seed,
    )

    results = []
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(all_indices, stratify_labels)
    ):
        print(f"\n========== Fold {fold + 1}/{config.folds} ==========")
        result = run_experiment(
            train_idx,
            val_idx,
            fold,
            train_dataset,
            val_dataset,
            stratify_labels,
            config,
            device,
        )
        results.append(result)

    summarize_results(results, config)


if __name__ == "__main__":
    main()


# ONLY IF POOR RESULTS:
# Orient all leaves to point in the same direction.
# Consider: segment -> estimate major axis -> infer tip/base -> rotate/flip.
