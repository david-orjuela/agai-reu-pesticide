"""Three-bin CNN training: tree holdout, GPU cache, shuffled minibatches.

Run from the repository root: python -m baseline_pipeline.train_2
Requires baseline_model_2.custom_cnn to accept 1024x1024 RGB and return 3 logits.
CSV labels are used by default; --bin-policy train_quantile fits training tertiles.
"""

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from baseline_pipeline.baseline_model_2 import custom_cnn
from baseline_pipeline.dataset import agai_correct_v3
from baseline_pipeline.experiment_protocol import CLASS_NAMES, encode_bins
from baseline_pipeline.tree_ordinal import resolve_tree_groups


@dataclass(frozen=True)
class Config:
    csv_path: str = '/home/davidorjuela/dev/agai-reu-pesticide/datasets/agai_correct/batch_2_v1/master_icp.csv'
    sample_id_column: str = 'sample_id'
    residue_column: str = 'mg_cm2'
    residue_bin_column: str = 'residue_bin'
    group_column: str | None = None
    bin_policy: str = 'csv'  # csv | train_quantile
    augmentation: str = 'none'  # none | hflip | d4
    seed: int = 42
    batch_size: int = 1
    epochs: int = 1000
    patience: int = 50
    learning_rate: float = 3e-2
    output_root: str = 'results/experiments'
    run_name: str | None = None


CONFIG = Config()
IMAGE_SIZE = (1024, 1024)


def validate_config(c):
    if min(c.batch_size, c.epochs, c.patience) < 1 or not 0 <= c.seed < 2**32:
        raise ValueError('Invalid batch size, epochs, patience, or seed.')
    if not np.isfinite(c.learning_rate) or c.learning_rate <= 0:
        raise ValueError('Learning rate must be finite and positive.')
    if c.bin_policy not in ('csv', 'train_quantile'):
        raise ValueError('Unsupported bin policy.')
    if c.augmentation not in ('none', 'hflip', 'd4'):
        raise ValueError('Unsupported augmentation.')
    if tuple(CLASS_NAMES) != ('low', 'medium', 'high'):
        raise ValueError('Class order must be low, medium, high.')


def split_trees(groups, seed, labels=None):
    """Choose a reproducible whole-tree split using metadata, never model scores.

    With fixed CSV bins, search 10,000 allocations for image-size and class-
    proportion agreement with 80/10/10. Quantile bins retain a random split:
    their thresholds must be fitted only after training membership is fixed.
    """
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 5:
        raise ValueError('At least five tree groups are required.')
    rng = np.random.default_rng(seed)
    n = max(1, round(.1 * len(unique)))
    sizes = np.bincount(inverse)
    counts = None
    if labels is not None:
        counts = np.zeros((len(unique), 3), dtype=int)
        np.add.at(counts, (inverse, labels), 1)
        if np.any((counts > 0).sum(axis=0) < 3):
            raise ValueError('Each class must occur in at least three trees for class-complete partitions.')
    best, best_score = None, float('inf')
    for _ in range(10000 if labels is not None else 1):
        order = rng.permutation(len(unique))
        allocation = (order[2*n:], order[n:2*n], order[:n])
        totals = np.array([sizes[g].sum() for g in allocation])
        score = np.mean(((totals / len(groups) - [.8, .1, .1]) / [.8, .1, .1]) ** 2)
        if counts is not None:
            hist = np.array([counts[g].sum(axis=0) for g in allocation])
            if np.any(hist == 0):
                continue
            proportions = hist / totals[:, None]
            score += np.mean(((proportions - counts.sum(axis=0) / len(groups)) * 3) ** 2)
        if score < best_score:
            best_score, best = score, allocation
    if best is None:
        raise ValueError('No class-complete tree split found; more held-out trees may be needed.')
    return tuple(np.flatnonzero(np.isin(inverse, g)) for g in best)


def check_split(parts, groups):
    if any(len(part) == 0 for part in parts) or not np.array_equal(
        np.sort(np.concatenate(parts)), np.arange(len(groups))
    ):
        raise ValueError('Every image must occur in exactly one nonempty partition.')
    sets = [set(groups[part]) for part in parts]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Tree leakage across partitions.')



def cache_images(base, residue, device):
    """Decode and transfer each image individually, exactly once."""
    shape = (len(base), 3, *IMAGE_SIZE)
    required = int(np.prod(shape)) * 4
    free, _ = torch.cuda.mem_get_info(device)
    print(f'Image cache: {required / 2**30:.2f} GiB; free GPU memory: {free / 2**30:.2f} GiB')
    if required >= free:
        raise RuntimeError('Images do not fit on the GPU; training also needs additional VRAM.')
    images = torch.empty(shape, dtype=torch.float32, device=device)
    for i in range(len(base)):
        image, target = base[i]
        if tuple(image.shape) != (3, *IMAGE_SIZE) or image.dtype != torch.float32:
            raise ValueError(f'Image {i}: expected {(3, *IMAGE_SIZE)} float32, '
                             f'got {tuple(image.shape)} {image.dtype}.')
        if not torch.isfinite(image).all() or not np.isclose(float(target), residue[i], rtol=1e-5, atol=1e-7):
            raise ValueError(f'Invalid pixels or dataset/metadata target mismatch at index {i}.')
        images[i].copy_(image)
    torch.cuda.synchronize(device)
    print(f'Cached {len(images)} individual images on {device}.')
    return images


def augment(images, mode, generator):
    if mode == 'none':
        return images
    flip = torch.rand(len(images), device=images.device, generator=generator) < .5
    images = torch.where(flip[:, None, None, None], images.flip(-1), images)
    if mode == 'd4':
        rotations = torch.randint(4, (len(images),), device=images.device, generator=generator)
        original = images
        for k in (1, 2, 3):
            images = torch.where((rotations == k)[:, None, None, None],
                                 torch.rot90(original, k, (-2, -1)), images)
    return images.contiguous()


def epoch(model, images, labels, indices, c, generator, optimizer=None):
    training = optimizer is not None
    model.train(training)
    order = indices
    if training:
        order = indices[torch.randperm(len(indices), device=indices.device, generator=generator)]
    total_loss = torch.zeros((), device=images.device)
    targets, scores = [], []
    with torch.set_grad_enabled(training):
        for ids in order.split(c.batch_size):
            batch = images.index_select(0, ids)  # Copy: never augment the cached originals.
            target = labels.index_select(0, ids)
            if training:
                batch = augment(batch, c.augmentation, generator)
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            if logits.shape != (len(ids), 3):
                raise ValueError(f'CNN must return (B, 3) logits, got {tuple(logits.shape)}.')
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss; inspect the data and learning rate.')
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.detach() * len(ids)
            targets.append(target.detach())
            scores.append(logits.detach())
    # Transfer metrics once per epoch, rather than copying predictions every batch.
    y = torch.cat(targets).cpu().numpy()
    logits = torch.cat(scores).cpu().numpy()
    predicted = logits.argmax(1)
    _, recall, f1, _ = precision_recall_fscore_support(y, predicted, labels=[0, 1, 2], zero_division=0)
    metrics = {'loss': float(total_loss.item() / len(indices)),
               'accuracy': float(accuracy_score(y, predicted)), 'macro_f1': float(f1.mean()),
               **{f'recall_{name}': float(recall[i]) for i, name in enumerate(CLASS_NAMES)}}
    return metrics, logits


def fit(model, images, labels, train, validation, c, generator, out):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=c.learning_rate, momentum=0.0, weight_decay=0.0)
    best_loss, best_epoch, best_state, stale = float('inf'), 0, None, 0
    history = []
    for number in range(1, c.epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        train_metrics, _ = epoch(model, images, labels, train, c, generator, optimizer)
        val_metrics, _ = epoch(model, images, labels, validation, c, generator)
        history.append({'epoch': number, 'learning_rate': lr,
                        **{f'train_{k}': v for k, v in train_metrics.items()},
                        **{f'validation_{k}': v for k, v in val_metrics.items()}})
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        print(f'Epoch {number:3d} | train loss {train_metrics["loss"]:.4f} '
              f'acc {train_metrics["accuracy"]:.3f} | val loss {val_metrics["loss"]:.4f} '
              f'acc {val_metrics["accuracy"]:.3f} F1 {val_metrics["macro_f1"]:.3f} | lr {lr:g}')
        if val_metrics['loss'] < best_loss:
            best_loss, best_epoch, stale = val_metrics['loss'], number, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= c.patience:
            print(f'Early stopping: {c.patience} epochs without validation-loss improvement.')
            break
    model.load_state_dict(best_state)
    return best_epoch, best_state, history[best_epoch - 1]


def main(c=CONFIG):
    validate_config(c)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; request a GPU allocation and use CUDA-enabled PyTorch.')
    device = torch.device('cuda')
    random.seed(c.seed)
    np.random.seed(c.seed)
    torch.manual_seed(c.seed)
    torch.cuda.manual_seed_all(c.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'GPU: {torch.cuda.get_device_name(device)} | batch size: {c.batch_size}')
    # ToTensor converts uint8 RGB to float32 and divides every channel by 255.
    transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize(IMAGE_SIZE, antialias=True),
        transforms.ToTensor(),
    ])
    base = agai_correct_v3(c.csv_path, transform=transform)
    metadata = base.icp_data.reset_index(drop=True).copy()
    required = [c.sample_id_column, c.residue_column]
    if c.bin_policy == 'csv':
        required.append(c.residue_bin_column)
    if c.group_column:
        required.append(c.group_column)
    if not len(base) or len(metadata) != len(base) or metadata[required].isna().any().any():
        raise ValueError('Empty dataset, inconsistent lengths, or missing metadata.')
    ids = metadata[c.sample_id_column].astype(str).to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Sample IDs must be unique.')
    residue = pd.to_numeric(metadata[c.residue_column], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(residue).all():
        raise ValueError('Residue values must be finite.')
    explicit = None if c.group_column is None else metadata[c.group_column].astype(str).to_numpy()
    tree_ids, groups = map(np.asarray, resolve_tree_groups(ids, explicit))
    split_labels = (np.asarray(encode_bins(metadata[c.residue_bin_column]), dtype=np.int64)
                    if c.bin_policy == 'csv' else None)
    parts = split_trees(groups, c.seed, split_labels)
    check_split(parts, groups)
    check_split(parts, tree_ids)
    train, validation, test = parts
    thresholds = None
    if c.bin_policy == 'csv':
        labels = np.asarray(encode_bins(metadata[c.residue_bin_column]))
    else:
        thresholds = np.quantile(residue[train], [1/3, 2/3])
        if thresholds[0] >= thresholds[1]:
            raise ValueError('Training tertiles cannot define three distinct bins.')
        labels = np.searchsorted(thresholds, residue, side='left')
    if not np.isin(labels, [0, 1, 2]).all() or len(np.unique(labels[train])) != 3:
        raise ValueError('Labels must be 0/1/2, with all three classes present in training.')
    labels = labels.astype(np.int64)
    run_name = c.run_name or f'cnn_ce_seed_{c.seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    if Path(run_name).name != run_name or run_name in ('.', '..'):
        raise ValueError('run_name must be a single folder name.')
    out = Path(c.output_root) / run_name
    out.mkdir(parents=True, exist_ok=False)
    counts, membership = [], np.empty(len(ids), dtype=object)
    print(f'Run: {run_name} | total images: {len(ids)}')
    for role, part in zip(('train', 'validation', 'test'), parts):
        membership[part] = role
        class_counts = np.bincount(labels[part], minlength=3)
        row = {'partition': role, 'n_images': len(part), 'percent_images': 100*len(part)/len(ids),
               'n_tree_groups': len(np.unique(groups[part])),
               **dict(zip(CLASS_NAMES, class_counts.tolist()))}
        counts.append(row)
        print(f'{role:10s}: {len(part)} images ({row["percent_images"]:.1f}%), '
              f'{row["n_tree_groups"]} groups; low/medium/high = {class_counts.tolist()}')
        if np.any(class_counts == 0):
            print(f'WARNING: {role} lacks a class; three-class metrics have limited coverage.')
    pd.DataFrame(counts).to_csv(out / 'split_counts.csv', index=False)
    splits = pd.DataFrame({'sample_id': ids, 'tree_id': tree_ids, 'group': groups,
                           'partition': membership, 'target': labels, 'residue_mg_cm2': residue})
    splits.to_csv(out / 'splits.csv', index=False)
    manifest = {'config': asdict(c), 'loss': 'cross_entropy', 'optimizer': 'SGD',
                'momentum': 0.0, 'weight_decay': 0.0, 'scheduler': None, 'class_names': list(CLASS_NAMES), 'image_size': list(IMAGE_SIZE),
                'input': 'Resized 1024x1024 RGB float32 [0,1]; pixel / 255 only; no mean/std normalization', 'selection_metric': 'validation_loss',
                'split_method': '10000 metadata-only candidates' if c.bin_policy == 'csv' else 'random trees',
                'thresholds': None if thresholds is None else thresholds.tolist(),
                'csv_sha256': hashlib.sha256(Path(c.csv_path).read_bytes()).hexdigest(),
                'gpu': torch.cuda.get_device_name(device), 'torch_version': str(torch.__version__)}
    (out / 'config.json').write_text(json.dumps(manifest, indent=2))
    images = cache_images(base, residue, device)
    gpu_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    gpu_train, gpu_val, gpu_test = [torch.as_tensor(part, dtype=torch.long, device=device) for part in parts]
    generator = torch.Generator(device=device).manual_seed(c.seed)  # Created once, never reset per epoch.
    model = custom_cnn(num_bins=3).to(device)
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError('The custom CNN must train all parameters.')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters()):,}')
    best_epoch, state, best_row = fit(model, images, gpu_labels, gpu_train, gpu_val, c, generator, out)
    torch.save({'model_state_dict': state, 'best_epoch': best_epoch, 'manifest': manifest,
                'model_description': str(model), 'sample_ids': ids.tolist(), 'group_ids': groups.tolist(),
                'train_indices': train.tolist(), 'validation_indices': validation.tolist(),
                'test_indices': test.tolist(), 'labels': labels.tolist()}, out / 'best.pt')
    # Test is evaluated only after training and best-checkpoint restoration.
    test_metrics, logits = epoch(model, images, gpu_labels, gpu_test, c, generator)
    prediction = logits.argmax(1)  # Highest raw logit selects the class.
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    majority = int(np.bincount(labels[train], minlength=3).argmax())
    test_metrics['majority_baseline_accuracy'] = float(np.mean(labels[test] == majority))
    report = {'best_epoch': best_epoch,
              'validation': {k.removeprefix('validation_'): v for k, v in best_row.items() if k.startswith('validation_')},
              'test': test_metrics}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    predictions = splits.iloc[test].copy()
    predictions['prediction'] = prediction
    predictions['predicted_label'] = np.asarray(CLASS_NAMES)[prediction]
    for i, name in enumerate(CLASS_NAMES):
        predictions[f'logit_{name}'] = logits[:, i]
        predictions[f'probability_{name}'] = probabilities[:, i]
    predictions.to_csv(out / 'test_predictions.csv', index=False)
    pd.DataFrame(confusion_matrix(labels[test], prediction, labels=[0, 1, 2]),
                 index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out / 'confusion_matrix.csv', index_label='actual')
    print(f'Best epoch: {best_epoch} | test: {test_metrics}\nSaved: {out}')


def parse_config():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('csv_path', 'sample_id_column', 'residue_column', 'residue_bin_column', 'group_column', 'output_root', 'run_name'):
        parser.add_argument('--' + name.replace('_', '-'))
    for name in ('seed', 'batch_size', 'epochs', 'patience'):
        parser.add_argument('--' + name.replace('_', '-'), type=int)
    for name in ('learning_rate',):
        parser.add_argument('--' + name.replace('_', '-'), type=float)
    parser.add_argument('--bin-policy', choices=('csv', 'train_quantile'))
    parser.add_argument('--augmentation', choices=('none', 'hflip', 'd4'))
    return replace(CONFIG, **{k: v for k, v in vars(parser.parse_args()).items() if v is not None})


if __name__ == '__main__':
    main(parse_config())
