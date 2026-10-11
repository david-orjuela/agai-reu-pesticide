"""Audit saved PlantNet multiscale/tile models without retraining or resplitting.

Examples (from the repository root):
  uv run plantnet_treatment_pilot/audit_model.py \
    --training-script plantnet_treatment_pilot/train_unlabeled.py \
    --run-dir plantnet_treatment_pilot/results_tree_split_20epochs/B_treatment_pretrained/icp

  uv run plantnet_treatment_pilot/audit_model.py \
    --training-script anthony/train_baseline_improved3.py \
    --run-dir anthony/plantnet_resnet18_multiscale_tiles_results

Default: check metadata and image readability in ALL saved splits, but infer only
on validation. Add --splits train validation for development diagnostics, or
--splits test explicitly for a retrospective test audit. No checkpoint selection.
Use the EXACT training-script version and settings used to produce the checkpoint.
Per-view results diagnose an existing model; they are not independently trained
architecture ablations. Use only trusted Python scripts and checkpoints.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torchvision.models import resnet18
from tqdm import tqdm

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIC decoding failures will be recorded in input_inventory.csv.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def derive_tree(row):
    # Confirmed dataset convention: control leaves all belong to one tree.
    sample = str(row.get('sample_id', '')).strip().lower()
    match = re.fullmatch(r'(\d+)\s+ppm\s+(\d+)\s+(\d+)', sample)
    if match:
        return f'{int(match[1])} ppm {int(match[2])}'
    if re.fullmatch(r'c\s*\d+', sample):
        return 'control_tree'
    value = str(row.get('tree_id', '')).strip()
    return value if value.lower() not in ('', 'unknown', 'nan', 'none') else 'unknown'


def read_manifest(path, split):
    frame = pd.read_csv(path)
    if 'image_path' not in frame:
        raise ValueError(f'{path}: no image_path column')
    frame['csv_image_path'] = frame.image_path
    def resolve(value):
        image = Path(str(value)).expanduser()
        return str((image if image.is_absolute() else path.parent / image).resolve())
    frame['image_path'] = frame.image_path.map(resolve)
    frame['tree_id'] = frame.apply(derive_tree, axis=1)
    frame['split'] = split
    frame['row_index'] = np.arange(len(frame))
    return frame


def summarize(y, pred, classes):
    labels = list(range(len(classes)))
    recalls = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
            **{f'recall_{name}': float(value) for name, value in zip(classes, recalls)},
            **{f'support_{name}': int(np.sum(np.asarray(y) == i))
               for i, name in enumerate(classes)}}


def scan_inputs(frames, label_column, out):
    records = []
    for split, frame in frames.items():
        for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f'Check {split}'):
            item = {'split': split, 'row_index': int(row.row_index),
                    'image_path': row.image_path, 'sample_id': row.get('sample_id', ''),
                    'tree_id': row.tree_id, 'label': row.get(label_column, ''),
                    'nominal_treatment': row.get('nominal_treatment', ''),
                    'extension': Path(row.image_path).suffix.lower(), 'error': ''}
            try:
                path = Path(row.image_path)
                item.update(file_bytes=path.stat().st_size, file_sha256=sha256(path))
                with Image.open(path) as image:
                    item.update(format=image.format, mode=image.mode, width=image.width,
                                height=image.height, exif_orientation=image.getexif().get(274, 1))
                    rgb = np.asarray(image.convert('RGB'))  # Match training: no added EXIF transpose.
                item['pixel_sha256'] = hashlib.sha256(
                    str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
                for i, channel in enumerate('rgb'):
                    item[f'{channel}_mean'] = float(rgb[..., i].mean() / 255)
                    item[f'{channel}_std'] = float(rgb[..., i].std() / 255)
                item['aspect_ratio'] = item['width'] / item['height']
                item['near_black_fraction'] = float((rgb.max(axis=2) < 5).mean())
                item['near_white_fraction'] = float((rgb.min(axis=2) > 250).mean())
            except Exception as exc:
                item['error'] = f'{type(exc).__name__}: {exc}'
            records.append(item)
    inventory = pd.DataFrame(records)
    inventory.to_csv(out / 'input_inventory.csv', index=False)
    duplicates = []
    for key in ('image_path', 'file_sha256', 'pixel_sha256'):
        if key not in inventory:
            continue
        for value, group in inventory.dropna(subset=[key]).groupby(key):
            if len(group) > 1:
                duplicates.append({'match_type': key, 'match_value': value,
                                   'count': len(group), 'cross_split': group.split.nunique() > 1,
                                   'splits': '|'.join(sorted(group.split.unique())),
                                   'paths': json.dumps(group.image_path.tolist())})
    pd.DataFrame(duplicates, columns=['match_type', 'match_value', 'count', 'cross_split',
                                     'splits', 'paths']).to_csv(out / 'duplicate_images.csv', index=False)
    shared_trees = []
    for tree, group in inventory[inventory.tree_id != 'unknown'].groupby('tree_id'):
        if group.split.nunique() > 1:
            shared_trees.append({'tree_id': tree, 'splits': '|'.join(sorted(group.split.unique())),
                                 'images': len(group)})
    pd.DataFrame(shared_trees, columns=['tree_id', 'splits', 'images']).to_csv(
        out / 'tree_overlap.csv', index=False)
    inventory.groupby(['split', 'label', 'extension'], dropna=False).size().rename('images').to_csv(
        out / 'label_format_counts.csv')
    return inventory, shared_trees, duplicates


def load_source(path):
    spec = importlib.util.spec_from_file_location('audit_training_source', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)  # __main__ training block is not executed.
    for name in ('LeafDataset', 'make_transforms', 'MultiScaleTilePlantNet', 'set_seed'):
        if not hasattr(module, name):
            raise ValueError(f'Training script does not expose {name}')
    return module


def dataset(module, frame, label_column, classes, training=False):
    options = {'training': training}
    parameters = inspect.signature(module.LeafDataset).parameters
    if 'label_column' in parameters:
        options.update(label_column=label_column, class_names=classes)
    elif label_column != module.LABEL_COLUMN or tuple(classes) != tuple(module.CLASS_NAMES):
        raise ValueError('This older training script supports only its original label mapping.')
    return module.LeafDataset(frame, *module.make_transforms(), **options)


@torch.inference_mode()
def view_outputs(model, inputs, device):
    large, small, tiles = [value.unsqueeze(0).to(device) for value in inputs]
    if tiles.shape[1] != 9:
        raise ValueError('This audit expects the current nine-tile architecture.')
    large_logits = model.backbone(large)
    small_logits = model.backbone(small)
    tile_logits = model.backbone(tiles.flatten(0, 1))
    combined = (large_logits + small_logits + tile_logits.mean(0, keepdim=True)) / 3
    actual = model(large, small, tiles)
    if not torch.allclose(actual, combined, rtol=1e-4, atol=1e-5):
        raise ValueError('Training forward differs from audited score averaging; audit must be adapted.')
    # Returning actual forward probabilities makes saved-prediction checks meaningful.
    logits = torch.cat([large_logits, small_logits, tile_logits,
                        tile_logits.mean(0, keepdim=True), actual], dim=0)
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite model logits detected.')
    return logits.cpu().numpy(), logits.softmax(-1).cpu().numpy()


def tensor_rgb(tensor, module):
    mean = np.asarray(module.IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.asarray(module.IMAGENET_STD).reshape(1, 1, 3)
    return np.clip(tensor.detach().cpu().permute(1, 2, 0).numpy() * std + mean, 0, 1)


def save_sheet(row, inputs, probs, classes, module, path, title):
    with Image.open(row.image_path) as image:
        original = image.convert('RGB')
        original.thumbnail((700, 700))
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    axes = axes.ravel()
    axes[0].imshow(original)
    axes[0].set_title('Original image (as loaded)')
    views = [inputs[0], inputs[1], *inputs[2]]
    names = ['Global 448', 'Global 224'] + [f'Tile {i//3+1},{i%3+1}' for i in range(9)]
    # Use configured sizes in case source settings differ.
    names[:2] = [f'Global {module.IMAGE_SIZE}', f'Global {module.SMALL_IMAGE_SIZE}']
    for i, (value, name) in enumerate(zip(views, names)):
        axes[i+1].imshow(tensor_rgb(value, module))
        label = classes[int(probs[i].argmax())]
        axes[i+1].set_title(f'{name}: {label} ({probs[i].max():.0%})', fontsize=10)
    for ax in axes:
        ax.axis('off')
    pred = classes[int(probs[-1].argmax())]
    fig.suptitle(f'{title}\nTree: {row.tree_id} | Sample: {row.get("sample_id", "")} | '
                 f'Combined: {pred} ({probs[-1].max():.1%})', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def audit_split(module, model, frame, split, classes, label_column, out, args, inventory, device):
    target_dir = out / split
    target_dir.mkdir()
    ds = dataset(module, frame, label_column, classes)
    view_names = ['global_large', 'global_small'] + [f'tile_{i+1}' for i in range(9)] + ['tiles_mean', 'combined']
    predictions, view_rows, tensor_rows, cached_probs = [], [], [], {}
    bad_rows = set(inventory.loc[(inventory.split == split) & inventory.error.ne(''), 'row_index'])
    for index in tqdm(range(len(ds)), desc=f'Infer {split}'):
        if index in bad_rows:
            continue
        values = ds[index]
        inputs, target = values[:3], int(values[3])
        expected = classes.index(frame.iloc[index][label_column])
        if target != expected:
            raise ValueError(f'Dataset label mapping mismatch at {split} row {index}')
        for name, value in zip(['global_large', 'global_small', 'tiles'], inputs):
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite input tensor: {split}, row {index}, {name}')
            tensor_rows.append({'row_index': index, 'view': name, 'shape': str(tuple(value.shape)),
                                'min': float(value.min()), 'max': float(value.max()),
                                'mean': float(value.mean()), 'std': float(value.std())})
        logits, probs = view_outputs(model, inputs, device)
        cached_probs[index] = probs
        source = frame.iloc[index]
        result = source.to_dict()
        prediction = int(probs[-1].argmax())
        result.update(target_index=target, prediction_index=prediction,
                      prediction_label=classes[prediction], confidence=float(probs[-1].max()),
                      correct=prediction == target,
                      branch_disagreement=len(set(probs[[0,1,11]].argmax(axis=1).tolist())) > 1)
        result.update({f'probability_{name}': float(probs[-1,i]) for i,name in enumerate(classes)})
        predictions.append(result)
        for name, scores, probability in zip(view_names, logits, probs):
            view_rows.append({'row_index': index, 'image_path': source.image_path, 'view': name,
                              'target_index': target, 'prediction_index': int(probability.argmax()),
                              'prediction_label': classes[int(probability.argmax())],
                              **{f'logit_{c}': float(v) for c,v in zip(classes,scores)},
                              **{f'probability_{c}': float(v) for c,v in zip(classes,probability)}})
    if not predictions:
        raise ValueError(f'No readable images in {split}; inspect input_inventory.csv.')
    pred = pd.DataFrame(predictions)
    views = pd.DataFrame(view_rows)
    pred.to_csv(target_dir / 'predictions.csv', index=False)
    pred.loc[~pred.correct].sort_values('confidence', ascending=False).to_csv(target_dir/'errors.csv', index=False)
    views.to_csv(target_dir/'per_view_predictions.csv', index=False)
    pd.DataFrame(tensor_rows).to_csv(target_dir/'tensor_stats.csv', index=False)
    metrics = summarize(pred.target_index, pred.prediction_index, classes)
    metrics.update(expected_images=len(frame), failed_images=len(bad_rows), complete=not bad_rows)
    (target_dir/'metrics.json').write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(confusion_matrix(pred.target_index,pred.prediction_index,labels=range(len(classes))),
                 index=classes,columns=classes).to_csv(target_dir/'confusion_matrix.csv')
    tree_metrics = [{'tree_id': tree, **summarize(group.target_index,group.prediction_index,classes)}
                    for tree,group in pred.groupby('tree_id')]
    pd.DataFrame(tree_metrics).to_csv(target_dir/'per_tree_metrics.csv', index=False)
    pd.DataFrame([{'view':name, **summarize(g.target_index,g.prediction_index,classes)}
                  for name,g in views.groupby('view')]).to_csv(target_dir/'per_view_metrics.csv', index=False)
    # Choose a mixture of representative examples and confident mistakes.
    selected = []
    for _, group in pred.groupby('target_index'):
        selected.append(int(group.iloc[0].row_index))
    candidates = pd.concat([pred.loc[~pred.correct].sort_values('confidence',ascending=False),
                            pred.loc[pred.branch_disagreement],pred])
    selected = list(dict.fromkeys(selected + candidates.row_index.astype(int).tolist()))[:args.max_sheets]
    for index in selected:
        row = frame.iloc[index]
        inputs = ds[index][:3]
        save_sheet(row, inputs, cached_probs[index], classes, module,
                   target_dir/f'row_{index:04d}_eval.png', f'{split} row {index} | True: {row[label_column]}')
        if args.augment_preview:
            module.set_seed(42 + index)
            augmented = dataset(module,frame,label_column,classes,training=True)[index][:3]
            _, probs = view_outputs(model,augmented,device)
            save_sheet(row,augmented,probs,classes,module,target_dir/f'row_{index:04d}_augmented.png',
                       f'Augmentation preview (eval-mode model; not scored) | True: {row[label_column]}')
    # Existing test predictions are used only when test was explicitly requested.
    saved = args.run_dir / f'{split}_predictions.csv'
    agreement = None
    if saved.exists() and 'prediction_index' in pd.read_csv(saved,nrows=0):
        prior = read_manifest(saved, split)
        if prior.image_path.duplicated().any() or pred.image_path.duplicated().any():
            raise ValueError('Duplicate paths prevent unambiguous saved-prediction comparison.')
        joined = pred[['image_path','prediction_index']].merge(
            prior[['image_path','prediction_index']],on='image_path',suffixes=('_audit','_saved'),how='outer',indicator=True)
        joined.to_csv(target_dir/'saved_prediction_comparison.csv',index=False)
        both=joined.loc[joined['_merge']=='both']
        agreement={'matched_images':len(both),'unmatched_images':int((joined['_merge']!='both').sum()),
                   'different_predictions':int((both.prediction_index_audit!=both.prediction_index_saved).sum())}
    return {**metrics, 'saved_prediction_comparison': agreement}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--training-script',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True,help='Folder containing saved split CSVs and checkpoint')
    parser.add_argument('--checkpoint',type=Path,help='Explicit checkpoint; otherwise discover one best_*.pt')
    parser.add_argument('--output-dir',type=Path,help='Defaults to run-dir/audit; must not already exist')
    parser.add_argument('--splits',nargs='+',choices=['train','validation','test'],default=['validation'])
    parser.add_argument('--max-sheets',type=int,default=12)
    parser.add_argument('--augment-preview',action='store_true')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--label-column',help='Override residue_bin/nominal_treatment inference')
    args=parser.parse_args()
    if args.max_sheets<0: parser.error('--max-sheets must be nonnegative')
    args.training_script=args.training_script.expanduser().resolve()
    args.run_dir=args.run_dir.expanduser().resolve()
    out=(args.output_dir or args.run_dir/'audit').expanduser().resolve()
    if out.exists(): parser.error(f'Output directory exists; choose a new --output-dir: {out}')
    checkpoint=args.checkpoint
    if checkpoint is None:
        candidates=[args.run_dir/name for name in ['best_icp.pt','best_treatment.pt','best_model.pt'] if (args.run_dir/name).is_file()]
        if len(candidates)!=1: parser.error('Specify --checkpoint; expected exactly one best checkpoint.')
        checkpoint=candidates[0]
    checkpoint=checkpoint.expanduser().resolve()
    source=load_source(args.training_script)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    classes=tuple(str(value) for value in saved['class_names'])
    label_column=args.label_column or ('nominal_treatment' if all(c.isdigit() for c in classes) else 'residue_bin')
    if saved.get('image_size',source.IMAGE_SIZE)!=source.IMAGE_SIZE:
        raise ValueError('Checkpoint image_size differs from the supplied training script.')
    # Build directly from saved weights. No PlantNet download or fresh training.
    model=source.MultiScaleTilePlantNet(resnet18(weights=None,num_classes=len(classes)))
    model.load_state_dict(saved['model_state_dict'],strict=True)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    model.to(device).eval()
    source.set_seed(42)
    frames={}
    for split in ['train','validation','test']:
        frame=read_manifest(args.run_dir/f'{split}_split.csv',split)
        if label_column not in frame: raise ValueError(f'{split}: missing {label_column}')
        if frame[label_column].isna().any(): raise ValueError(f'{split}: missing labels')
        if label_column=='nominal_treatment':
            values=pd.to_numeric(frame[label_column],errors='raise')
            if not values.isin([int(c) for c in classes]).all(): raise ValueError('Unexpected treatment labels')
            frame[label_column]=values.astype(int).astype(str)
        else: frame[label_column]=frame[label_column].astype(str).str.strip().str.lower()
        if not set(frame[label_column])<=set(classes): raise ValueError(f'{split}: unexpected labels')
        frames[split]=frame
    out.mkdir(parents=True)
    inventory, overlap, duplicates=scan_inputs(frames,label_column,out)
    result={'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
            'training_script':str(args.training_script),'script_sha256':sha256(args.training_script),
            'checkpoint_stage':saved.get('stage'),'checkpoint_epoch':saved.get('epoch'),
            'classes':classes,'device':str(device),'torch_version':torch.__version__,
            'selected_splits':args.splits,'tree_overlap_groups':len(overlap),
            'unknown_tree_rows':int(inventory.tree_id.eq('unknown').sum()),
            'unreadable_images':int(inventory.error.ne('').sum()),
            'cross_split_duplicate_groups':sum(bool(d['cross_split']) for d in duplicates),
            'normalization_mean':source.IMAGENET_MEAN,'normalization_std':source.IMAGENET_STD,
            'configured_sizes':[source.IMAGE_SIZE,source.SMALL_IMAGE_SIZE,source.TILE_IMAGE_SIZE],
            'results':{}}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    for split in dict.fromkeys(args.splits):
        result['results'][split]=audit_split(source,model,frames[split],split,classes,label_column,
                                            out,args,inventory,device)
        (out/'summary.json').write_text(json.dumps(result,indent=2))
    notes='''# Model audit\n\nOpen the selected split folder for PNG contact sheets. Each shows the original\nimage, two global inputs, and nine local inputs with per-view predictions.\nTensor panels undo normalization for display; tensor_stats.csv reports actual\nnormalized inputs. Optional augmented panels show one seeded training augmentation\nwith the model in evaluation mode; they are excluded from metrics.\n\n- input_inventory.csv: every saved split image, readability, dimensions, colors, hashes.\n- tree_overlap.csv: known trees appearing across saved splits.\n- duplicate_images.csv: identical paths, bytes, or decoded pixels; not near-duplicate/crop detection.\n- label_format_counts.csv: image formats by label and split.\n- Per split: predictions, confident errors, confusion matrix, per-tree and per-view metrics.\n- Per-view scores are diagnostics from ONE trained model, not separately trained ablations.\n- Tree macro-F1 uses all task classes, even when one is absent; inspect supports too.\n- Missing images make subset metrics incomplete, not directly comparable to original results.\n- Scores can differ slightly from batched inference due to numerical differences.\n- Use the exact training source/settings: checkpoints do not record every preprocessing choice.\n- No inference is performed on test unless explicitly requested. Existing test metrics\n  should not guide hyperparameter selection.\n- This audit cannot verify unseen source-image/leaf/tree identities across datasets.\n'''
    (out/'README.md').write_text(notes)
    print(json.dumps(result,indent=2))
    print(f'Audit saved to {out}')
    if result['unreadable_images']:
        raise SystemExit('Audit found unreadable images; see input_inventory.csv. Partial metrics are marked incomplete.')


if __name__=='__main__':
    main()
