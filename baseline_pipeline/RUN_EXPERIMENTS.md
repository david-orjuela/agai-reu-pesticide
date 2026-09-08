# Running the experiments for Dr. Chen

Place the updated train.py and new experiment_protocol.py in your existing baseline_pipeline directory. Keep your current baseline_model.py and dataset.py there. The code uses the frozen_dinov3 factory already added to baseline_model.py. This package does not replace your data loader, metadata or checkpoint weights.

Run commands from /home/davidorjuela/dev/agai-reu-pesticide. No new analysis dependency is required beyond the existing pandas, NumPy, SciPy, scikit-learn, matplotlib, PyTorch and torchvision environment. Keep the DINO import dependencies already installed.

## 1. Get TTA results from the existing checkpoints first

This performs inference only. It loads each original fold's saved model and val_indices, verifies targets and replays the single-view predictions before accepting a TTA comparison. The default checkpoint directory is checkpoints. Specify a different source directory if needed. Do not change the dataset order or preprocessing to make a failed replay pass.

```bash
uv run python -m baseline_pipeline.train --task bin_classification --backbone dinov3 --evaluate-only --checkpoint-dir checkpoints --preprocessing dinov3_256 --inference-modes single d4 scale --tta-scales 1.0 1.25 --seeds 42 --run-name dino_existing_class_tta

uv run python -m baseline_pipeline.train --task icp_regression --backbone dinov3 --evaluate-only --checkpoint-dir checkpoints --preprocessing dinov3_256 --inference-modes single d4 scale --tta-scales 1.0 1.25 --seeds 42 --run-name dino_existing_reg_tta
```

D4 uses the eight unique quarter-turn rotation/reflection orientations. Scale uses two source-image resizings, 256 and 320 pixels for DINO. These conditions are separate to distinguish orientation from resolution effects; d4_scale is an optional combined 16-view condition. Test images receive the same transform set regardless of their class.

Single-view replay is checked against checkpoint predictions with numerical tolerance. Older checkpoints have no CSV hash or sample IDs, so target and prediction matching is the available compatibility check, not a complete reconstruction of their original provenance. Existing checkpoint files include NumPy objects and are loaded as trusted local files with weights_only=False.

Evaluation-only output records the checkpoint's original selection method in its OOF rows. Reusing the old models retains their outer-fold epoch-selection limitation. TTA is not nested retraining and does not repair that limitation.

## 2. Establish the new matched baseline

The new default is inner_refit: select an epoch using a split inside the outer-training set, then reinitialize and refit on the full outer-training set for that epoch count. The outer test is evaluated after refitting. The original four outer splits are reproduced for the same sample ordering and seed when grouping is not enabled.

```bash
uv run python -m baseline_pipeline.train --task bin_classification --backbone dinov3 --preprocessing dinov3_256 --augmentation hflip --selection inner_refit --inference-modes single d4 --seeds 42 --run-name dino_class_hflip_inner

uv run python -m baseline_pipeline.train --task bin_classification --backbone resnet50 --preprocessing dinov3_256 --augmentation hflip --selection inner_refit --inference-modes single d4 --seeds 42 --run-name resnet_class_hflip_inner

uv run python -m baseline_pipeline.train --task icp_regression --backbone dinov3 --preprocessing dinov3_256 --augmentation hflip --selection inner_refit --inference-modes single d4 --seeds 42 --run-name dino_reg_hflip_inner

uv run python -m baseline_pipeline.train --task icp_regression --backbone resnet50 --preprocessing dinov3_256 --augmentation hflip --selection inner_refit --inference-modes single d4 --seeds 42 --run-name resnet_reg_hflip_inner
```

Both backbones deliberately use the same geometry and normalization here. ResNet BatchNorm statistics are now frozen by default. Older ResNet results can differ for that reason as well as preprocessing and selection changes. --legacy-resnet-bn and --selection legacy_outer are for explicit historical replication, not the recommended paper evaluation.

## 3. Change training augmentation only

```bash
uv run python -m baseline_pipeline.train --task bin_classification --backbone dinov3 --preprocessing dinov3_256 --augmentation d4 --selection inner_refit --inference-modes single d4 --seeds 42 --run-name dino_class_d4_inner
```

Compare with dino_class_hflip_inner. The single-view rows isolate training augmentation; D4 inference rows assess the additional TTA effect. medium_d4 is a separate optional training condition: all leaves retain random horizontal flips, and medium leaves also receive random quarter-turn rotations. It does not oversample medium or increase its independent sample count. Do not use it as the default just because the class is difficult.

## 4. A bounded threshold-sensitivity comparison

These two runs change the definition of the classification task. Both fit quantiles using only outer-training targets; all outer splits remain stratified on the original CSV bins so sample assignments are matched. The outer-training boundaries define the task for that fold's inner validation and final refit. This uses inner-label distribution in the task definition, but never the outer-test labels to fit boundaries.

```bash
uv run python -m baseline_pipeline.train --task bin_classification --backbone dinov3 --preprocessing dinov3_256 --bin-policy train_quantile --quantiles 0.3333333333333333 0.6666666666666666 --augmentation hflip --selection inner_refit --inference-modes single --seeds 42 --run-name dino_train_tertiles

uv run python -m baseline_pipeline.train --task bin_classification --backbone dinov3 --preprocessing dinov3_256 --bin-policy train_quantile --quantiles 0.30 0.70 --augmentation hflip --selection inner_refit --inference-modes single --seeds 42 --run-name dino_train_30_70
```

These quantiles are a prespecified sensitivity example, not an empirically optimal choice. Report thresholds.csv and class_distributions.csv with every score. Changed class ranges require training again; simply replacing labels in a classification CSV is not a valid new experiment. For chemistry-defined fixed limits, use --bin-policy fixed --thresholds LOWER UPPER, supplying the actual agreed numerical values. Ties are assigned to the lower class.

CSV bins remain the default for direct comparisons to the existing definition. Because original CSV boundaries were calculated on the full dataset, this estimates performance on those fixed exploratory categories, not a future training-only thresholding procedure. The code does not optimize boundaries on OOF performance.

## 5. Grouping and repeated seeds

If the team confirms a true independent group variable, add an explicit column to the metadata and pass --group-column confirmed_group_id on every matched run. Do not assume the ID prefix is the grouping key. Grouped outer and inner splitting enforce no group overlap. They may produce uneven class counts; inspect the exported distributions. Threshold variants still stratify using original CSV bins.

After retaining a small set of conditions, replace --seeds 42 with --seeds 42 7 21 and use a new run name. The script saves pooled metrics separately for each seed and a mean/SD across seeds. It does not treat 627 repeated predictions as 627 independent leaves. --selection fixed uses a predetermined epoch count without an inner selection stage; decide that count before inspecting the new outer results.

## Output files

Every run writes to results/experiments/RUN_NAME. An existing run-name directory causes an error to protect earlier results; choose a new name for reruns.

| File | Purpose |
|---|---|
| manifest.json | Settings, dataset hash, sample order, resolved preprocessing and DINO repository commit |
| dataset_manifest.csv | Metadata used by the image loader |
| splits.csv | Every outer/inner membership, sample ID and configured group |
| thresholds.csv | Fold-specific thresholds when estimated or explicitly specified; CSV policy retains source labels |
| class_distributions.csv | Counts and residue summaries by fold, role and class |
| history_seed_*_fold_*.csv | Inner-selection and refit learning histories, or explicit fixed/legacy histories |
| oof_predictions.csv | One row per seed/fold/image/inference condition, including continuous target and class probabilities/residuals |
| view_predictions.csv | Individual TTA view probabilities or scalar predictions for instability analysis |
| fold_metrics.csv | Metrics for each held-out fold and inference condition |
| seed_metrics.csv | Primary pooled OOF metrics per seed and inference condition |
| seed_stability.csv | Mean/SD across seed-level metrics when more than one seed runs |
| tta_deltas.csv | Each TTA condition minus the same model's single-view score |
| *_confusion.csv and *.png | Confusion counts/row percentages, or regression scatter/residual plots |
| *.pt | Final models fitted on each outer-training set; none written in evaluation-only mode |

For probability-averaged classification, prediction is the argmax of mean probabilities. For voting rows, prediction is the vote outcome, while probability_* columns still contain mean probabilities. Thus voting accuracy/F1 may differ while voting and averaging log loss/Brier are identical by construction; those probability scores describe the shared mean-probability forecast. Do not require voting labels to equal probability argmax.

view_instability means the fraction of TTA views disagreeing with the modal vote for classification, and the population SD across views for regression. It is sensitivity to the chosen transformations, not a confidence interval or calibrated probability of error. Raw per-view predictions are available for further paired diagnostics. No sample is evaluated using fold models that trained on it.
