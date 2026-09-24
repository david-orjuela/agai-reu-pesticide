# Technical analysis — ICP-OES copper residue prediction

## Scope and defensible conclusions

**209 unique sample IDs, 209 unique image paths, 21 trees.** Twenty sprayed trees have 10 samples each; one control tree has nine (`c 2` through `c 10`). `c 1` is absent; no exclusion log explains why. Unique paths do not prove unique biological leaves or absence of duplicate pixels because source images are not included.

All three uploaded runs are **bin classification** with ResNet50: CE, CORAL and CE+RPS. Each has 627 OOF rows = 209 samples × three seeds (42, 123, 2026), one inference mode, four outer folds. There are 33 files in total: ten CSVs and one manifest per objective. No regression run, treatment-only run, histories, search trajectories, code, raw image, or model checkpoint is in this archive. DINO paths and regression-selection fields in configuration are unused settings, not evidence of DINO or regression experiments.

The strongest supported conclusion is that these image classifiers have modest held-out-tree performance, no demonstrated macro-F1 winner among objectives, and no demonstrated advantage over the newly computed treatment-only baseline. Regression performance of the image model and incremental benefit of adding images to treatment remain unanswered.

## Validity and comparability audit

| Check | Result and limitation |
|---|---|
| Completeness | Exactly one prediction per sample per seed and `single` inference in every run; no missing target, probability or prediction. |
| Sample identity | All dataset manifests match exactly; all manifest CSV hashes match. Dataset indices agree with sample IDs in saved splits. |
| Target pairing | Same sample, seed, fold, continuous residue and class target across all three objectives. |
| Tree identity | Independently parsed `X ppm Y Z` as `Xppm_tree_Y`; all `c X` as `control_tree`. Every saved tree/group matches. |
| Outer disjointness | Zero shared trees between saved outer training and test sets in all 12 seed/fold combinations per method. Every tree occurs in one outer test fold per seed. |
| Inner disjointness | Zero shared trees between inner training and validation; their union equals outer training. Thus neither contains an outer-test sample. |
| Paired splits | Entire saved split tables are identical across objectives, including inner roles. Splits change across seeds, so seed differences combine split allocation and training randomness. |
| Split sizes | Outer test: 49–60 samples / 5–6 trees; outer train: 149–160 / 15–16 trees. Inner train: 110–120 / 11–12 trees; inner validation: 29–40 / 3–4 trees. |
| Inner evaluation | Records show **one inner training/validation partition per outer fold**. `inner_group_folds=4` does not establish that four inner folds were all evaluated. No inner-fold-indexed search log exists. |
| Probability integrity | Finite nonnegative probabilities sum to one within numerical tolerance; saved predictions equal probability argmax. |
| Recalculated metrics | Macro-F1, accuracy and ordinal MAE match supplied seed metrics exactly; log loss and RPS match within about 10⁻⁸. |
| Leakage limits | Recorded memberships pass. Without source code, images, labeling script and histories, duplicate-image leakage, threshold provenance, exact weight freezing and actual selection execution cannot be independently verified. |

**Common recorded configuration:** `bin_policy=csv`, target `residue_bin`, continuous field `mg_cm2`; `resolved_preprocessing=legacy_224`; horizontal-flip training augmentation; `inner_refit`; macro-F1 selection; 30 epochs; learning rates 0.0001, 0.001, 0.01; momentum 0.9; batch size 4; frozen ResNet batch-normalization flag; float32/CUDA; single-view inference with view count one. TTA scale fields exist but no TTA predictions were produced. Image paths identify `largest_leaf_crop`, but no image or transformation code is supplied to verify crop content or exact resize/normalization. The frozen-backbone description comes from the experimental context; the manifest explicitly documents frozen BN but does not enumerate trainable weights.

**Tuning budget:** Three learning-rate candidates and the same 30-epoch cap are declared for every objective. CE+RPS has a fixed weight of 1.0; no weight sweep is recorded. These are comparable declared budgets, but per-candidate histories are absent, so successful completion of every candidate and the selection trajectory cannot be audited. Different selected learning rates and epochs are expected results of separate tuning, not themselves a mismatch.

| seed | fold | CE LR | CE epoch | CE+RPS LR | CE+RPS epoch | CORAL LR | CORAL epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42.0000 | 1.0000 | 0.0010 | 24.0000 | 0.0100 | 7.0000 | 0.0010 | 25.0000 |
| 42.0000 | 2.0000 | 0.0100 | 7.0000 | 0.0100 | 7.0000 | 0.0100 | 8.0000 |
| 42.0000 | 3.0000 | 0.0100 | 30.0000 | 0.0100 | 30.0000 | 0.0010 | 25.0000 |
| 42.0000 | 4.0000 | 0.0100 | 14.0000 | 0.0010 | 20.0000 | 0.0100 | 4.0000 |
| 123.0000 | 1.0000 | 0.0100 | 28.0000 | 0.0010 | 21.0000 | 0.0100 | 20.0000 |
| 123.0000 | 2.0000 | 0.0100 | 19.0000 | 0.0100 | 19.0000 | 0.0100 | 21.0000 |
| 123.0000 | 3.0000 | 0.0001 | 26.0000 | 0.0001 | 8.0000 | 0.0010 | 19.0000 |
| 123.0000 | 4.0000 | 0.0010 | 28.0000 | 0.0100 | 25.0000 | 0.0100 | 6.0000 |
| 2026.0000 | 1.0000 | 0.0100 | 16.0000 | 0.0100 | 16.0000 | 0.0010 | 20.0000 |
| 2026.0000 | 2.0000 | 0.0100 | 7.0000 | 0.0100 | 7.0000 | 0.0010 | 27.0000 |
| 2026.0000 | 3.0000 | 0.0010 | 28.0000 | 0.0010 | 28.0000 | 0.0100 | 16.0000 |
| 2026.0000 | 4.0000 | 0.0100 | 29.0000 | 0.0100 | 29.0000 | 0.0100 | 4.0000 |

Several CE/CE+RPS selections are near the 30-epoch cap; this is a reason to inspect learning curves, **not proof of underfitting**. CORAL selected epochs 4–27. There is no basis here to compare training accuracy with inner-validation accuracy.

## Targets, cutpoints and chemistry validity

The labels were read from CSV, **not calculated as training-fold quantiles by these runs**. Every row of `thresholds.csv` says `csv` with missing lower and upper values; the quantile configuration is not evidence that quantiles were applied.

Observed ranges are shown below solely to describe the labels already present. They are **not reconstructed or proposed decision thresholds**; no outer-test outcomes were used to assign new labels.

| residue_bin | count | min | max |
| --- | --- | --- | --- |
| low | 70 | 0.010516 | 2.565247 |
| medium | 69 | 2.574977 | 3.635233 |
| high | 70 | 3.636463 | 13.219182 |

Exact cutpoints, boundary inclusion rules, their derivation dataset and whether they were fixed before evaluation are unknown. If these are full-dataset empirical tertiles, test outcomes influenced the target definition: all objectives still share those labels, but the evaluation does not cleanly establish performance for independently defined residue categories. Confirm provenance before publication. For a new quantile protocol, learn thresholds on inner training for selection and outer training for refit/test; record fold thresholds and recognize that categories then vary across folds. Chemically meaningful prespecified fixed thresholds offer a stable scientific target when available.

`mg_cm2` equals `total_mg / area_cm2` numerically for all 209 rows. That verifies arithmetic only: original ICP concentration units, dilution recovery, extraction volume, copper mass and projected-area calibration are not in this archive. Do not equate nominal spray ppm with measured copper density or formulated Kocide mass. Residue magnitudes and outliers require chemistry confirmation before quantitative application claims.

## Measured variation within treatments and trees

All statistics in this section use each sample exactly once; seeds are irrelevant to measurement summaries. Units: mg Cu/cm² as recorded. SD describes leaf variation, not uncertainty in the treatment mean.

| treatment | count | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| c | 9 | 0.1168 | 0.1675 | 0.0367 | 0.0242 | 0.5160 |
| 400 | 50 | 2.3858 | 0.9866 | 2.2249 | 0.6122 | 6.1506 |
| 600 | 50 | 3.3693 | 1.2893 | 3.0736 | 1.5995 | 7.3647 |
| 800 | 50 | 3.3548 | 1.1635 | 3.4090 | 0.0105 | 7.2284 |
| 1000 | 50 | 4.5703 | 1.8105 | 4.0772 | 2.5758 | 13.2192 |

The 600 and 800 ppm means are 3.369 and 3.355; all sprayed-treatment ranges overlap. Within-treatment deviations account for 62.5% of total squared variation about the overall mean; within-tree deviations account for 47.9%. These are **descriptive decompositions on the measured data**, not held-out R² or a causal estimate. Treatment has information, but cannot specify each leaf's measured value. Images could potentially explain residual variation through visible deposition or related leaf properties; that possibility is not demonstrated by the current comparison.

Every tree is shown below to avoid hiding heterogeneity inside treatment means.

| tree | count | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| 1000ppm_tree_1 | 10 | 5.8021 | 2.9017 | 5.4923 | 2.7436 | 13.2192 |
| 1000ppm_tree_2 | 10 | 3.8194 | 1.8563 | 3.1867 | 2.5758 | 8.9204 |
| 1000ppm_tree_3 | 10 | 4.1659 | 0.9708 | 3.7831 | 3.3068 | 6.4054 |
| 1000ppm_tree_4 | 10 | 5.1532 | 1.0874 | 4.8589 | 3.5705 | 7.1896 |
| 1000ppm_tree_5 | 10 | 3.9108 | 0.7074 | 3.7391 | 3.0598 | 4.8964 |
| 400ppm_tree_1 | 10 | 1.9555 | 0.5329 | 2.1185 | 1.1567 | 2.6619 |
| 400ppm_tree_2 | 10 | 2.1636 | 0.4370 | 2.0625 | 1.5374 | 3.0927 |
| 400ppm_tree_3 | 10 | 1.7161 | 0.7772 | 1.6562 | 0.6122 | 3.1078 |
| 400ppm_tree_4 | 10 | 2.6875 | 0.5375 | 2.5829 | 1.9182 | 3.7664 |
| 400ppm_tree_5 | 10 | 3.4060 | 1.3838 | 2.8340 | 1.9633 | 6.1506 |
| 600ppm_tree_1 | 10 | 3.0772 | 0.6164 | 3.0877 | 2.3092 | 4.2946 |
| 600ppm_tree_2 | 10 | 4.8108 | 1.3419 | 4.9917 | 2.4258 | 7.3647 |
| 600ppm_tree_3 | 10 | 2.9159 | 0.4846 | 2.9858 | 1.9135 | 3.4645 |
| 600ppm_tree_4 | 10 | 3.6041 | 1.5854 | 2.9900 | 1.5995 | 6.7465 |
| 600ppm_tree_5 | 10 | 2.4383 | 0.6816 | 2.2333 | 1.7455 | 3.5457 |
| 800ppm_tree_1 | 10 | 3.4577 | 0.4753 | 3.4934 | 2.4339 | 4.2047 |
| 800ppm_tree_2 | 10 | 3.2170 | 1.3826 | 3.9169 | 0.0105 | 4.5198 |
| 800ppm_tree_3 | 10 | 3.6685 | 0.8078 | 3.5087 | 2.5652 | 4.9602 |
| 800ppm_tree_4 | 10 | 3.2687 | 1.6819 | 2.7186 | 1.6019 | 7.2284 |
| 800ppm_tree_5 | 10 | 3.1620 | 1.2489 | 3.3770 | 0.7600 | 4.6251 |
| control_tree | 9 | 0.1168 | 0.1675 | 0.0367 | 0.0242 | 0.5160 |

Review these observations without automatically deleting them:

- `1000 ppm 1 7`: 13.219182, from recorded mass 16.20 mg and area 1.225492 cm². The small denominator contributes to the high density. Its validity requires inspecting area calibration, leaf identity and mass conversion.
- `1000 ppm 2 8`: 8.920432, area 1.478628 cm²; another small-area high-density sample.
- `800 ppm 2 4`: 0.010516, below every observed control; verify sample identity, extraction and label linkage.
- Controls: mean 0.116800, median 0.036669, range 0.024247–0.515981. They are not uniformly zero. Nine leaves from one tree do not establish variability across control trees.
- The observed non-monotonic treatment means and broad ranges are observations. Their causes—deposition variability, sampling, area measurement, extraction or metadata errors—cannot be separated with these outputs.

## Classification performance and uncertainty

| Method | Macro-F1 | Accuracy | Medium precision | Medium recall | Predicted medium fraction | Ordinal MAE (bins) | Low↔high fraction | Log loss | RPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CE | 0.453 | 0.455 | 0.332 | 0.353 | 0.351 | 0.643 | 0.097 | 1.400 | 0.222 |
| CE+RPS | 0.459 | 0.456 | 0.325 | 0.367 | 0.370 | 0.630 | 0.086 | 1.454 | 0.223 |
| CORAL | 0.484 | 0.496 | 0.370 | 0.280 | 0.258 | 0.604 | 0.100 | 1.080 | 0.197 |
| Treatment only (computed here) | 0.511 | 0.518 | 0.357 | 0.319 | 0.287 | 0.557 | 0.075 | 1.264 | 0.191 |

Metrics are computed separately for each seed over all OOF leaves, then averaged. Macro-F1 is the mean of low, medium and high F1. Ordinal MAE uses ordered labels 0, 1, 2 and is measured in **class steps, not mg Cu/cm²**. Extreme errors are low→high or high→low. Log loss uses natural logarithms; RPS is mean squared CDF error over the two boundaries. Lower is better for the last four metrics.

The treatment-only baseline is newly calculated for this report, not an uploaded run. It uses empirical class frequencies among outer-training leaves at each treatment. Any tie in predicted probability is resolved by the first class in low/medium/high order. No test-frequency fitting or smoothing tuning was performed.

The raw frequency baseline assigns zero true-class probability to five seed/sample evaluations; its exact unsmoothed mathematical log loss is infinite for seed 123. The displayed finite value 1.264 uses an explicitly disclosed numerical floor of 10⁻¹⁵, applied consistently to all models. Do not interpret that number as evidence of calibrated probabilities. A separately labeled, prespecified smoothed baseline should be used for a future probability-quality comparison. Classification F1 and RPS are unaffected by this numerical log-loss floor.

All seeds have low/medium/high support 70/69/70. Outer test supports range 14–22 low, 14–22 medium and 16–18 high. Poor medium performance is therefore not explained by a gross overall class-count imbalance.

| Method | precision_low | recall_low | f1_low | precision_medium | recall_medium | f1_medium | precision_high | recall_high | f1_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CE | 0.431 | 0.414 | 0.420 | 0.332 | 0.353 | 0.342 | 0.607 | 0.595 | 0.599 |
| CE+RPS | 0.439 | 0.429 | 0.428 | 0.325 | 0.367 | 0.344 | 0.643 | 0.571 | 0.604 |
| CORAL | 0.459 | 0.500 | 0.478 | 0.370 | 0.280 | 0.313 | 0.632 | 0.705 | 0.661 |
| Treatment only (computed here) | 0.684 | 0.605 | 0.637 | 0.357 | 0.319 | 0.332 | 0.511 | 0.629 | 0.563 |

**Does medium recall improve simply by predicting medium more often?** CE predicts medium on 35.1% of leaves, CE+RPS on 37.0%, and CORAL on 25.8%. CE+RPS increases mean recall only 1.45 percentage points, while precision drops from 33.2% to 32.5%. In seed 42 it predicts medium 90 times versus CE's 77, with 32 versus 25 correct medium predictions. The other two seeds have lower medium recall than CE. This is not a robust medium-class solution. CORAL is more selective about medium and improves average precision to 37.0%, but reduces recall to 28.0% and medium F1 to 0.313.

Seed-specific paired results:

| model | seed | macro_f1 | precision_medium | recall_medium | predicted_medium |
| --- | --- | --- | --- | --- | --- |
| CE | 42 | 0.443 | 0.325 | 0.362 | 0.368 |
| CE | 123 | 0.462 | 0.333 | 0.362 | 0.359 |
| CE | 2026 | 0.455 | 0.338 | 0.333 | 0.325 |
| CE+RPS | 42 | 0.474 | 0.356 | 0.464 | 0.431 |
| CE+RPS | 123 | 0.461 | 0.303 | 0.333 | 0.364 |
| CE+RPS | 2026 | 0.441 | 0.318 | 0.304 | 0.316 |
| CORAL | 42 | 0.483 | 0.381 | 0.232 | 0.201 |
| CORAL | 123 | 0.460 | 0.312 | 0.348 | 0.368 |
| CORAL | 2026 | 0.509 | 0.419 | 0.261 | 0.206 |
| Treatment only (computed here) | 42 | 0.504 | 0.386 | 0.391 | 0.335 |
| Treatment only (computed here) | 123 | 0.480 | 0.300 | 0.174 | 0.191 |
| Treatment only (computed here) | 2026 | 0.548 | 0.386 | 0.391 | 0.335 |

All 12 paired fold macro-F1 comparisons (folds overlap in their training data and are not independent replicates):

| seed | fold | ce | ce_rps | coral | treatment_only | CORAL − CE | CE+RPS − CE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42.000 | 1.000 | 0.522 | 0.584 | 0.506 | 0.348 | -0.016 | 0.062 |
| 42.000 | 2.000 | 0.333 | 0.350 | 0.353 | 0.461 | 0.020 | 0.017 |
| 42.000 | 3.000 | 0.423 | 0.423 | 0.442 | 0.590 | 0.020 | 0.000 |
| 42.000 | 4.000 | 0.399 | 0.480 | 0.532 | 0.510 | 0.133 | 0.082 |
| 123.000 | 1.000 | 0.565 | 0.508 | 0.348 | 0.622 | -0.217 | -0.056 |
| 123.000 | 2.000 | 0.337 | 0.372 | 0.481 | 0.469 | 0.144 | 0.035 |
| 123.000 | 3.000 | 0.433 | 0.414 | 0.408 | 0.444 | -0.025 | -0.019 |
| 123.000 | 4.000 | 0.499 | 0.517 | 0.541 | 0.345 | 0.042 | 0.018 |
| 2026.000 | 1.000 | 0.418 | 0.374 | 0.444 | 0.637 | 0.026 | -0.044 |
| 2026.000 | 2.000 | 0.340 | 0.327 | 0.441 | 0.590 | 0.101 | -0.013 |
| 2026.000 | 3.000 | 0.482 | 0.478 | 0.539 | 0.503 | 0.057 | -0.004 |
| 2026.000 | 4.000 | 0.537 | 0.521 | 0.540 | 0.430 | 0.003 | -0.016 |

CORAL exceeds CE in 9/12 fold comparisons but only 2/3 seed-level comparisons; it loses substantially in seed 123 fold 1. CE+RPS exceeds CE in 5/12 folds, ties once, and exceeds it in only 1/3 seed-level comparisons. Averaging fold F1 is not the reported primary estimate; the primary estimate is average seed-specific pooled-OOF F1.

Uncertainty: 5,000 nonparametric bootstrap draws of **21 whole trees**, with replacement. Each draw uses the same tree multiplicities for all models and seeds, computes each seed metric, then averages metrics across seeds. No seed-averaged prediction ensemble is created. Percentile 95% intervals are conditional on these fitted models and splits, do not include retraining uncertainty, and are exploratory with only 21 trees, five trees per treatment and one control tree. Bootstrap trees are sampled across the full set, not stratified by treatment; rare-class supports and the lone control deserve caution. They do not justify confirmatory significance claims after many comparisons.

| comparison | estimate | lower | upper |
| --- | --- | --- | --- |
| ce | 0.453 | 0.388 | 0.506 |
| ce_rps | 0.459 | 0.403 | 0.505 |
| coral | 0.484 | 0.402 | 0.550 |
| treatment_only | 0.511 | 0.411 | 0.583 |
| coral minus ce | 0.030 | -0.034 | 0.092 |
| ce_rps minus ce | 0.005 | -0.021 | 0.032 |
| coral minus ce_rps | 0.025 | -0.043 | 0.085 |
| ce minus treatment_only | -0.057 | -0.161 | 0.054 |
| coral minus treatment_only | -0.027 | -0.154 | 0.106 |
| ce_rps minus treatment_only | -0.052 | -0.151 | 0.059 |

CORAL's secondary RPS improvement over CE is −0.0250, interval −0.0452 to −0.0035, and its log loss is lower. This supports a possible probability-quality advantage conditional on these runs, but does **not** establish a primary macro-F1 win. CORAL's extreme-error rate is 10.0%, slightly higher than CE's 9.7%; ordinal loss did not uniformly reduce the most serious class mistakes. CE+RPS has lower extreme-error rate (8.6%) but slightly worse mean RPS than CE, so optimizing an auxiliary term did not guarantee better held-out RPS.

### Confusion matrices

Rows = actual class; columns = predicted low/medium/high. Each matrix is for **one seed and 209 samples**. Matrices are not pooled and described as independent samples.

**CE, seed 42**

| Actual | Predicted low | Predicted medium | Predicted high |
| --- | --- | --- | --- |
| Actual low (70) | 24 | 33 | 13 |
| Actual medium (69) | 28 | 25 | 16 |
| Actual high (70) | 7 | 19 | 44 |

**CE, seed 123**

| Actual | Predicted low | Predicted medium | Predicted high |
| --- | --- | --- | --- |
| Actual low (70) | 26 | 31 | 13 |
| Actual medium (69) | 28 | 25 | 16 |
| Actual high (70) | 5 | 19 | 46 |

**CE, seed 2026**

| Actual | Predicted low | Predicted medium | Predicted high |
| --- | --- | --- | --- |
| Actual low (70) | 37 | 25 | 8 |
| Actual medium (69) | 31 | 23 | 15 |
| Actual high (70) | 15 | 20 | 35 |

**CE+RPS, seed 42**

| Actual | Predicted low | Predicted medium | Predicted high |
| --- | --- | --- | --- |
| Actual low (70) | 24 | 35 | 11 |
| Actual medium (69) | 22 | 32 | 15 |
| Actual high (70) | 4 | 23 | 43 |

**CE+RPS, seed 123**

| Actual | Predicted low | Predicted medium | Predicted high |
| --- | --- | --- | --- |
| Actual low (70) | 31 | 32 | 7 |
| Actual medium (69) | 35 | 23 | 11 |
| Actual high (70) | 8 | 21 | 41 |

**CE+RPS, seed 2026**

| Actual | Predicted low | Predicted medium | Predicted high |
| --- | --- | --- | --- |
| Actual low (70) | 35 | 26 | 9 |
| Actual medium (69) | 34 | 21 | 14 |
| Actual high (70) | 15 | 19 | 36 |

**CORAL, seed 42**

| Actual | Predicted low | Predicted medium | Predicted high |
| --- | --- | --- | --- |
| Actual low (70) | 35 | 18 | 17 |
| Actual medium (69) | 30 | 16 | 23 |
| Actual high (70) | 7 | 8 | 55 |

**CORAL, seed 123**

| Actual | Predicted low | Predicted medium | Predicted high |
| --- | --- | --- | --- |
| Actual low (70) | 29 | 33 | 8 |
| Actual medium (69) | 34 | 24 | 11 |
| Actual high (70) | 8 | 20 | 42 |

**CORAL, seed 2026**

| Actual | Predicted low | Predicted medium | Predicted high |
| --- | --- | --- | --- |
| Actual low (70) | 41 | 18 | 11 |
| Actual medium (69) | 32 | 18 | 19 |
| Actual high (70) | 12 | 7 | 51 |

**Treatment only (computed here), seed 42**

| Actual | Predicted low | Predicted medium | Predicted high |
| --- | --- | --- | --- |
| Actual low (70) | 35 | 21 | 14 |
| Actual medium (69) | 10 | 27 | 32 |
| Actual high (70) | 5 | 22 | 43 |

**Treatment only (computed here), seed 123**

| Actual | Predicted low | Predicted medium | Predicted high |
| --- | --- | --- | --- |
| Actual low (70) | 48 | 12 | 10 |
| Actual medium (69) | 23 | 12 | 34 |
| Actual high (70) | 8 | 16 | 46 |

**Treatment only (computed here), seed 2026**

| Actual | Predicted low | Predicted medium | Predicted high |
| --- | --- | --- | --- |
| Actual low (70) | 44 | 21 | 5 |
| Actual medium (69) | 10 | 27 | 32 |
| Actual high (70) | 5 | 22 | 43 |

Low→high / high→low counts by seed 42, 123, 2026: CE **13/7, 13/5, 8/15**; CE+RPS **11/4, 7/8, 9/15**; CORAL **17/7, 8/8, 11/12**.

### Where classification errors concentrate

| Treatment | ce | ce_rps | coral | treatment_only |
| --- | --- | --- | --- | --- |
| 1000 | 0.527 | 0.520 | 0.500 | 0.360 |
| 400 | 0.573 | 0.580 | 0.520 | 0.300 |
| 600 | 0.620 | 0.593 | 0.560 | 0.600 |
| 800 | 0.493 | 0.487 | 0.467 | 0.693 |
| c | 0.370 | 0.519 | 0.333 | 0.333 |

Tree-specific failures are substantial: CORAL misclassifies 86.7% of evaluations on `600ppm_tree_3`, 83.3% on `600ppm_tree_1`, and 80.0% on `1000ppm_tree_3`, versus 10.0% on `600ppm_tree_2`. These percentages average the three evaluations of each leaf and do not add biological replication. CE's worst tree is `600ppm_tree_5` (86.7% errors); CE+RPS's is `600ppm_tree_3` (83.3%). The full per-tree, per-treatment and per-class error tables are supplied with the calculations. Error concentration can reflect tree heterogeneity, covariate shifts or weak signal; the outputs do not identify the cause.

Excluding controls as a descriptive sensitivity analysis, mean macro-F1 is CE 0.442, CE+RPS 0.456, CORAL 0.475, treatment-only 0.508. The lack of an observed image advantage is not explained solely by the control fallback.

## Continuous prediction: what exists and what does not

**No image-regression predictions exist in this upload.** A class probability or integer bin is not a calibrated copper-density prediction. No image MAE, RMSE, bias or R² can be reported, and it is impossible to say whether image regression beats a mean predictor. Negative image-regression R² from any earlier experiment must not be imported into this comparison.

I computed three no-image regressors on the exact saved outer splits: global training mean, global training median, and treatment-specific training mean. Each is fitted on outer-training leaves only, with all outer-test trees excluded. Every sprayed tree has ten leaves, so the treatment-specific leaf mean also equals an equally weighted mean of its available training-tree means.

**Unseen treatment/control handling:** The one control tree is entirely held out once per seed, leaving zero training controls in that fold. For any unseen treatment, the treatment-specific regressor falls back to the outer-training global mean; the classifier falls back to the global class distribution. It does not use the held-out control mean, assign zero by fiat, or borrow labels from another seed's fit. This fallback is deliberately simple and reproducible, but it cannot estimate a control-specific response. All sprayed concentrations remain represented in training.

Values below average seed-specific OOF metrics. Bias is predicted minus measured. MAE, RMSE and bias are in recorded mg Cu/cm²; R² is unitless.

| model | unit | MAE | RMSE | bias | R2 |
| --- | --- | --- | --- | --- | --- |
| global_mean | leaf | 1.1880 | 1.6553 | 0.0010 | -0.0041 |
| global_mean | tree | 0.8757 | 1.2139 | 0.0162 | -0.0084 |
| global_median | leaf | 1.1829 | 1.6591 | -0.1551 | -0.0088 |
| global_median | tree | 0.8952 | 1.2172 | -0.1399 | -0.0139 |
| treatment_mean | leaf | 1.0991 | 1.5312 | 0.1375 | 0.1406 |
| treatment_mean | tree | 0.7445 | 1.0393 | 0.1520 | 0.2606 |

The global baselines have negative held-out R²; treatment means improve average leaf errors modestly but still leave large residual errors. These are **baseline** results only. The treatment baseline's leaf MAE/RMSE on the 200 sprayed leaves alone are 1.005/1.411 (R² 0.163); global mean is 1.098/1.551 (R² −0.010). With the control tree held out, its nine leaves have treatment-baseline MAE 3.186 because no control training tree exists.

Treatment-mean baseline subgroup performance (not image-model performance):

| factor | label | n | MAE | RMSE | bias |
| --- | --- | --- | --- | --- | --- |
| range | (-inf, 1.0] | 13.0000 | 2.9172 | 2.9786 | 2.9172 |
| range | (1.0, 2.5] | 52.0000 | 0.8226 | 1.0011 | 0.7579 |
| range | (2.5, 4.0] | 90.0000 | 0.7044 | 0.8912 | 0.3518 |
| range | (4.0, 6.0] | 44.0000 | 1.1013 | 1.3041 | -0.9792 |
| range | (6.0, inf] | 10.0000 | 3.7166 | 4.1954 | -3.7166 |
| treatment | 1000 | 50.0000 | 1.3495 | 1.8804 | 0.0349 |
| treatment | 400 | 50.0000 | 0.7311 | 1.0539 | -0.0605 |
| treatment | 600 | 50.0000 | 1.0832 | 1.4063 | 0.0128 |
| treatment | 800 | 50.0000 | 0.8569 | 1.1590 | 0.0141 |
| treatment | c | 9.0000 | 3.1864 | 3.1903 | 3.1864 |

The residue ranges are descriptive error-analysis ranges, chosen for readability; they were not used to fit predictors, select models or redefine classes. Extremely negative within-range R² is not informative when target variance is restricted, so it is omitted from this subgroup table.

Treatment-mean absolute errors have mean-across-seed median 0.820, 90th percentile 2.410, 95th percentile 3.261 and maximum 8.869 mg Cu/cm². About 13.4% of evaluations exceed 2 and 1.44% exceed 4. These rates average repeated evaluations of the same 209 samples. Largest mean absolute errors: `1000 ppm 1 7` 8.869; `600 ppm 2 5` 4.363; `1000 ppm 2 8` 4.147; `400 ppm 5 10` 4.024; `800 ppm 4 2` 3.817. High-residue leaves are usually underpredicted by treatment means; near-zero samples are overpredicted. Individual baseline predictions and subgroup results are included for every seed.

## Leaf versus tree aggregation

For continuous baselines, average measured and predicted density **within each held-out tree and separately within each seed**, then evaluate 21 equally weighted trees. This changes the unit from 209 leaves to 21 tree means. It reduces treatment-baseline MAE/RMSE from 1.099/1.531 at leaf level to 0.745/1.039 at tree level; it does not establish an image-model aggregation benefit. Trees have 10 sampled leaves, except nine controls. A mean over these sampled leaves is not automatically a representative whole-canopy residue estimate.

Arithmetic mean leaf density is the requested endpoint here. If chemistry instead wants total copper divided by total sampled area, that is an **area-weighted density**, a different endpoint; define it before evaluation. Neither is a direct whole-tree total copper measurement.

For classification, an average probability vector estimates the class composition of sampled leaves. It does not estimate the bin of mean continuous residue. Because exact cutpoints and continuous image predictions are absent, I did **not** fabricate a tree-mean residue class.

Exploratory aggregation: within tree and seed, average predicted class probabilities. Compare that vector with the empirical fractions of low/medium/high measured leaves. All 21 trees contribute equally; probability-composition MAE is the mean absolute error over three fractions and CDF-composition MSE averages two cumulative-fraction errors. This is a different endpoint from leaf RPS. For an additional “most common leaf class” diagnostic, take argmax of the mean probability vector and compare with the actual modal leaf class; exclude the single tied tree (`1000ppm_tree_5`, five medium and five high leaves), leaving 20 trees. This is explicitly not average integer labels or a majority vote over hard model predictions.

| model | probability_composition_MAE | CDF_composition_MSE | modal_macro_f1 | modal_accuracy |
| --- | --- | --- | --- | --- |
| CE | 0.210 | 0.069 | 0.506 | 0.533 |
| CE+RPS | 0.213 | 0.069 | 0.541 | 0.550 |
| CORAL | 0.190 | 0.058 | 0.414 | 0.483 |
| Treatment only (computed here) | 0.186 | 0.061 | 0.584 | 0.600 |

The 20-tree modal target has support low 8, medium 6, high 6. These exploratory, small-sample results do not establish a winner or a benefit for predicting continuous tree residue. No probabilities or predictions were averaged across seeds for any metric.

## Underfitting, overfitting and model capacity

**Unanswered from this archive.** No training or inner-validation histories were supplied. OOF errors and selected epoch numbers cannot distinguish capacity limitations, weak visual signal, distribution shift, noise and overfitting. Old training-versus-validation accuracies from other splits do not diagnose these runs.

Request, for every objective, seed, outer fold and learning-rate candidate: epoch, training loss/F1, inner-validation loss/F1, selected epoch and learning rate, and trainable-parameter count. Compare predictive metrics across objectives; their raw training losses have different definitions. Evaluate training metrics under the same deterministic single-view preprocessing as validation when diagnosing the gap.

- Persistent high training performance but lower or declining inner-validation performance, especially after the validation optimum, supports overfitting within that split.
- Poor training and validation performance that plateau can support limited capacity or weak features, but label noise or lack of signal are alternatives. Both improving at the epoch cap supports trying longer optimization on training/inner-validation data only.
- A head can overfit even when the encoder is frozen: 209 samples are clustered in just 21 trees, and selection uses only 3–4 validation trees.
- Adding a small regularized nonlinear head is a reasonable controlled capacity experiment, not an established remedy. It changes the decision function on existing features; it cannot recover visual detail the frozen representation discarded. Partial backbone fine-tuning can adapt features but increases compute and flexibility. Neither approach is justified as “needed” before inspecting histories and baselines.

## Additional treatment-labeled images: rigorous feasibility experiment

### What the two stages can learn

Stage 1 predicts **nominal treatment**, using correctly linked extra images; stage 2 predicts **measured leaf residue** using only ICP-linked leaves. Treatment supervision may help learn visible spraying-associated features and adapt representations to this imaging domain. It supplies no new measured residue targets and does not reveal the actual within-treatment density of an unmeasured leaf. Do not label those leaves with a treatment-average ICP value or describe them as additional ICP measurements.

A crucial implementation detail: training only a disposable treatment head on a completely frozen encoder, then throwing away that head and initializing a new ICP head, leaves the encoder unchanged. That procedure transfers no newly learned encoder representation. To make stage 1 meaningful, update at least a controlled upper backbone block, or explicitly retain a learned adapter/projector into stage 2. Replace the treatment output head with the ICP head. Distinguish this representation adaptation from ordinary frozen-feature classification. The distinction between frozen feature extraction and fine-tuning is documented in the [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

Treatment prediction is a proxy task. A treatment classifier could recognize an acquisition batch, background, leaf pose, camera, visible tag or scale reference instead of copper-related leaf appearance. High treatment accuracy alone is not evidence that this pretraining improves residue estimation. Additional photos of the same leaf or tree increase image diversity but not independent biological sample size. If treatment is perfectly confounded with batch/background, this dataset alone may be unable to distinguish their effects; standardized new images or a separate batch are needed.

### Nested evaluation protocol

1. Create an authoritative mapping for **every image**, including unmeasured leaves: file → biological leaf → unique tree → treatment → acquisition batch. Use treatment-plus-tree number; add experiment/batch identity if tree numbering also restarts across experiments. Unknown tree identities are quarantined until resolved. Keep duplicates and views of the same leaf in its tree group.
2. Reuse the existing outer held-out trees. Exclude **all images from every outer-test tree** from treatment pretraining, ICP training, preprocessing fitting, early stopping and hyperparameter selection. This applies even to images that have no ICP label. Do not perform a one-time treatment pretraining on the entire image pool and reuse it across outer folds.
3. Inside each outer-training set, reuse the inner tree split. During selection, exclude all images from inner-validation trees from both stages' training. Select pretraining epochs/LR, unfreezing depth, ICP-head regularization and stage-2 epochs using training trees and inner validation only. Prefer final ICP validation performance as the selection criterion; treatment accuracy is a diagnostic. Start with a small declared budget. Any additional trees without ICP need a fixed, documented group allocation; never make their allocation depend on outer-test performance.
4. After selection, restart from the common pretrained initialization and refit the two-stage pipeline using all eligible outer-training trees, including the former inner-validation trees, with selected settings and epoch counts. Outer-test trees remain excluded from both stages. Save the stage-1 and stage-2 image lists and their tree IDs so exclusion is auditable.
5. Pair two otherwise identical image architectures, preprocessing, ICP splits and tuning budgets: **without treatment pretraining** versus **with treatment pretraining**. Keep the stage-2 trainability policy the same. If stage 1 requires an adapter, give the control arm the same adapter architecture. Distinguish extra optimization compute from extra supervision when interpreting any improvement.
6. For both arms, compare held-out ICP regression and prespecified bins against treatment-only, and fit an **image + treatment** model on the same trees. A practical combined model uses image features plus one-hot treatment, with a regularized head; a training-only residual model is another option. Select all hyperparameters in inner splits. Assess paired improvement in MAE/RMSE, macro-F1, tree means and within-treatment errors—not just treatment-classification accuracy. Report controls and unseen levels separately.
7. Expand from a one-seed pilot to the full paired three-seed evaluation only if the pipeline passes audits. A pilot is a feasibility result, not evidence for a winner. The current outer results are already visible and guide this proposal; repeated development against these same outer folds makes subsequent comparisons exploratory. A final untouched set of new trees or acquisition batch would strengthen a publication claim.

The requirement that data-dependent fitting use training data only follows the standard leakage principle in [scikit-learn's guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage); the two-stage tree-exclusion design above is the proposed application to this dataset.

**Controls:** A single control tree cannot be present in training and also support independent held-out control testing. If stage 1 uses control as a class, some folds have no training example for that class. Declare handling in advance; do not borrow the held-out controls. A first feasibility comparison can explicitly train the proxy task on the four sprayed treatments, while evaluating downstream ICP controls separately. It then makes no claim to learn a generalizable control class.

### When segmentation may wait, and shortcut checks

Full segmentation can wait for a first pilot if images can be consistently cropped to a single identified leaf, useful leaf detail remains at the chosen resolution, visible labels are excluded and backgrounds are not perfectly confounded with treatment. Use the same framing policy in both comparison arms. Existing ICP inputs are described by paths as largest-leaf crops; mixing them with wide-frame stage-1 images without documenting scale would be a domain mismatch. When a photo contains multiple leaves, the ICP-measured leaf must be identified before assigning its ICP target.

Start with a stratified visual audit across treatment, tree and batch, plus duplicate/perceptual-hash review. Record leaf occupancy, resolution, focus, exposure and occlusion. Use resize-with-padding or a controlled crop to avoid arbitrary aspect-ratio distortion. Compare full frame, rough leaf crop and background-only/leaf-occluded inputs on the same valid split. If background-only treatment prediction is strong, investigate acquisition confounding. Test masking stickers and scale markers, and evaluate changes in crop padding or background replacement where masks exist. These are diagnostic perturbations, not proof of causality; saliency pictures alone do not rule out shortcuts. If framing dominates, perform targeted segmentation or standardize acquisition before expensive pretraining. Avoid aggressive color augmentation without validating that it preserves the residue-related signal.

## Chemistry metadata questions, ordered by urgency

| Priority | Information needed | Why it matters / can it wait? |
|---|---|---|
| Essential before extra-image model development | Exact image count; distinct leaf/tree counts; every image's treatment and unique tree ID; numbering resets within treatment and across collection rounds | Prevents cross-stage tree leakage and inflated sample counts. Count alone is insufficient. |
| Essential before splitting | Control provenance; whether all current control images belong to the same physical tree; any additional control trees | Establishes valid independent control evaluation. |
| Essential before splitting/labeling | Duplicate views of the same leaf; biological leaf ID; which exact imaged leaf was measured by ICP | Prevents identity leakage and assigning another leaf's target. |
| Essential before interpreting proxy-task success | Acquisition date/batch, camera, lighting, backgrounds, framing; tags, stickers or visible treatment labels; whether file names are ever model inputs | Reveals treatment shortcuts and confounding. Filenames alone are not a pixel shortcut unless fed to the model or represented visibly. |
| Essential before use | Permission to use all images for model development; any exclusions, ownership or planned test set restrictions | Determines the eligible pool. |
| Essential for quantitative claims | ICP conversion and dilution/extraction volumes; area units/convention; blank handling; bin creation script/cutpoints; outlier identity checks | Validates mg Cu/cm² and target provenance. |
| Needed for cleaning and interpretation | Quality failures, corrupt/missing images, missing labels, multi-leaf scenes; reason for missing `c 1` | Prevents silent exclusions; image review can proceed while metadata is completed. |
| Needed before pooling batches | Spray application, drying time, collection timing, extraction/storage protocol, treatment balance by batch | Determines whether images describe comparable residue conditions and whether batch-aware tests are necessary. |
| Can follow the initial leakage audit | Detailed camera settings, mask refinements, exact deployment sampling count and acceptable error | Useful for optimization and application design; do not delay basic identity mapping for perfect segmentation. Define acceptable error before final claims. |

## Ranked next experiments and computational effort

These are **planning estimates, not measured runtimes**. Hardware, input size and the extra-image count are unknown. Baseline arithmetic has already been completed on CPU. Preserve the paired splits and declare new search budgets before training.

| Rank | Action | Status / estimated effort |
|---|---|---|
| 1 | Confirm units, outliers, CSV cutpoints, leaf/tree mapping; obtain omitted histories/regression OOF files | No GPU. Metadata/export review roughly 15–60 minutes if files are accessible. Do not rerun completed classification just to obtain its metrics. |
| 2 | Recalculate existing image-regression metrics if the run exists; compare against the supplied baselines and aggregate by tree | Seconds to minutes of CPU after files arrive; no retraining. If no run exists, train continuous image regression next. |
| 3 | Frozen-feature regression and image + treatment with simple regularized heads; training-only scaling/tuning | Cached features: seconds to minutes for heads; feature extraction minutes to an hour depending on I/O/GPU. An unchanged three-LR, three-seed, four-outer-fold neural schedule implies 36 inner candidate fits + 12 refits per model, potentially hours. Time one fold before scheduling the full run. |
| 4 | Inspect histories; test small regularized head versus linear head or limited backbone fine-tuning only if diagnostics warrant | Existing-curve analysis minutes. Cached nonlinear head pilot minutes to an hour; partial backbone training hours. Keep budgets matched; inspect inner validation, not outer results, to choose settings. |
| 5 | Treatment-pretraining feasibility with strict per-fold exclusions and crop/background checks | First index/review images (roughly hours of human/data work). One seed × four outer folds for each matched arm; stage-1 training may take hours to a day or more for thousands of images. This must be timed after image count and hardware are known. |
| 6 | Expand successful feasibility protocol to three seeds and a new-tree/new-batch test; refine masks only where needed | Approximately 3× the one-seed fit workload, plus repeated inner selection; full-backbone training can require days. New independent trees improve evidence more directly than many additional photos of existing trees. |

**Report today:** all recorded classification comparisons and audit results; residue variability; new no-image baselines; missing cutpoint/history/regression limitations. **Quick computation:** existing-regression analysis, history diagnostics, metadata checks, fixed smoothing sensitivity and alternative prespecified aggregation if required. **Retraining:** absent continuous models, image+treatment, representation pretraining, and a clean threshold protocol if CSV target creation used held-out outcomes.

## Reproducibility and evidence inventory

The companion calculations archive contains the original extracted output tables, `analyze.py`, `supplement.py`, the report-generation script, recomputed metric tables, 5,000-draw paired tree-bootstrap intervals, generated OOF baseline predictions, residue summaries and all subgroup diagnostics. `analyze.py` uses assertions for saved split disjointness, cross-objective pairing, completeness, target identity and probability consistency. It uses NumPy, pandas and scikit-learn. The bootstrap RNG seed is 2026. It does not retrain image models.

Numeric source files: `dataset_manifest.csv` for observed residues and image/sample mapping; `splits.csv` for training/test memberships; `oof_predictions.csv` for held-out probabilities and selected settings; each `manifest.json` for configuration; uploaded `seed_metrics.csv` for numerical cross-checks. No performance claim is inferred from folder names. All new calculations use the archive attached to this turn; no other similarly named archives or historical results are merged.
