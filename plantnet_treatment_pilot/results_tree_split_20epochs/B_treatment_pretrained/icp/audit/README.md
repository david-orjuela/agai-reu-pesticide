# Model audit

Open the selected split folder for PNG contact sheets. Each shows the original
image, two global inputs, and nine local inputs with per-view predictions.
Tensor panels undo normalization for display; tensor_stats.csv reports actual
normalized inputs. Optional augmented panels show one seeded training augmentation
with the model in evaluation mode; they are excluded from metrics.

- input_inventory.csv: every saved split image, readability, dimensions, colors, hashes.
- tree_overlap.csv: known trees appearing across saved splits.
- duplicate_images.csv: identical paths, bytes, or decoded pixels; not near-duplicate/crop detection.
- label_format_counts.csv: image formats by label and split.
- Per split: predictions, confident errors, confusion matrix, per-tree and per-view metrics.
- Per-view scores are diagnostics from ONE trained model, not separately trained ablations.
- Tree macro-F1 uses all task classes, even when one is absent; inspect supports too.
- Missing images make subset metrics incomplete, not directly comparable to original results.
- Scores can differ slightly from batched inference due to numerical differences.
- Use the exact training source/settings: checkpoints do not record every preprocessing choice.
- No inference is performed on test unless explicitly requested. Existing test metrics
  should not guide hyperparameter selection.
- This audit cannot verify unseen source-image/leaf/tree identities across datasets.
