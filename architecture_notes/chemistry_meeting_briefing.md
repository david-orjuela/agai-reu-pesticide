# Chemistry meeting briefing — 18 September 2026

Based exclusively on the attached `26-9-18_experiments-coral-ce-resnet50(3).rar`, plus new training-fold-only baseline calculations from its saved splits. All residue units below are the dataset's recorded **mg Cu/cm²**; the original ICP mass conversion is not independently verified here.

## Three-minute narrative

“We have 209 measured leaf samples from 21 trees: 50 leaves from five trees at each of four spray concentrations, plus nine control leaves from one control tree. The three classification experiments are complete. For each method, every sample has one held-out prediction per seed, and the saved records keep trees separate in both model selection and testing. The methods use the same splits, so their comparisons are paired.

“Spray concentration is associated with measured residue, but it does not determine what remains on an individual leaf. For example, the 600 and 800 ppm groups both average about 3.36 mg Cu/cm², despite their different spray concentrations. Even leaves from the same tree vary considerably. On one 1000 ppm tree, measured residue ranges from 2.74 to 13.22. That leaves a useful scientific question: can images explain variation within a treatment or within a tree? We have not yet established that they do.

“For low, medium and high classification, average macro-F1 is 0.454 for ordinary cross-entropy, 0.459 with the RPS term, and 0.484 for CORAL. CORAL has the highest average image-model score, but the uncertainty on its improvement includes no improvement. It also misses more medium leaves: medium recall is 28%, compared with 35% for CE and 37% for CE plus RPS. The small average recall increase with CE plus RPS comes with more medium predictions and slightly lower precision.

“I also calculated a treatment-only baseline using only the training trees in each existing split. Its macro-F1 is 0.511, higher than all three image models on average. This does not prove images contain no useful information, but today we cannot claim that these image models improve on knowing the spray treatment. An image-plus-treatment model is still needed to test incremental value directly.

“The archive has no image-regression results. A treatment-mean baseline gives a leaf-level MAE of 1.10 and RMSE of 1.53 mg Cu/cm²; these are reference results, not image-model performance. We also need to confirm where the existing low/medium/high cutpoints came from, because the numeric thresholds were not saved.

“My next priorities are to confirm the residue units and unusual samples, evaluate continuous residue prediction against these baselines, and test the additional images through treatment-supervised pretraining. That experiment must exclude every photo of a test tree from pretraining, including photos without ICP measurements. Today I especially need a reliable map from every image to its treatment, tree, leaf and acquisition batch.”

## Compact verified classification results

Each entry is the arithmetic mean of three seed-specific OOF metrics on the same 209 samples. Class support per seed: low 70, medium 69, high 70. These are **209 samples, not 627 independent leaves**. Single-view inference only.

| Method | Macro-F1 | Accuracy | Medium precision | Medium recall | Predicted medium fraction | Ordinal MAE (bins) | Low↔high fraction | Log loss | RPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CE | 0.453 | 0.455 | 0.332 | 0.353 | 0.351 | 0.643 | 0.097 | 1.400 | 0.222 |
| CE+RPS | 0.459 | 0.456 | 0.325 | 0.367 | 0.370 | 0.630 | 0.086 | 1.454 | 0.223 |
| CORAL | 0.484 | 0.496 | 0.370 | 0.280 | 0.258 | 0.604 | 0.100 | 1.080 | 0.197 |
| Treatment only (computed here) | 0.511 | 0.518 | 0.357 | 0.319 | 0.287 | 0.557 | 0.075 | 1.264 | 0.191 |

CORAL − CE macro-F1 = +0.030; paired 95% tree-bootstrap interval **−0.034 to +0.092**. CE+RPS − CE = +0.005; interval **−0.021 to +0.032**. No primary-metric ordinal winner is established. CORAL − treatment-only = −0.027; interval −0.154 to +0.106. Intervals resample 21 whole trees together across all seeds, conditional on the existing fits; they are exploratory, not a substitute for independent trees.

## Five questions to ask aloud

1. **Can we get the exact additional-image count and an image-level table linking treatment, unique tree, unique leaf, and ICP status?** Please confirm that tree numbers restart within treatment, all current controls share one tree, and identify repeated photos of the same leaf.
2. **Were treatment groups photographed in different sessions or under different conditions?** We need dates/batches, camera, lighting, background, framing, and whether stickers, labels or other visible cues identify treatment.
3. **Can we verify the copper-mass conversion, dilution factors and area convention, and review `1000 ppm 1 7` and `800 ppm 2 4`?** Their recorded residues are 13.219 and 0.0105 mg Cu/cm². Also, who created the CSV bins, using which cutpoints and which samples?
4. **For the additional images, what were the spray, drying and collection timelines, and may all images be used for model development?** Were acquisition and collection protocols consistent across trees and batches?
5. **What prediction would be useful operationally: a single-leaf value or an average over sampled leaves from a tree?** What error in mg Cu/cm² would be acceptable, and would nominal treatment be known at prediction time?

## What is ready today

- **Report now:** completed paired classification; recorded tree-disjoint splits; residue variation; newly computed treatment/global baselines; exploratory class-composition aggregation.
- **Quick computation once missing files are supplied:** image-regression metrics and tree means from regression OOF files; train–inner-validation curves from histories; exact bin provenance from the labeling script. No retraining is needed for these calculations if those outputs already exist.
- **Requires training if not already run:** continuous image regression, image plus treatment, revised bins if current labels used all-data quantiles, and treatment-pretraining comparisons.

The archive contains neither training histories nor model checkpoint files. Completed OOF outputs—not folder names or checkpoint appearances—establish that classification finished. Underfitting versus overfitting cannot be diagnosed from this archive.
