"""Tree grouping, ordinal objectives, and clustered evaluation helpers."""

import math
import re
from decimal import Decimal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

from baseline_pipeline.experiment_protocol import CLASS_NAMES, write_csv


def resolve_tree_groups(sample_ids, explicit_groups=None):
    """Treatment-local tree numbers; every c N sample belongs to one tree."""
    trees = []
    for sample_id in sample_ids:
        name = str(sample_id).strip()

        if re.fullmatch(r"c\s+\d+", name, flags=re.IGNORECASE):
            trees.append("control_tree")
            continue

        match = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s+ppm\s+(\d+)\s+(\d+)",
            name,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError(
                f"Cannot identify the tree for sample_id={sample_id!r}. "
                "Expected 'X ppm Y Z' or 'c N'."
            )

        treatment, tree, _leaf = match.groups()
        treatment = format(Decimal(treatment).normalize(), "f")
        trees.append(f"{treatment}ppm_tree_{int(tree)}")

    trees = np.asarray(trees, dtype=str)
    groups = (
        trees.copy() if explicit_groups is None
        else np.asarray(explicit_groups, dtype=str)
    )

    if groups.shape != trees.shape or any(not g.strip() for g in groups):
        raise ValueError("Invalid explicit group column.")

    # Explicit groups can merge trees, but must never split a physical tree.
    for tree in np.unique(trees):
        if len(np.unique(groups[trees == tree])) != 1:
            raise ValueError(
                f"Explicit group column splits physical tree {tree!r}. "
                "Omit --group-column to use the derived tree groups."
            )
    return trees, groups


class CoralModel(nn.Module):
    """Shared scalar score with two explicitly ordered CORAL thresholds.

    Thresholds are center +/- positive_gap/2. This parameterizes the
    ordered shared-score model directly; no post-hoc probability sorting
    or repair is used.
    """

    def __init__(self, score_model):
        super().__init__()
        self.score_model = score_model
        self.center = nn.Parameter(torch.tensor(0.0))

        # With score zero, these initial thresholds give approximately
        # equal low/medium/high probabilities.
        initial_gap = 2.0 * math.log(2.0)
        self.raw_gap = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_gap)))
        )

    @property
    def fc(self):
        # Preserve the existing ResNet BatchNorm-freezing path.
        return self.score_model.fc

    def forward(self, images):
        score = self.score_model(images)
        if score.ndim != 2 or score.shape[1] != 1:
            raise ValueError("CORAL's underlying model must output (N, 1).")
        gap = F.softplus(self.raw_gap) + 1e-6
        thresholds = torch.stack((
            self.center - gap / 2,
            self.center + gap / 2,
        ))
        return score - thresholds.unsqueeze(0)


def ordinal_probabilities(output, config):
    if config.loss != "coral":
        return output.softmax(dim=1)

    # q[:, k] = P(Y > k), with q[:, 0] >= q[:, 1].
    q = output.sigmoid()
    return torch.stack((
        1.0 - q[:, 0],
        q[:, 0] - q[:, 1],
        q[:, 1],
    ), dim=1)


def normalized_rps(probabilities, target):
    """Mean squared CDF error across the K-1 nonredundant boundaries."""
    k = probabilities.shape[1]
    boundaries = torch.arange(k - 1, device=target.device)
    target_cdf = (target[:, None] <= boundaries).to(probabilities.dtype)
    predicted_cdf = probabilities.cumsum(dim=1)[:, :-1]
    return (predicted_cdf - target_cdf).square().mean()


def classification_loss(output, target, config):
    if config.loss == "coral":
        boundaries = torch.arange(2, device=target.device)
        levels = (target[:, None] > boundaries).to(output.dtype)
        # Sum binary tasks, then average over samples.
        return F.binary_cross_entropy_with_logits(
            output, levels, reduction="none"
        ).sum(dim=1).mean()

    if config.loss == "ce":
        return F.cross_entropy(output, target)

    probabilities = output.softmax(dim=1)
    rps = normalized_rps(probabilities, target)
    if config.loss == "rps":
        return rps
    if config.loss == "ce_rps":
        return F.cross_entropy(output, target) + config.rps_weight * rps
    raise ValueError(f"Unknown classification loss: {config.loss!r}")


def decision_metrics(y, prediction):
    """Discrete metrics only; regression does not supply class probabilities."""
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    distance = np.abs(y - prediction)

    result = {
        "accuracy": float(np.mean(y == prediction)),
        "macro_f1": float(f1_score(
            y, prediction, labels=[0, 1, 2],
            average="macro", zero_division=0,
        )),
        "ordinal_mae": float(distance.mean()),
        "extreme_error_rate": float(np.mean(distance == 2)),
        "low_to_high_count": int(np.sum((y == 0) & (prediction == 2))),
        "high_to_low_count": int(np.sum((y == 2) & (prediction == 0))),
    }
    for name, function in (
        ("recall", recall_score),
        ("precision", precision_score),
    ):
        values = function(
            y, prediction, labels=[0, 1, 2],
            average=None, zero_division=0,
        )
        result.update({
            f"{name}_{label}": float(value)
            for label, value in zip(CLASS_NAMES, values)
        })
    for i, label in enumerate(CLASS_NAMES):
        result[f"support_{label}"] = int(np.sum(y == i))
        result[f"predicted_{label}"] = int(np.sum(prediction == i))
    return result


def validate_ordinal_config(config):
    if config.loss not in ("ce", "coral", "ce_rps", "rps"):
        raise ValueError("Unknown classification loss.")
    if not np.isfinite(config.rps_weight) or config.rps_weight < 0:
        raise ValueError("rps_weight must be finite and nonnegative.")
    if config.task == "icp_regression" and config.loss != "ce":
        raise ValueError(
            "--loss selects a classification objective. "
            "Omit it for regression, which retains SmoothL1 training."
        )
    if config.selection == "legacy_outer":
        raise ValueError(
            "legacy_outer is disabled for this grouped protocol. "
            "Use inner_refit or fixed."
        )

    rates = config.learning_rates or (config.learning_rate,)
    if len(set(rates)) != len(rates):
        raise ValueError("Duplicate learning-rate candidates.")
    if any(not np.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Learning rates must be finite and positive.")
    if (
        not config.evaluate_only
        and config.selection != "inner_refit"
        and len(rates) > 1
    ):
        raise ValueError("A learning-rate search requires inner_refit.")

    if config.inner_group_folds < 2:
        raise ValueError("inner_group_folds must be at least two.")
    if config.bootstrap_replicates < 0 or config.bootstrap_seed < 0:
        raise ValueError("Invalid bootstrap configuration.")
    if config.regression_bins and config.task != "icp_regression":
        raise ValueError("--regression-bins is only for regression.")
    if (
        config.regression_bins
        and config.bin_policy == "csv"
        and config.thresholds is None
    ):
        raise ValueError(
            "Binning regression predictions with bin_policy='csv' requires "
            "--thresholds LOWER UPPER. Supply the original CSV cutpoints; "
            "the code verifies that they reproduce every CSV label."
        )


def write_tree_bootstrap(all_rows, config, out, metric_function):
    """Whole-group bootstrap of OOF predictions, averaging metrics over seeds.

    The same sampled groups are used for every seed and inference condition.
    Intervals are conditional on the fitted models and existing CV splits;
    models are not retrained inside bootstrap replicates.
    """
    if config.bootstrap_replicates == 0:
        return

    if config.task == "bin_classification":
        keys = [
            "macro_f1", "recall_medium", "precision_medium",
            "ordinal_mae", "extreme_error_rate", "log_loss", "rps",
        ]
    else:
        keys = ["mae", "rmse"]
        if config.regression_bins:
            keys += [
                "binned_macro_f1", "binned_recall_medium",
                "binned_precision_medium", "binned_ordinal_mae",
                "binned_extreme_error_rate",
            ]

    intervals, draw_rows = [], []
    conditions = list(dict.fromkeys(r["inference"] for r in all_rows))

    for condition in conditions:
        by_seed = []
        for seed in config.random_seeds:
            rows = sorted(
                (
                    row for row in all_rows
                    if row["seed"] == seed and row["inference"] == condition
                ),
                key=lambda row: row["dataset_index"],
            )
            by_seed.append(rows)

        reference = [
            (row["sample_id"], row["group"]) for row in by_seed[0]
        ]
        if len({sample for sample, _group in reference}) != len(reference):
            raise ValueError("Duplicate OOF samples within a bootstrap seed.")
        for rows in by_seed[1:]:
            if [(r["sample_id"], r["group"]) for r in rows] != reference:
                raise ValueError("OOF samples/groups differ across seeds.")

        groups = np.asarray([group for _sample, group in reference])
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError("Bootstrap requires at least two groups.")
        members = [
            np.flatnonzero(groups == group) for group in unique_groups
        ]

        original_metrics = [
            metric_function(rows, config) for rows in by_seed
        ]
        point = {
            key: float(np.mean([metrics[key] for metrics in original_metrics]))
            for key in keys
        }

        # Reset for each inference condition to preserve paired draws.
        rng = np.random.default_rng(config.bootstrap_seed)
        samples = {key: [] for key in keys}
        missing_class_draws = 0

        for draw in range(config.bootstrap_replicates):
            selected = rng.integers(
                0, len(unique_groups), size=len(unique_groups)
            )
            indices = np.concatenate([members[i] for i in selected])
            metrics_across_seeds = []
            missing_class = False

            for rows in by_seed:
                sampled = [rows[int(i)] for i in indices]
                metrics_across_seeds.append(
                    metric_function(sampled, config)
                )
                if config.task == "bin_classification":
                    missing_class |= len({
                        row["target"] for row in sampled
                    }) < 3
                elif config.regression_bins:
                    missing_class |= len({
                        row["target_bin"] for row in sampled
                    }) < 3

            missing_class_draws += int(missing_class)
            result = {
                "inference": condition,
                "draw": draw,
                "missing_class": int(missing_class),
            }
            for key in keys:
                value = float(np.mean([
                    metrics[key] for metrics in metrics_across_seeds
                ]))
                samples[key].append(value)
                result[key] = value
            draw_rows.append(result)

        for key in keys:
            lower, upper = np.quantile(samples[key], [0.025, 0.975])
            intervals.append({
                "inference": condition,
                "metric": key,
                "estimate_mean_across_seeds": point[key],
                "lower_95": float(lower),
                "upper_95": float(upper),
                "n_groups": len(unique_groups),
                "n_seeds": len(by_seed),
                "bootstrap_replicates": config.bootstrap_replicates,
                "fraction_draws_missing_class": (
                    missing_class_draws / config.bootstrap_replicates
                ),
                "method": "whole_group_percentile_conditional_on_fits",
            })

    write_csv(out / "tree_bootstrap_intervals.csv", intervals)
    write_csv(out / "tree_bootstrap_draws.csv", draw_rows)


def self_test():
    """CPU smoke checks; no images, pretrained weights, or CUDA needed."""
    from types import SimpleNamespace
    from baseline_pipeline.experiment_protocol import (
        assign_labels, classification_metrics,
        make_outer_splits, make_inner_split,
    )

    trees, groups = resolve_tree_groups([
        "1000 ppm 1 7", "1000 ppm 1 8", "400 ppm 1 7", "c 1", "C 2",
    ])
    assert trees[0] == trees[1]
    assert trees[0] != trees[2]
    assert trees[3] == trees[4] == "control_tree"

    try:
        resolve_tree_groups(["1000 ppm 1 7", "1000 ppm 1 8"], ["a", "b"])
    except ValueError:
        pass
    else:
        raise AssertionError("A physical tree was allowed to span groups.")

    try:
        resolve_tree_groups(["unrecognized sample"])
    except ValueError:
        pass
    else:
        raise AssertionError("An unrecognized sample ID was accepted.")

    ids, labels = [], []
    for dose in (400, 600, 800, 1000):
        for tree in (1, 2, 3):
            for leaf in (1, 2, 3):
                ids.append(f"{dose} ppm {tree} {leaf}")
                labels.append(leaf - 1)
    ids += ["c 1", "c 2", "c 3"]
    labels += [0, 0, 0]
    labels = np.asarray(labels)
    _, groups = resolve_tree_groups(ids)

    splits = make_outer_splits(labels, 4, 42, groups)
    assert sorted(np.concatenate([test for _, test in splits])) == list(
        range(len(ids))
    )
    for train, test in splits:
        assert not set(groups[train]) & set(groups[test])
        a, b = make_inner_split(
            train, labels, 42, groups=groups, group_folds=3,
        )
        assert set(a) | set(b) == set(train)
        assert not set(groups[a]) & set(groups[b])
        assert not (set(a) | set(b)) & set(test)

    torch.manual_seed(42)
    model = CoralModel(nn.Linear(4, 1))
    x = torch.randn(9, 4)
    y = torch.arange(9) % 3
    cfg = SimpleNamespace(loss="coral", rps_weight=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(5):
        logits = model(x)
        probabilities = ordinal_probabilities(logits, cfg)
        assert logits.shape == (9, 2)
        assert probabilities.shape == (9, 3)
        assert torch.all(probabilities >= 0)
        assert torch.allclose(probabilities.sum(1), torch.ones(9))
        assert torch.all(logits[:, 0] >= logits[:, 1])
        loss = classification_loss(logits, y, cfg)
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        optimizer.step()

    logits = torch.randn(9, 3, requires_grad=True)
    cfg = SimpleNamespace(loss="ce_rps", rps_weight=0.0)
    assert torch.allclose(
        classification_loss(logits, y, cfg),
        F.cross_entropy(logits, y),
    )

    perfect = F.one_hot(y, 3).float()
    assert normalized_rps(perfect, y).item() == 0
    wrong_extreme = torch.tensor([[0.0, 0.0, 1.0]])
    assert normalized_rps(wrong_extreme, torch.tensor([0])).item() == 1
    metrics = classification_metrics([0], [2], [[0.0, 0.0, 1.0]])
    assert metrics["rps"] == 1
    assert metrics["low_to_high_count"] == 1

    residue = np.array([0.0, 1.0, 2.0])
    csv_labels = np.array([0, 1, 2])
    assigned, edges = assign_labels(
        residue, csv_labels, [0, 1],
        policy="csv", thresholds=(0.5, 1.5),
    )
    assert np.array_equal(assigned, csv_labels)
    assert np.array_equal(edges, [0.5, 1.5])
    try:
        assign_labels(
            residue, csv_labels, [0, 1],
            policy="csv", thresholds=(1.0, 1.5),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Inconsistent CSV cutpoints were accepted.")

    print("Tree grouping, split isolation, ordinal loss, and RPS checks passed.")


if __name__ == "__main__":
    self_test()