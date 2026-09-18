"""Sensitivity of the late-fusion versus visual game-cluster bootstrap to empty classes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from game_bootstrap import (
    ARTIFACTS,
    BOOTSTRAPS,
    BOOTSTRAP_SEED,
    CLASSES,
    Corpus,
    checkpoint_profile,
    prepare_class,
)


def bootstrap_map(profile, weights: np.ndarray, drop_empty: bool) -> np.ndarray:
    class_values = []
    class_present = []
    for klass in range(len(CLASSES)):
        hits, game_ids, truth_counts = prepare_class(profile, klass)
        truth_total = weights @ truth_counts
        ap = np.zeros(weights.shape[0], dtype=np.float64)
        for start in range(0, len(weights), 50):
            stop = min(start + 50, len(weights))
            prediction_weights = weights[start:stop, game_ids]
            tp = prediction_weights * hits[None, :]
            cumulative_tp = np.cumsum(tp, axis=1)
            cumulative_predictions = np.cumsum(prediction_weights, axis=1)
            precision = np.divide(
                cumulative_tp,
                cumulative_predictions,
                out=np.zeros_like(cumulative_tp),
                where=cumulative_predictions > 0,
            )
            numerator = np.sum(precision * tp, axis=1)
            ap[start:stop] = np.divide(
                numerator,
                truth_total[start:stop],
                out=np.zeros_like(numerator),
                where=truth_total[start:stop] > 0,
            )
        class_values.append(ap)
        class_present.append(truth_total > 0)
    values = np.stack(class_values, axis=1)
    present = np.stack(class_present, axis=1)
    if not drop_empty:
        return values.mean(axis=1)
    return np.divide(
        np.sum(values * present, axis=1),
        np.sum(present, axis=1),
        out=np.zeros(values.shape[0], dtype=np.float64),
        where=np.sum(present, axis=1) > 0,
    )


def main() -> None:
    corpus = Corpus(ARTIFACTS / "modality_coverage.json")
    seeds = [2026, 2027, 2028, 2029, 2030]
    profiles = {}
    for model in ("visual", "late"):
        for seed in seeds:
            print(json.dumps({"profiling": model, "seed": seed}), flush=True)
            profiles[(model, seed)] = checkpoint_profile(model, seed, 20, corpus)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = np.stack([
        np.bincount(
            rng.integers(0, len(corpus.games["valid"]), len(corpus.games["valid"])),
            minlength=len(corpus.games["valid"]),
        )
        for _ in range(BOOTSTRAPS)
    ])
    results = {}
    for label, drop_empty in (("fixed_17_class_macro", False), ("available_class_macro", True)):
        differences = []
        for seed in seeds:
            late = bootstrap_map(profiles[("late", seed)], weights, drop_empty)
            visual = bootstrap_map(profiles[("visual", seed)], weights, drop_empty)
            differences.append(late - visual)
        averaged = np.stack(differences).mean(axis=0)
        results[label] = {
            "ci95_low": float(np.quantile(averaged, 0.025)),
            "ci95_high": float(np.quantile(averaged, 0.975)),
            "replicate_mean": float(np.mean(averaged)),
        }
    payload = {
        "comparison": "late minus visual",
        "games": len(corpus.games["valid"]),
        "training_seeds": seeds,
        "replicates": BOOTSTRAPS,
        "seed_aggregation": "Within each replicate, recompute AP separately for each paired seed and model; subtract visual from late within seed; then average the five paired seed differences.",
        "duplicate_games": "Game multiplicities weight both detections and ground-truth counts identically for both models.",
        "empty_class_primary": "The prespecified 17-class macro average assigns AP=0 to a class absent from a replicate and retains the fixed denominator of 17; because resampling is paired, that class contributes zero to the model difference.",
        "sensitivity": "The available-class macro average excludes classes with no ground truth in that replicate from both paired models.",
        "results": results,
    }
    out = ARTIFACTS / "game_cluster_bootstrap_sensitivity.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
