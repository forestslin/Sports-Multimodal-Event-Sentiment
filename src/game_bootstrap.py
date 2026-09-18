"""Game-cluster bootstrap for paired model differences on the validation subset."""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tgre_experiment import ARTIFACTS, CLASSES, Corpus, make_model, nms_predictions, score_half


TOLERANCE = 10.0
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 142751


def ranked_hits(predicted: list[tuple[float, float]], truth: list[float]) -> tuple[np.ndarray, np.ndarray, int]:
    truth = sorted(truth)
    used = np.zeros(len(truth), dtype=bool)
    scores = []
    hits = []
    for time, score in sorted(predicted, key=lambda item: -item[1]):
        left = bisect_left(truth, time - TOLERANCE)
        right = bisect_right(truth, time + TOLERANCE)
        candidates = [(abs(time - truth[index]), index) for index in range(left, right) if not used[index]]
        if candidates:
            _, index = min(candidates)
            used[index] = True
            hits.append(1)
        else:
            hits.append(0)
        scores.append(score)
    return np.asarray(scores, dtype=np.float32), np.asarray(hits, dtype=np.uint8), len(truth)


def checkpoint_profile(model_name: str, seed: int, epochs: int, corpus: Corpus) -> list[list[tuple[np.ndarray, np.ndarray, int]]]:
    checkpoint = torch.load(
        ARTIFACTS / "checkpoints" / f"{model_name}_seed{seed}_epoch{epochs}.pt",
        map_location="cpu", weights_only=False,
    )
    model = make_model(model_name, 512, len(corpus.vocab))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    profile = []
    for game in corpus.games["valid"]:
        predicted: dict[int, list[tuple[float, float]]] = defaultdict(list)
        truth: dict[int, list[float]] = defaultdict(list)
        for half in (1, 2):
            data = corpus.half(game, half)
            scores = score_half(model, data, torch.device("cpu"))
            offset = (half - 1) * 10000.0
            for klass, entries in nms_predictions(scores).items():
                predicted[klass].extend((offset + time, score) for time, score in entries)
            for time, klass in data.events:
                truth[klass].append(offset + time)
        profile.append([ranked_hits(predicted[klass], truth[klass]) for klass in range(len(CLASSES))])
    return profile


def prepare_class(profile, klass: int):
    scores, hits, game_ids = [], [], []
    truth_counts = np.zeros(len(profile), dtype=np.int32)
    for game_index, game in enumerate(profile):
        game_scores, game_hits, n_truth = game[klass]
        scores.append(game_scores)
        hits.append(game_hits)
        game_ids.append(np.full(len(game_scores), game_index, dtype=np.int16))
        truth_counts[game_index] = n_truth
    scores = np.concatenate(scores)
    hits = np.concatenate(hits)
    game_ids = np.concatenate(game_ids)
    order = np.argsort(-scores, kind="stable")
    return hits[order].astype(np.float64), game_ids[order], truth_counts


def bootstrap_map(profile, weights: np.ndarray, batch_size: int = 50) -> np.ndarray:
    output = np.zeros(weights.shape[0], dtype=np.float64)
    used_classes = 0
    for klass in range(len(CLASSES)):
        hits, game_ids, truth_counts = prepare_class(profile, klass)
        if truth_counts.sum() == 0:
            continue
        used_classes += 1
        class_ap = np.zeros(weights.shape[0], dtype=np.float64)
        for start in range(0, len(weights), batch_size):
            sample_weights = weights[start:start + batch_size]
            prediction_weights = sample_weights[:, game_ids]
            tp = prediction_weights * hits[None, :]
            cumulative_tp = np.cumsum(tp, axis=1)
            cumulative_predictions = np.cumsum(prediction_weights, axis=1)
            precision = np.divide(cumulative_tp, cumulative_predictions, out=np.zeros_like(cumulative_tp), where=cumulative_predictions > 0)
            truth_total = sample_weights @ truth_counts
            numerator = np.sum(precision * tp, axis=1)
            class_ap[start:start + len(sample_weights)] = np.divide(numerator, truth_total, out=np.zeros_like(numerator), where=truth_total > 0)
        output += class_ap
    return output / used_classes


def main() -> None:
    rows = json.loads((ARTIFACTS / "experiment_results.json").read_text(encoding="utf-8"))
    corpus = Corpus(ARTIFACTS / "modality_coverage.json")
    selected = {"visual", "late", "rtgre", "rtgre_flow", "tcn_visual", "tcn_fusion"}
    profiles = {}
    for row in rows:
        model = row["model"]
        if model not in selected:
            continue
        seed = int(row["seed"])
        print(json.dumps({"profiling": model, "seed": seed}), flush=True)
        profiles[(model, seed)] = checkpoint_profile(model, seed, int(row["epochs"]), corpus)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = np.stack([np.bincount(rng.integers(0, len(corpus.games["valid"]), len(corpus.games["valid"])), minlength=len(corpus.games["valid"])) for _ in range(BOOTSTRAPS)])
    maps = {key: bootstrap_map(profile, weights) for key, profile in profiles.items()}
    comparisons = [
        ("late", "visual"), ("rtgre", "visual"), ("rtgre_flow", "visual"),
        ("tcn_visual", "visual"), ("tcn_fusion", "visual"),
        ("rtgre", "late"), ("rtgre_flow", "rtgre"),
    ]
    output = []
    row_lookup = {(row["model"], int(row["seed"])): row for row in rows}
    for left, right in comparisons:
        seeds = sorted({seed for model, seed in maps if model == left} & {seed for model, seed in maps if model == right})
        if not seeds:
            continue
        seed_bootstrap_differences = np.stack([maps[(left, seed)] - maps[(right, seed)] for seed in seeds])
        bootstrap_difference = seed_bootstrap_differences.mean(axis=0)
        point_differences = [row_lookup[(left, seed)]["event_mAP@10s"] - row_lookup[(right, seed)]["event_mAP@10s"] for seed in seeds]
        output.append({
            "comparison": f"{left} minus {right}",
            "metric": "class-macro event mAP with +/-10-s matching",
            "games": len(corpus.games["valid"]),
            "paired_training_seeds": seeds,
            "bootstrap_replicates": BOOTSTRAPS,
            "point_mean_difference_across_seeds": float(np.mean(point_differences)),
            "game_cluster_bootstrap_ci95_low": float(np.quantile(bootstrap_difference, 0.025)),
            "game_cluster_bootstrap_ci95_high": float(np.quantile(bootstrap_difference, 0.975)),
        })
    payload = {
        "method": "paired nonparametric bootstrap over validation games; identical game multiplicities for both models; AP recomputed from game-weighted ranked detections within each replicate",
        "results": output,
    }
    (ARTIFACTS / "game_cluster_bootstrap.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
