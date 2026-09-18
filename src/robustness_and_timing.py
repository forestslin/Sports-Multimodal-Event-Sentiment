"""Evaluate missing ASR and measured CPU inference cost for final checkpoints."""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from tgre_experiment import ARTIFACTS, Corpus, evaluate, make_model, score_half


def main() -> None:
    results_path = ARTIFACTS / "experiment_results.json"
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    corpus = Corpus(ARTIFACTS / "modality_coverage.json")
    device = torch.device("cpu")
    output = []

    for row in rows:
        model_name = row["model"]
        seed = int(row["seed"])
        epochs = int(row["epochs"])
        checkpoint = torch.load(
            ARTIFACTS / "checkpoints" / f"{model_name}_seed{seed}_epoch{epochs}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = make_model(model_name, 512, len(corpus.vocab)).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        record = {"model": model_name, "seed": seed}
        if model_name in {"late", "rtgre", "rtgre_flow", "tcn_fusion"}:
            missing = evaluate(model, corpus, device, zero_text=True)
            record["event_mAP@10s_zero_text"] = missing["event_mAP@10s"]

        # Warm up one half, then time complete validation inference three times.
        first_game = corpus.games["valid"][0]
        score_half(model, corpus.half(first_game, 1), device)
        elapsed = []
        bins = 0
        for _ in range(3):
            count = 0
            started = time.perf_counter()
            for game in corpus.games["valid"]:
                for half in (1, 2):
                    data = corpus.half(game, half)
                    score_half(model, data, device)
                    count += len(data.visual)
            elapsed.append(time.perf_counter() - started)
            bins = count
        median_seconds = statistics.median(elapsed)
        record.update(
            {
                "validation_bins": bins,
                "timing_repeats": 3,
                "inference_seconds_median": median_seconds,
                "bins_per_second": bins / median_seconds,
                "milliseconds_per_bin": 1000.0 * median_seconds / bins,
            }
        )
        output.append(record)
        print(json.dumps(record), flush=True)

    payload = {
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "CPU",
        },
        "results": output,
    }
    (ARTIFACTS / "robustness_and_timing.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
