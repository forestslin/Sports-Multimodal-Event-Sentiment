"""Evaluate whether the main comparison depends on the NMS radius."""

from __future__ import annotations

import json

import torch

from tgre_experiment import ARTIFACTS, Corpus, evaluate, make_model


def main() -> None:
    rows = json.loads((ARTIFACTS / "experiment_results.json").read_text(encoding="utf-8"))
    corpus = Corpus(ARTIFACTS / "modality_coverage.json")
    output = []
    for row in rows:
        if row["model"] not in {"visual", "late"}:
            continue
        model_name = row["model"]
        seed = int(row["seed"])
        epochs = int(row["epochs"])
        checkpoint = torch.load(
            ARTIFACTS / "checkpoints" / f"{model_name}_seed{seed}_epoch{epochs}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = make_model(model_name, 512, len(corpus.vocab))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        for radius in (4.0, 8.0, 12.0):
            metrics = evaluate(model, corpus, torch.device("cpu"), nms_seconds=radius)
            output.append({
                "model": model_name,
                "seed": seed,
                "nms_radius_seconds": radius,
                "event_mAP@5s": metrics["event_mAP@5s"],
                "event_mAP@10s": metrics["event_mAP@10s"],
                "event_mAP@20s": metrics["event_mAP@20s"],
            })
            print(json.dumps(output[-1]), flush=True)
    (ARTIFACTS / "nms_sensitivity.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
