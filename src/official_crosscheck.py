"""Rebuild per-game predictions and cross-check with SoccerNet's evaluator."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from SoccerNet.Evaluation.ActionSpotting import evaluate as official_evaluate

from tgre_experiment import ARTIFACTS, CLASSES, Corpus, make_model, nms_predictions, score_half, game_path, LABEL_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    corpus = Corpus(ARTIFACTS / "modality_coverage.json")
    checkpoint = torch.load(ARTIFACTS / "checkpoints" / f"{args.model}_seed{args.seed}_epoch{args.epochs}.pt", map_location="cpu", weights_only=False)
    model = make_model(args.model, 512, len(corpus.vocab))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    root = ARTIFACTS / "official_crosscheck" / f"{args.model}_seed{args.seed}"
    labels_root = root / "labels"
    predictions_root = root / "predictions"
    for game in corpus.games["valid"]:
        label_target = game_path(labels_root, game) / "Labels-v2.json"
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(game_path(LABEL_ROOT, game) / "Labels-v2.json", label_target)
        output = {"UrlLocal": game.replace("\\", "/"), "predictions": []}
        for half in (1, 2):
            data = corpus.half(game, half)
            scores = score_half(model, data, torch.device("cpu"))
            for klass, entries in nms_predictions(scores).items():
                for seconds, confidence in entries:
                    output["predictions"].append({"gameTime": f"{half} - {int(seconds)//60:02d}:{int(seconds)%60:02d}", "label": CLASSES[klass], "position": str(int(seconds * 1000)), "half": str(half), "confidence": str(confidence)})
        prediction_target = game_path(predictions_root, game) / "results_spotting.json"
        prediction_target.parent.mkdir(parents=True, exist_ok=True)
        prediction_target.write_text(json.dumps(output), encoding="utf-8")

    serializable = {}
    for metric in ("tight", "loose", "at5"):
        result = official_evaluate(
            str(labels_root), str(predictions_root), prediction_file="results_spotting.json",
            split="valid", version=2, framerate=2, metric=metric, dataset=None,
            EVENT_DICTIONARY={name: index for index, name in enumerate(CLASSES)},
        )
        serializable[metric] = {key: value.tolist() if hasattr(value, "tolist") else value for key, value in result.items()}
    (root / "official_metrics.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
