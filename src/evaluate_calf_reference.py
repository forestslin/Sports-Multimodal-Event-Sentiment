"""Evaluate the released CALF SoccerNet-v2 checkpoint on our 54-game subset.

This is an external reference only: the checkpoint was trained on the full
official SoccerNet-v2 training split and its 120-s windows are offline.  Our
models use the 176-game vision/text intersection and causal histories.
"""

from __future__ import annotations

import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from SoccerNet.Evaluation.ActionSpotting import evaluate as official_evaluate

from tgre_experiment import (
    ARTIFACTS,
    BIN_SECONDS,
    CLASSES,
    LABEL_ROOT,
    FEATURE_ROOT,
    Corpus,
    average_precision,
    game_path,
)


ROOT = Path(__file__).resolve().parents[1]
CALF_ROOT = ROOT / "external" / "sn-spotting-main" / "Benchmarks" / "CALF"
CHECKPOINT = CALF_ROOT / "models" / "CALF_benchmark" / "model.pth.tar"
CALF_CLASSES = [
    "Penalty", "Kick-off", "Goal", "Substitution", "Offside",
    "Shots on target", "Shots off target", "Clearance", "Ball out of play",
    "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick", "Corner",
    "Yellow card", "Red card", "Yellow->red card",
]
OUR_INDEX = {name: i for i, name in enumerate(CLASSES)}


def load_model() -> torch.nn.Module:
    spec = importlib.util.spec_from_file_location("calf_model", CALF_ROOT / "src" / "model.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ContextAwareModel(
        input_size=512, num_classes=17, chunk_size=240, dim_capsule=16,
        receptive_field=80, num_detections=15, framerate=2,
    )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def feats2clip(feats: torch.Tensor, stride: int = 160, clip_length: int = 240) -> torch.Tensor:
    starts = torch.arange(start=0, end=feats.shape[0] - 1, step=stride)
    idx = torch.stack([starts + i for i in range(clip_length)], dim=1)
    idx = idx.clamp(0, feats.shape[0] - 1)
    idx[-1] = torch.arange(clip_length) + feats.shape[0] - clip_length
    return feats[idx, :]


def timestamps2long(output: torch.Tensor, video_size: int, chunk_size: int = 240, receptive_field: int = 80) -> torch.Tensor:
    start = 0
    last = False
    margin = receptive_field // 2
    dense = torch.full((video_size, output.shape[-1] - 2), -1.0)
    for batch in range(output.shape[0]):
        local = torch.full((chunk_size, output.shape[-1] - 2), -1.0)
        for detection in output[batch]:
            frame = int(torch.floor(detection[1] * (chunk_size - 1)).item())
            klass = int(torch.argmax(detection[2:]).item())
            local[frame, klass] = max(local[frame, klass], detection[0])
        if start == 0:
            dense[: chunk_size - margin] = local[: chunk_size - margin]
        elif last:
            dense[start + margin : start + chunk_size] = local[margin:]
            break
        else:
            dense[start + margin : start + chunk_size - margin] = local[margin : chunk_size - margin]
        start += chunk_size - 2 * margin
        if start + chunk_size >= video_size:
            start = video_size - chunk_size
            last = True
    return dense


def nms_dense(scores: np.ndarray, radius_frames: int = 16) -> dict[int, list[tuple[float, float]]]:
    result: dict[int, list[tuple[float, float]]] = {}
    for calf_index, name in enumerate(CALF_CLASSES):
        values = scores[:, calf_index]
        order = np.argsort(-values)
        kept: list[tuple[float, float]] = []
        for frame in order:
            score = float(values[frame])
            if score < 0:
                break
            seconds = float(frame) / 2.0
            if all(abs(seconds - prior) > radius_frames / 2.0 for prior, _ in kept):
                kept.append((seconds, score))
        result[OUR_INDEX[name]] = kept
    return result


def write_predictions(game: str, predictions: dict[int, list[tuple[float, float]]], half: int, output: dict) -> None:
    for klass, entries in predictions.items():
        for seconds, confidence in entries:
            output["predictions"].append({
                "gameTime": f"{half} - {int(seconds) // 60:02d}:{int(seconds) % 60:02d}",
                "label": CLASSES[klass],
                "position": str(int(seconds * 1000)),
                "half": str(half),
                "confidence": str(confidence),
            })


def main() -> None:
    torch.set_num_threads(4)
    model = load_model()
    corpus = Corpus(ARTIFACTS / "modality_coverage.json")
    output_root = ARTIFACTS / "calf_external_reference"
    labels_root = output_root / "labels"
    predictions_root = output_root / "predictions"
    all_predictions: dict[int, list[tuple[float, float]]] = defaultdict(list)
    all_truth: dict[int, list[float]] = defaultdict(list)
    offset = 0.0
    started = time.perf_counter()

    with torch.no_grad():
        for game_index, game in enumerate(corpus.games["valid"], 1):
            label_target = game_path(labels_root, game) / "Labels-v2.json"
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_bytes((game_path(LABEL_ROOT, game) / "Labels-v2.json").read_bytes())
            output = {"UrlLocal": game.replace("\\", "/"), "predictions": []}
            for half in (1, 2):
                features = np.load(game_path(FEATURE_ROOT, game) / f"{half}_ResNET_TF2_PCA512.npy")
                clips = feats2clip(torch.from_numpy(features).float()).unsqueeze(1)
                chunk_outputs = []
                for start in range(0, len(clips), 8):
                    _, spotting = model(clips[start : start + 8])
                    chunk_outputs.append(spotting.cpu())
                dense = timestamps2long(torch.cat(chunk_outputs), len(features)).numpy()
                predictions = nms_dense(dense)
                write_predictions(game, predictions, half, output)
                for klass, entries in predictions.items():
                    all_predictions[klass].extend((offset + t, score) for t, score in entries)
                for event_time, klass in corpus.half(game, half).events:
                    all_truth[klass].append(offset + event_time)
                offset += 10000.0
            prediction_target = game_path(predictions_root, game) / "results_spotting.json"
            prediction_target.parent.mkdir(parents=True, exist_ok=True)
            prediction_target.write_text(json.dumps(output), encoding="utf-8")
            print(f"CALF {game_index}/54 {game}", flush=True)

    metrics: dict[str, object] = {
        "reference_status": "external non-comparable reference",
        "training_data": "released checkpoint trained on full official SoccerNet-v2 training split",
        "evaluation_subset": "54 games in the aligned visual/text/label validation intersection",
        "temporal_access": "offline 120-s windows with an 80-frame receptive field",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "runtime_seconds": time.perf_counter() - started,
    }
    for tolerance in (5.0, 10.0, 20.0):
        aps = []
        for klass in range(len(CLASSES)):
            ap, _ = average_precision(all_predictions[klass], all_truth[klass], tolerance)
            if not np.isnan(ap):
                aps.append(ap)
        metrics[f"event_mAP@{int(tolerance)}s"] = float(np.mean(aps))

    official = official_evaluate(
        str(labels_root), str(predictions_root), prediction_file="results_spotting.json",
        split="valid", version=2, framerate=2, metric="loose", dataset=None,
        EVENT_DICTIONARY={name: index for index, name in enumerate(CLASSES)},
    )
    metrics["official_loose"] = {
        key: value.tolist() if hasattr(value, "tolist") else value for key, value in official.items()
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "calf_reference_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
