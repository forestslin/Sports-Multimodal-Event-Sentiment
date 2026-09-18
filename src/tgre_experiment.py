"""Reproducible real-data experiment for the revised PeerJ manuscript.

Task: temporal spotting of 17 SoccerNet-v2 event classes using public 2-fps
visual descriptors and time-stamped SoccerNet-Echoes English ASR text.  The
text representation at time t uses only segments that ended no later than t.
No raw video, test labels, or synthetic labels are used.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.ipc as ipc
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
LABEL_ROOT = ROOT / "data" / "SN-Labels"
FEATURE_ROOT = ROOT.parent / "f"
ECHO_PATH = ROOT / "data" / "SN-echoes" / "whisper_v1_en" / "1.0.0" / "soccer_net_echoes_hf_dataset-train.arrow"
ARTIFACTS = ROOT / "artifacts_causal_v2"
BIN_SECONDS = 2.0
CONTEXT = 5  # Six causal bins: the current bin plus 10 seconds of history.
TEXT_LOOKBACK = 20.0
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

CLASSES = [
    "Ball out of play", "Clearance", "Corner", "Direct free-kick", "Foul",
    "Goal", "Indirect free-kick", "Kick-off", "Offside", "Penalty", "Red card",
    "Shots off target", "Shots on target", "Substitution", "Throw-in",
    "Yellow card", "Yellow->red card",
]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(max(1, min(8, (Path.cwd().stat().st_size if False else 8))))


def game_path(root: Path, game: str) -> Path:
    return root.joinpath(*game.split("\\"))


def parse_game_time(value: str) -> tuple[int, float] | None:
    match = re.search(r"(\d+)\s*-\s*(\d+):(\d+)", value)
    if not match:
        return None
    half, minute, second = map(int, match.groups())
    if half not in (1, 2):
        return None
    return half, 60.0 * minute + second


def read_labels(game: str) -> dict[int, list[tuple[float, int]]]:
    path = game_path(LABEL_ROOT, game) / "Labels-v2.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, list[tuple[float, int]]] = {1: [], 2: []}
    for row in data.get("annotations", []):
        parsed = parse_game_time(row.get("gameTime", ""))
        label = row.get("label")
        if parsed is None or label not in CLASS_INDEX:
            continue
        half, seconds = parsed
        result[half].append((seconds, CLASS_INDEX[label]))
    return result


def read_echoes() -> dict[tuple[str, int], list[tuple[float, float, str]]]:
    with ECHO_PATH.open("rb") as handle:
        rows = ipc.open_stream(handle).read_all().to_pylist()
    out: dict[tuple[str, int], list[tuple[float, float, str]]] = defaultdict(list)
    for row in rows:
        game, half_str = row["game"].rsplit("/", 1)
        if half_str not in {"1", "2"}:
            continue
        out[(game.replace("/", "\\"), int(half_str))].append(
            (float(row["start_time"]), float(row["end_time"]), row["text"])
        )
    for key in out:
        out[key].sort(key=lambda item: item[1])
    return out


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def text_terms(text: str, include_bigrams: bool = False) -> list[str]:
    words = tokens(text)
    if not include_bigrams:
        return words
    return words + [f"{left}__{right}" for left, right in zip(words, words[1:])]


def build_vocabulary(
    echoes: dict[tuple[str, int], list[tuple[float, float, str]]],
    train_games: Iterable[str],
    size: int,
    include_bigrams: bool = False,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    allowed = set(train_games)
    for (game, _), segments in echoes.items():
        if game in allowed:
            for _, _, text in segments:
                counts.update(text_terms(text, include_bigrams=include_bigrams))
    terms = [term for term, count in counts.most_common() if count >= 5][:size]
    return {term: idx for idx, term in enumerate(terms)}


def build_idf(
    echoes: dict[tuple[str, int], list[tuple[float, float, str]]],
    train_games: Iterable[str],
    vocab: dict[str, int],
    include_bigrams: bool,
) -> np.ndarray:
    allowed = set(train_games)
    document_frequency = np.zeros(len(vocab), dtype=np.float64)
    documents = 0
    for (game, _), segments in echoes.items():
        if game not in allowed:
            continue
        for _, _, text in segments:
            documents += 1
            ids = {vocab[term] for term in text_terms(text, include_bigrams) if term in vocab}
            for token_id in ids:
                document_frequency[token_id] += 1
    return (np.log((1.0 + documents) / (1.0 + document_frequency)) + 1.0).astype(np.float32)


def read_visual(game: str, half: int) -> np.ndarray:
    path = game_path(FEATURE_ROOT, game) / f"{half}_ResNET_TF2_PCA512.npy"
    values = np.load(path).astype(np.float32, copy=False)
    usable = values.shape[0] - values.shape[0] % 4  # 2 fps -> 2-second bin
    if usable == 0:
        raise ValueError(f"empty feature file: {path}")
    return values[:usable].reshape(-1, 4, values.shape[1]).mean(axis=1)


def observation_time(index: int | np.ndarray) -> float | np.ndarray:
    """Right edge of a completed two-second visual bin.

    Public 2-fps descriptors are sampled within the half. Four consecutive
    descriptors are pooled into [2k, 2k+2], and the corresponding prediction
    is available only at the right edge, 2(k+1) seconds.
    """
    return (np.asarray(index) + 1) * BIN_SECONDS


def causal_text_matrix(
    segments: list[tuple[float, float, str]],
    n_bins: int,
    vocab: dict[str, int],
    *,
    include_bigrams: bool = False,
    idf: np.ndarray | None = None,
    past_lookback: float = TEXT_LOOKBACK,
    future_access: float = 0.0,
) -> np.ndarray:
    matrix = np.zeros((n_bins, len(vocab)), dtype=np.float32)
    active: deque[tuple[float, list[int]]] = deque()
    ptr = 0
    for index in range(n_bins):
        time = float(observation_time(index))
        while ptr < len(segments) and segments[ptr][1] <= time + future_access:
            end = segments[ptr][1]
            ids = [vocab[term] for term in text_terms(segments[ptr][2], include_bigrams) if term in vocab]
            active.append((end, ids))
            ptr += 1
        while active and active[0][0] <= time - past_lookback:
            active.popleft()
        for _, ids in active:
            for token_id in ids:
                matrix[index, token_id] += 1.0
    matrix = np.log1p(matrix)
    if idf is not None:
        matrix *= idf[None, :]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return matrix


def targets_for_half(n_bins: int, events: list[tuple[float, int]]) -> tuple[np.ndarray, list[tuple[float, int]]]:
    targets = np.zeros((n_bins, len(CLASSES)), dtype=np.float32)
    retained: list[tuple[float, int]] = []
    for seconds, klass in events:
        # Assign an event to the first completed visual bin whose right edge is
        # at or after the event timestamp. This prevents a target at time s
        # from being paired with visual descriptors sampled after the stated
        # prediction time.
        index = max(0, int(math.ceil(seconds / BIN_SECONDS)) - 1)
        if 0 <= index < n_bins:
            targets[index, klass] = 1.0
            retained.append((seconds, klass))
    return targets, retained


def windowed(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    padded = np.pad(values, ((CONTEXT, 0), (0, 0)), mode="edge")
    offsets = np.arange(CONTEXT + 1)
    return padded[indices[:, None] + offsets]


@dataclass
class HalfData:
    visual: np.ndarray
    text: np.ndarray
    targets: np.ndarray
    events: list[tuple[float, int]]


class Corpus:
    def __init__(
        self,
        coverage_path: Path,
        vocab_size: int = 256,
        text_mode: str = "counts",
        text_access: str = "causal",
    ):
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))["splits"]
        self.games = {split: data["games"] for split, data in coverage.items()}
        # A deterministic competition/season-balanced development split is
        # used only for choosing the training budget. Formal runs use all
        # official training games and are evaluated on the official validation
        # intersection after the budget is frozen.
        grouped: dict[str, list[str]] = defaultdict(list)
        for game in self.games["train"]:
            parts = game.split("\\")
            grouped["\\".join(parts[:2])].append(game)
        fit, dev = [], []
        for games in grouped.values():
            for index, game in enumerate(sorted(games)):
                (dev if index % 5 == 0 else fit).append(game)
        self.games["fit"] = sorted(fit)
        self.games["dev"] = sorted(dev)
        self.echoes = read_echoes()
        if text_mode not in {"counts", "tfidf"}:
            raise ValueError(f"unsupported text mode: {text_mode}")
        self.text_mode = text_mode
        if text_access not in {"causal", "symmetric10"}:
            raise ValueError(f"unsupported text access: {text_access}")
        self.text_access = text_access
        self.include_bigrams = text_mode == "tfidf"
        self.vocab = build_vocabulary(self.echoes, self.games["train"], vocab_size, self.include_bigrams)
        self.idf = build_idf(self.echoes, self.games["train"], self.vocab, self.include_bigrams) if text_mode == "tfidf" else None
        self._cache: dict[tuple[str, int], HalfData] = {}

    def half(self, game: str, half: int) -> HalfData:
        key = (game, half)
        if key in self._cache:
            return self._cache[key]
        visual = read_visual(game, half)
        text = causal_text_matrix(
            self.echoes.get((game, half), []), len(visual), self.vocab,
            include_bigrams=self.include_bigrams, idf=self.idf,
            past_lookback=10.0 if self.text_access == "symmetric10" else TEXT_LOOKBACK,
            future_access=10.0 if self.text_access == "symmetric10" else 0.0,
        )
        labels = read_labels(game)[half]
        target, retained = targets_for_half(len(visual), labels)
        result = HalfData(visual, text, target, retained)
        self._cache[key] = result
        return result

    def sampled_batches(
        self,
        split: str,
        rng: np.random.Generator,
        positives_per_half: int = 16,
        negative_ratio: int = 4,
        text_control: str = "observed",
    ):
        games = list(self.games[split])
        rng.shuffle(games)
        for game in games:
            for half in (1, 2):
                data = self.half(game, half)
                positive = np.where(data.targets.max(axis=1) > 0)[0]
                negative = np.where(data.targets.max(axis=1) == 0)[0]
                if len(positive) == 0 or len(negative) == 0:
                    continue
                chosen_pos = rng.choice(positive, size=min(len(positive), positives_per_half), replace=False)
                chosen_neg = rng.choice(negative, size=min(len(negative), len(chosen_pos) * negative_ratio), replace=False)
                indices = np.concatenate([chosen_pos, chosen_neg])
                rng.shuffle(indices)
                text_values = data.text
                if text_control == "zero":
                    text_values = np.zeros_like(data.text)
                elif text_control == "shift60":
                    shift = int(round(60.0 / BIN_SECONDS))
                    text_values = np.zeros_like(data.text)
                    text_values[shift:] = data.text[:-shift]
                elif text_control != "observed":
                    raise ValueError(text_control)
                yield windowed(data.visual, indices), windowed(text_values, indices), data.targets[indices]


class SingleStream(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, len(CLASSES))

    def forward(self, visual: torch.Tensor, text: torch.Tensor):
        values = visual if visual.shape[-1] == self.proj.in_features else text
        state, _ = self.gru(F.gelu(self.proj(values)))
        return self.head(state[:, -1]), state.new_zeros(())


class LateFusion(nn.Module):
    def __init__(self, visual_dim: int, text_dim: int, hidden: int = 96):
        super().__init__()
        self.vproj = nn.Linear(visual_dim, hidden)
        self.tproj = nn.Linear(text_dim, hidden)
        self.gru = nn.GRU(2 * hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, len(CLASSES))

    def forward(self, visual: torch.Tensor, text: torch.Tensor):
        state, _ = self.gru(torch.cat([F.gelu(self.vproj(visual)), F.gelu(self.tproj(text))], dim=-1))
        return self.head(state[:, -1]), state.new_zeros(())


class CausalTCN(nn.Module):
    """Capacity-matched causal temporal-convolution baseline."""

    def __init__(self, visual_dim: int, text_dim: int | None = None, hidden: int = 96):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        input_dim = visual_dim if text_dim is None else visual_dim + text_dim
        self.proj = nn.Linear(input_dim, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=3)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3)
        self.head = nn.Linear(hidden, len(CLASSES))

    def forward(self, visual: torch.Tensor, text: torch.Tensor):
        values = visual if self.text_dim is None else torch.cat([visual, text], dim=-1)
        state = F.gelu(self.proj(values)).transpose(1, 2)
        residual = state
        state = F.gelu(self.conv1(F.pad(state, (2, 0))))
        state = self.conv2(F.pad(state, (2, 0)))
        state = F.gelu(state + residual)
        return self.head(state[:, :, -1]), state.new_zeros(())


class TGRE(nn.Module):
    """Two-node temporal graph encoder with an optional latent-flow penalty."""
    def __init__(self, visual_dim: int, text_dim: int, hidden: int = 96):
        super().__init__()
        self.vproj = nn.Linear(visual_dim, hidden)
        self.tproj = nn.Linear(text_dim, hidden)
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.gru = nn.GRU(2 * hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, len(CLASSES))
        self.scale = hidden ** -0.5

    def forward(self, visual: torch.Tensor, text: torch.Tensor):
        nodes = torch.stack([F.gelu(self.vproj(visual)), F.gelu(self.tproj(text))], dim=2)
        weights = torch.softmax((self.q(nodes) @ self.k(nodes).transpose(-1, -2)) * self.scale, dim=-1)
        graph = weights @ self.v(nodes)
        state, _ = self.gru(graph.flatten(2))
        flow = (state[:, 1:] - state[:, :-1]).square().mean()
        return self.head(state[:, -1]), flow


class ResidualTGRE(TGRE):
    """TGRE that preserves unimodal evidence through a learned residual gate."""

    def __init__(self, visual_dim: int, text_dim: int, hidden: int = 96):
        super().__init__(visual_dim, text_dim, hidden)
        self.gate = nn.Parameter(torch.full((2, hidden), -2.0))

    def forward(self, visual: torch.Tensor, text: torch.Tensor):
        nodes = torch.stack([F.gelu(self.vproj(visual)), F.gelu(self.tproj(text))], dim=2)
        weights = torch.softmax((self.q(nodes) @ self.k(nodes).transpose(-1, -2)) * self.scale, dim=-1)
        message = weights @ self.v(nodes)
        graph = nodes + torch.sigmoid(self.gate)[None, None] * message
        state, _ = self.gru(graph.flatten(2))
        flow = (state[:, 1:] - state[:, :-1]).square().mean()
        return self.head(state[:, -1]), flow


def make_model(name: str, visual_dim: int, text_dim: int) -> nn.Module:
    if name == "visual":
        return SingleStream(visual_dim)
    if name == "text":
        return SingleStream(text_dim)
    if name in {"late", "late_zero", "late_shift60"}:
        return LateFusion(visual_dim, text_dim)
    if name == "tcn_visual":
        return CausalTCN(visual_dim)
    if name == "tcn_fusion":
        return CausalTCN(visual_dim, text_dim)
    if name in {"tgre", "tgre_flow"}:
        return TGRE(visual_dim, text_dim)
    if name in {"rtgre", "rtgre_flow", "rtgre_zero", "rtgre_shift60"}:
        return ResidualTGRE(visual_dim, text_dim)
    raise ValueError(name)


@torch.no_grad()
def score_half(
    model: nn.Module,
    data: HalfData,
    device: torch.device,
    batch_size: int = 512,
    text_delay_bins: int = 0,
    zero_text: bool = False,
) -> np.ndarray:
    model.eval()
    values = []
    all_indices = np.arange(len(data.visual))
    for start in range(0, len(all_indices), batch_size):
        indices = all_indices[start : start + batch_size]
        visual = torch.from_numpy(windowed(data.visual, indices)).to(device)
        text_values = data.text
        if zero_text:
            text_values = np.zeros_like(data.text)
        elif text_delay_bins > 0:
            text_values = np.zeros_like(data.text)
            text_values[text_delay_bins:] = data.text[:-text_delay_bins]
        text = torch.from_numpy(windowed(text_values, indices)).to(device)
        logits, _ = model(visual, text)
        values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values)


def nms_predictions(scores: np.ndarray, seconds: float = 8.0) -> dict[int, list[tuple[float, float]]]:
    result: dict[int, list[tuple[float, float]]] = {}
    radius_bins = int(math.floor(seconds / BIN_SECONDS + 1e-9))
    for klass in range(scores.shape[1]):
        order = np.argsort(-scores[:, klass])
        kept: list[tuple[float, float]] = []
        suppressed = np.zeros(scores.shape[0], dtype=bool)
        for index in order:
            if suppressed[index]:
                continue
            kept.append((float(observation_time(index)), float(scores[index, klass])))
            left = max(0, int(index) - radius_bins)
            right = min(scores.shape[0], int(index) + radius_bins + 1)
            suppressed[left:right] = True
        result[klass] = kept
    return result


def average_precision(predicted: list[tuple[float, float]], truth: list[float], tolerance: float) -> tuple[float, list[float]]:
    if not truth:
        return float("nan"), []
    truth = sorted(truth)
    used = np.zeros(len(truth), dtype=bool)
    hit = []
    errors: list[float] = []
    for time, _ in sorted(predicted, key=lambda item: -item[1]):
        left = bisect_left(truth, time - tolerance)
        right = bisect_right(truth, time + tolerance)
        candidates = [(abs(time - truth[idx]), idx) for idx in range(left, right) if not used[idx]]
        if candidates:
            error, idx = min(candidates)
            used[idx] = True
            hit.append(1.0)
            errors.append(error)
        else:
            hit.append(0.0)
    if not hit:
        return 0.0, errors
    hit_array = np.asarray(hit)
    precision = np.cumsum(hit_array) / np.arange(1, len(hit_array) + 1)
    recall = np.cumsum(hit_array) / len(truth)
    # Interpolated AP at each true-positive recall step.
    ap = sum(precision[i] for i, value in enumerate(hit_array) if value) / len(truth)
    return float(ap), errors


def evaluate(
    model: nn.Module,
    corpus: Corpus,
    device: torch.device,
    split: str = "valid",
    text_delay_seconds: float = 0.0,
    zero_text: bool = False,
    nms_seconds: float = 8.0,
) -> dict[str, float]:
    all_predictions: dict[int, list[tuple[float, float]]] = defaultdict(list)
    all_truth: dict[int, list[float]] = defaultdict(list)
    # Offset each half/game to avoid accidental cross-game matching.
    offset = 0.0
    for game in corpus.games[split]:
        for half in (1, 2):
            data = corpus.half(game, half)
            scores = score_half(
                model,
                data,
                device,
                text_delay_bins=int(round(text_delay_seconds / BIN_SECONDS)),
                zero_text=zero_text,
            )
            for klass, entries in nms_predictions(scores, seconds=nms_seconds).items():
                all_predictions[klass].extend((offset + t, score) for t, score in entries)
            for time, klass in data.events:
                all_truth[klass].append(offset + time)
            offset += 10000.0
    summary: dict[str, float] = {}
    errors_at_10: list[float] = []
    for tolerance in (5.0, 10.0, 20.0):
        aps = []
        for klass in range(len(CLASSES)):
            ap, errors = average_precision(all_predictions[klass], all_truth[klass], tolerance)
            if not np.isnan(ap):
                aps.append(ap)
            if tolerance == 10.0:
                errors_at_10.extend(errors)
                summary[f"class_AP@10s/{CLASSES[klass]}"] = ap
        summary[f"event_mAP@{int(tolerance)}s"] = float(np.mean(aps))
    summary["onset_MAE@10s"] = float(np.mean(errors_at_10)) if errors_at_10 else float("nan")
    summary["valid_events"] = float(sum(map(len, all_truth.values())))
    summary["nms_radius_seconds"] = float(nms_seconds)
    return summary


def train(
    model_name: str,
    seed: int,
    epochs: int = 8,
    flow_weight: float = 0.02,
    corpus: Corpus | None = None,
    train_split: str = "train",
    eval_split: str = "valid",
    eval_epochs: set[int] | None = None,
    requested_device: str = "auto",
) -> dict[str, float]:
    set_seed(seed)
    coverage = ARTIFACTS / "modality_coverage.json"
    corpus = corpus or Corpus(coverage)
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch build cannot access CUDA")
    print(json.dumps({"runtime_device": str(device), "torch_version": torch.__version__}), flush=True)
    model = make_model(model_name, 512, len(corpus.vocab)).to(device)
    started = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    final_metrics: dict[str, float] | None = None
    losses_by_epoch: list[float] = []
    evaluation_history: list[dict[str, float]] = []
    text_control = (
        "zero" if model_name in {"late_zero", "rtgre_zero"}
        else "shift60" if model_name in {"late_shift60", "rtgre_shift60"}
        else "observed"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for visual, text, target in corpus.sampled_batches(train_split, rng, text_control=text_control):
            visual_t = torch.from_numpy(visual).to(device)
            text_t = torch.from_numpy(text).to(device)
            target_t = torch.from_numpy(target).to(device)
            logits, flow = model(visual_t, text_t)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                target_t,
                pos_weight=torch.full((len(CLASSES),), 8.0, device=device),
            )
            if model_name in {"tgre_flow", "rtgre_flow"}:
                loss = loss + flow_weight * flow
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_loss = float(np.mean(losses))
        losses_by_epoch.append(epoch_loss)
        print(json.dumps({"model": model_name, "seed": seed, "epoch": epoch, "training_loss": epoch_loss}), flush=True)
        if eval_epochs and epoch in eval_epochs:
            checkpoint_metrics = evaluate(
                model, corpus, device, split=eval_split,
                zero_text=text_control == "zero",
                text_delay_seconds=60.0 if text_control == "shift60" else 0.0,
            )
            checkpoint_metrics.update({"epoch": float(epoch), "training_loss": epoch_loss})
            evaluation_history.append(checkpoint_metrics)
            print(json.dumps({"model": model_name, "seed": seed, "evaluation": checkpoint_metrics}), flush=True)
    if evaluation_history and int(evaluation_history[-1]["epoch"]) == epochs:
        final_metrics = evaluation_history[-1].copy()
    else:
        final_metrics = evaluate(
            model, corpus, device, split=eval_split,
            zero_text=text_control == "zero",
            text_delay_seconds=60.0 if text_control == "shift60" else 0.0,
        )
    final_metrics["epoch"] = float(epochs)
    final_metrics["training_loss"] = losses_by_epoch[-1]
    final_metrics["training_loss_history"] = losses_by_epoch
    final_metrics["text_control"] = text_control
    if evaluation_history:
        final_metrics["evaluation_history"] = evaluation_history
    if model_name in {"late", "rtgre", "rtgre_flow", "tcn_fusion"}:
        for delay in (10.0, 20.0):
            delayed = evaluate(model, corpus, device, split=eval_split, text_delay_seconds=delay)
            final_metrics[f"event_mAP@10s_text_delay_{int(delay)}s"] = delayed["event_mAP@10s"]
    final_metrics["parameter_count"] = float(sum(p.numel() for p in model.parameters()))
    final_metrics["runtime_seconds"] = float(time.perf_counter() - started)
    final_metrics["runtime_device"] = str(device)
    final_metrics["torch_version"] = torch.__version__
    checkpoint_dir = ARTIFACTS / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    suffix = []
    if corpus.text_mode != "counts":
        suffix.append(f"{corpus.text_mode}{len(corpus.vocab)}")
    if corpus.text_access != "causal":
        suffix.append(corpus.text_access)
    checkpoint_tag = model_name if not suffix else f"{model_name}_{'_'.join(suffix)}"
    torch.save(
        {
            "model_name": model_name,
            "seed": seed,
            "epochs": epochs,
            "train_split": train_split,
            "eval_split": eval_split,
            "vocabulary": corpus.vocab,
            "text_mode": corpus.text_mode,
            "text_access": corpus.text_access,
            "idf": corpus.idf,
            "time_anchor": "right edge of completed 2-second visual bin",
            "prediction_time_seconds": "2*(bin_index+1)",
            "state_dict": model.state_dict(),
        },
        checkpoint_dir / f"{checkpoint_tag}_seed{seed}_epoch{epochs}.pt",
    )
    final_metrics.update({"model": model_name, "configuration": checkpoint_tag, "text_mode": corpus.text_mode, "text_access": corpus.text_access, "seed": float(seed), "vocabulary_size": float(len(corpus.vocab)), "epochs": float(epochs), "train_split": train_split, "eval_split": eval_split})
    return final_metrics
