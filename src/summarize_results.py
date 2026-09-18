"""Create submission tables from the completed repeated-seed experiment JSON."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts_causal_v2"
rows = json.loads((ART / "experiment_results.json").read_text(encoding="utf-8"))

metrics = ["event_mAP@5s", "event_mAP@10s", "event_mAP@20s", "onset_MAE@10s"]
models = sorted({row["model"] for row in rows})
summary = []
for model in models:
    group = [row for row in rows if row["model"] == model]
    record = {"model": model, "runs": len(group), "seeds": [int(row["seed"]) for row in group]}
    for metric in metrics:
        values = np.asarray([row[metric] for row in group], dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        half = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))) if len(values) > 1 else float("nan")
        record[metric] = {"mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}
    record["parameter_count"] = int(group[0]["parameter_count"])
    record["runtime_seconds_mean"] = float(np.mean([row["runtime_seconds"] for row in group]))
    for delay in (10, 20):
        key = f"event_mAP@10s_text_delay_{delay}s"
        if key in group[0]:
            values = np.asarray([row[key] for row in group], dtype=float)
            record[key] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
    summary.append(record)

comparisons = []
by_model_seed = {(row["model"], int(row["seed"])): row for row in rows}
for model in ["late", "rtgre", "rtgre_flow", "tcn_visual", "tcn_fusion"]:
    shared = sorted(set(int(row["seed"]) for row in rows if row["model"] == model) & set(int(row["seed"]) for row in rows if row["model"] == "visual"))
    diffs = np.asarray([by_model_seed[(model, seed)]["event_mAP@10s"] - by_model_seed[("visual", seed)]["event_mAP@10s"] for seed in shared])
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan")
    half = float(t.ppf(0.975, len(diffs) - 1) * sd / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
    comparisons.append({"comparison": f"{model} minus visual", "n_paired_seeds": len(shared), "mean_difference": mean, "sd_difference": sd, "ci95_low": mean-half, "ci95_high": mean+half})

payload = {"model_summary": summary, "paired_seed_comparisons": comparisons}
(ART / "acceptance_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "# Repeated seed experiment summary",
    "",
    f"Values are mean ± SD across independent training seeds. The 95% confidence interval uses a t interval over seed-level estimates and is descriptive with n = {max(record['runs'] for record in summary)}.",
    "",
    "| Model | mAP@5 s | mAP@10 s | mAP@20 s | Onset MAE@10 s | Parameters |",
    "|---|---:|---:|---:|---:|---:|",
]
for record in summary:
    fields = []
    for metric in metrics:
        value = record[metric]
        fields.append(f"{value['mean']:.3f} ± {value['sd']:.3f}")
    lines.append(f"| {record['model']} | " + " | ".join(fields) + f" | {record['parameter_count']:,} |")
lines.extend(["", "## Paired differences in mAP@10 s", "", "| Comparison | Mean difference | 95% CI |", "|---|---:|---:|"])
for row in comparisons:
    lines.append(f"| {row['comparison']} | {row['mean_difference']:+.3f} | [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] |")
(ART / "acceptance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
