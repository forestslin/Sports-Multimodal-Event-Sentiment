"""Assemble the final, manuscript-facing evidence table from frozen runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tgre_experiment import ARTIFACTS, CLASSES


METRICS = ["event_mAP@5s", "event_mAP@10s", "event_mAP@20s", "onset_MAE@10s"]
T975_DF4 = 2.7764451051977987


def load(name: str):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def summarize(rows, key="model"):
    output = {}
    for group in sorted({str(row[key]) for row in rows}):
        selected = [row for row in rows if str(row[key]) == group]
        record = {"n": len(selected), "seeds": [int(row["seed"]) for row in selected if "seed" in row]}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected if metric in row]
            if values:
                record[metric] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "values": values}
        record["parameter_count"] = int(selected[0]["parameter_count"]) if "parameter_count" in selected[0] else None
        output[group] = record
    return output


def paired(main_rows, left, right):
    lookup = {(row["model"], int(row["seed"])): row for row in main_rows}
    seeds = sorted({int(row["seed"]) for row in main_rows if row["model"] == left} & {int(row["seed"]) for row in main_rows if row["model"] == right})
    values = [lookup[(left, seed)]["event_mAP@10s"] - lookup[(right, seed)]["event_mAP@10s"] for seed in seeds]
    mean = float(np.mean(values)); sd = float(np.std(values, ddof=1)); half = T975_DF4 * sd / math.sqrt(len(values))
    return {"comparison": f"{left} minus {right}", "seeds": seeds, "differences": values, "mean": mean, "sd": sd, "descriptive_t_ci95": [mean-half, mean+half]}


def main():
    main_rows = load("experiment_results.json")
    controls = load("text_control_results.json")
    symmetric = load("symmetric_text_results.json")
    tfidf = load("tfidf_results.json")
    nms = load("nms_sensitivity.json")
    robustness = load("robustness_and_timing.json")
    bootstrap = load("game_cluster_bootstrap.json")
    bootstrap_sensitivity = load("game_cluster_bootstrap_sensitivity.json")
    selection = load("selection_audit.json")
    coverage = load("text_coverage_audit.json")

    main_summary = summarize(main_rows)
    control_summary = summarize(controls)
    symmetric_summary = summarize(symmetric, key="configuration")
    tfidf_summary = summarize(tfidf, key="configuration")
    nms_summary = {}
    for model in ("visual", "late"):
        nms_summary[model] = {}
        for radius in (4.0, 8.0, 12.0):
            values = [r["event_mAP@10s"] for r in nms if r["model"] == model and float(r["nms_radius_seconds"]) == radius]
            nms_summary[model][str(int(radius))] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)), "values": values}

    late_rows = [r for r in main_rows if r["model"] == "late"]
    delays = {}
    for metric in ("event_mAP@10s", "event_mAP@10s_text_delay_10s", "event_mAP@10s_text_delay_20s"):
        values = [r[metric] for r in late_rows]
        delays[metric] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)), "values": values}

    class_effects = []
    for klass in CLASSES:
        metric = f"class_AP@10s/{klass}"
        late = np.array([r[metric] for r in late_rows])
        visual = np.array([r[metric] for r in main_rows if r["model"] == "visual"])
        class_effects.append({"class": klass, "late_minus_visual": float(np.mean(late-visual)), "late_mean": float(np.mean(late)), "visual_mean": float(np.mean(visual))})
    class_effects.sort(key=lambda r: r["late_minus_visual"], reverse=True)

    robust_rows = robustness["results"]
    timing = {}
    zero_inference = {}
    for model in sorted({r["model"] for r in robust_rows}):
        selected = [r for r in robust_rows if r["model"] == model]
        timing[model] = {
            "median_bins_per_second": float(np.median([r["bins_per_second"] for r in selected])),
            "median_milliseconds_per_bin": float(np.median([r["milliseconds_per_bin"] for r in selected])),
        }
        values = [r["event_mAP@10s_zero_text"] for r in selected if "event_mAP@10s_zero_text" in r]
        if values:
            zero_inference[model] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)), "values": values}

    payload = {
        "metric_note": "event mAP within a symmetric +/- matching radius; this custom diagnostic differs from SoccerNet's official delta full-window convention",
        "main_models": main_summary,
        "paired_seed_differences": [paired(main_rows, left, right) for left, right in [("late","visual"),("rtgre","visual"),("rtgre_flow","visual"),("tcn_visual","visual"),("tcn_fusion","visual"),("rtgre","late")]],
        "paired_game_cluster_bootstrap": bootstrap,
        "game_cluster_bootstrap_sensitivity": bootstrap_sensitivity,
        "text_controls": control_summary,
        "future_text_control": symmetric_summary,
        "tfidf_sensitivity": tfidf_summary,
        "late_text_delay": delays,
        "nms_sensitivity": nms_summary,
        "zero_text_inference": zero_inference,
        "timing": timing,
        "timing_environment": robustness["environment"],
        "class_effects_late_minus_visual": class_effects,
        "selection_audit": selection,
        "text_coverage": coverage,
    }
    official = {}
    root = ARTIFACTS / "official_crosscheck"
    if root.exists():
        for path in root.glob("*/official_metrics.json"):
            official[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    payload["official_crosschecks"] = official
    (ARTIFACTS / "final_revision_evidence.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    late = main_summary["late"]["event_mAP@10s"]
    visual = main_summary["visual"]["event_mAP@10s"]
    future = next(iter(symmetric_summary.values()))["event_mAP@10s"]
    lines = [
        "# Final revision evidence", "",
        f"- Visual GRU: {visual['mean']:.4f} +/- {visual['sd']:.4f} event mAP within +/-10 s.",
        f"- Late fusion: {late['mean']:.4f} +/- {late['sd']:.4f}; mean paired gain {late['mean']-visual['mean']:+.4f}.",
        f"- Symmetric future-text control: {future['mean']:.4f} +/- {future['sd']:.4f}; this quantifies optimistic leakage when future commentary is admitted.",
        f"- Zero-text retraining: {control_summary['late_zero']['event_mAP@10s']['mean']:.4f}; 60-s text shift: {control_summary['late_shift60']['event_mAP@10s']['mean']:.4f}.",
        f"- TF-IDF late fusion: {next(iter(tfidf_summary.values()))['event_mAP@10s']['mean']:.4f}.",
        "- The paired game-cluster bootstrap supports the modest late-fusion gain, while the five-seed t interval remains wider and should be reported as training-instability uncertainty.",
        "- Residual graph message passing and latent-flow regularization do not improve upon simple late fusion.",
    ]
    (ARTIFACTS / "final_revision_evidence.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"json": str(ARTIFACTS / 'final_revision_evidence.json'), "markdown": str(ARTIFACTS / 'final_revision_evidence.md')}, indent=2))


if __name__ == "__main__":
    main()
