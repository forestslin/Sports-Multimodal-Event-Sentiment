"""Create the three submission figures and traceable source-data tables.

Figure contract
---------------
Figure 1 (schematic-led): the implemented time-bounded pipeline maps public
inputs to completed observation bins, the compared models, and event scores.
Figure 2 (quantitative): repeated-seed performance and paired differences show
the comparative result together with its uncertainty.
Figure 3 (quantitative): timing perturbations and class-wise AP expose where
text/graph fusion is robust and where the conclusion is limited.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "mplconfig"))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ART = ROOT / "artifacts_causal_v2"
OUT = ROOT / "figures_causal_v2"
SRC = OUT / "source_data"

COLORS = {
    "visual": "#4C78A8",
    "text": "#E39C37",
    "late": "#72B7B2",
    "rtgre": "#7B6FD0",
    "rtgre_flow": "#B279A2",
    "tcn_visual": "#59A14F",
    "tcn_fusion": "#F28E2B",
    "neutral": "#6B7280",
    "light": "#EEF2F6",
    "dark": "#25313C",
}
DISPLAY = {
    "visual": "Visual GRU",
    "text": "Text GRU",
    "late": "Late fusion",
    "rtgre": "Residual TGRE",
    "rtgre_flow": "TGRE + smoothness",
    "tcn_visual": "Causal TCN (visual)",
    "tcn_fusion": "Causal TCN (fusion)",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, width, height, text, face, edge=None, fontsize=7, weight="normal"):
    edge = edge or face
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, color=COLORS["dark"])
    return patch


def arrow(ax, start, end, color=None, style="-|>", dashed=False, curve=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=8,
        linewidth=1.0,
        color=color or COLORS["neutral"],
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(patch)


def figure1() -> None:
    fig, ax = plt.subplots(figsize=(7.08, 4.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.985, "a", weight="bold", fontsize=8, va="top")
    box(ax, (0.04, 0.70), 0.19, 0.14, "ResNet PCA-512\nvisual descriptors\n2 fps", "#DDEAF6", COLORS["visual"])
    box(ax, (0.04, 0.47), 0.19, 0.14, "Whisper-v1\nEnglish-normalized ASR\n(start, end, text)", "#FBE8C9", COLORS["text"], fontsize=6.5)
    box(ax, (0.04, 0.24), 0.19, 0.14, "SoccerNet-v2\n17-class event\ntimestamps", "#E6E8EB", COLORS["neutral"])

    ax.text(0.285, 0.985, "b", weight="bold", fontsize=8, va="top")
    box(ax, (0.29, 0.67), 0.21, 0.18, "Average four descriptors\nper completed 2-s bin\nstamp at right edge", "#DDEAF6", COLORS["visual"])
    box(ax, (0.29, 0.43), 0.21, 0.18, "256-term train vocabulary\n20-s log-count window\nASR end time ≤ t", "#FBE8C9", COLORS["text"])
    box(ax, (0.29, 0.20), 0.21, 0.14, "Multi-label target\nfor each 2-s bin", "#E6E8EB", COLORS["neutral"])
    arrow(ax, (0.23, 0.77), (0.29, 0.77), COLORS["visual"])
    arrow(ax, (0.23, 0.54), (0.29, 0.54), COLORS["text"])
    arrow(ax, (0.23, 0.31), (0.29, 0.27), COLORS["neutral"])

    ax.text(0.555, 0.985, "c", weight="bold", fontsize=8, va="top")
    ax.text(0.66, 0.85, "Mutually alternative models", ha="center", fontsize=6.5, weight="bold", color=COLORS["dark"])
    box(ax, (0.57, 0.62), 0.18, 0.13, "Visual/Text GRU\nLate-fusion GRU\nVisual/Fusion TCN", "#E7E4F6", COLORS["rtgre"], fontsize=6.0)
    box(ax, (0.57, 0.39), 0.18, 0.15, "Residual TGRE\n2 × 2 modality attention\n+ residual message gates", "#F4F1FB", COLORS["rtgre"], fontsize=5.9)
    arrow(ax, (0.50, 0.76), (0.57, 0.70), COLORS["visual"])
    arrow(ax, (0.50, 0.52), (0.57, 0.68), COLORS["text"])
    arrow(ax, (0.50, 0.73), (0.57, 0.48), COLORS["visual"])
    arrow(ax, (0.50, 0.49), (0.57, 0.45), COLORS["text"])
    box(ax, (0.57, 0.10), 0.18, 0.09, "Optional latent-state\nsmoothness penalty", "#F2E5EE", COLORS["rtgre_flow"], fontsize=6.2)
    arrow(ax, (0.66, 0.39), (0.66, 0.19), COLORS["rtgre_flow"], dashed=True)

    ax.text(0.80, 0.985, "d", weight="bold", fontsize=8, va="top")
    box(ax, (0.82, 0.62), 0.15, 0.14, "17 sigmoid\nevent scores\nat time t", "#E7E4F6", COLORS["rtgre"])
    box(ax, (0.82, 0.39), 0.15, 0.14, "Class-wise\n8-s NMS", "#E6E8EB", COLORS["neutral"])
    box(ax, (0.81, 0.13), 0.17, 0.19, "mAP within\n±5/±10/±20 s\nTimestamp MAE (±10 s)\nOfficial cross-check", "#E6E8EB", COLORS["neutral"], fontsize=5.8)
    arrow(ax, (0.75, 0.68), (0.82, 0.69), COLORS["rtgre"])
    arrow(ax, (0.75, 0.46), (0.82, 0.67), COLORS["rtgre"])
    arrow(ax, (0.895, 0.62), (0.895, 0.53), COLORS["neutral"])
    arrow(ax, (0.895, 0.39), (0.895, 0.31), COLORS["neutral"])
    arrow(ax, (0.50, 0.27), (0.82, 0.23), COLORS["neutral"], dashed=True, curve=0.10)

    ax.text(0.52, 0.035, "Model inputs: ASR segment end time ≤ cutoff; future text excluded. Full-half NMS is offline.", ha="center", va="center", fontsize=6.5, color=COLORS["dark"], weight="bold")
    save_figure(fig, "Fig1")


def load_results():
    rows = json.loads((ART / "experiment_results.json").read_text(encoding="utf-8"))
    evidence = json.loads((ART / "final_revision_evidence.json").read_text(encoding="utf-8"))
    return rows, evidence


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def figure2(rows, evidence) -> None:
    order = ["visual", "text", "late", "rtgre", "rtgre_flow", "tcn_visual", "tcn_fusion"]
    summary_by_model = evidence["main_models"]
    paired = {row["comparison"].split()[0]: row for row in evidence["paired_seed_differences"]}
    source_rows = [
        {"model": row["model"], "seed": int(row["seed"]), "mAP_10s": row["event_mAP@10s"]}
        for row in rows
    ]
    write_csv(SRC / "Fig2_source_data.csv", ["model", "seed", "mAP_10s"], source_rows)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.08, 3.55), gridspec_kw={"width_ratios": [1.75, 1]}, constrained_layout=True)
    y_main = np.arange(len(order))
    means = [summary_by_model[m]["event_mAP@10s"]["mean"] for m in order]
    sds = [summary_by_model[m]["event_mAP@10s"]["sd"] for m in order]
    ax.errorbar(means, y_main, xerr=sds, fmt="none", ecolor=COLORS["dark"], elinewidth=1.0, capsize=3, zorder=2)
    for i, model in enumerate(order):
        vals = [r["event_mAP@10s"] for r in rows if r["model"] == model]
        jitter = np.linspace(-0.08, 0.08, len(vals))
        ax.scatter(vals, i + jitter, s=25, color=COLORS[model], edgecolor="white", linewidth=0.5, zorder=3)
        ax.vlines(means[i], i - 0.16, i + 0.16, color=COLORS["dark"], linewidth=1.4, zorder=4)
    ax.set_yticks(y_main, [DISPLAY[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlabel("Event mAP within ±10 s")
    ax.grid(axis="x", color="#D9DEE5", linewidth=0.6, zorder=0)
    ax.text(-0.12, 1.03, "a", transform=ax.transAxes, weight="bold", fontsize=8)
    ax.text(0.0, 1.03, "Seed-level estimates and mean ± SD", transform=ax.transAxes, weight="bold", fontsize=7)

    compare_order = ["late", "rtgre", "rtgre_flow", "tcn_visual", "tcn_fusion"]
    y = np.arange(len(compare_order))
    effects = [paired[m]["mean"] for m in compare_order]
    lower = [effects[i] - paired[m]["descriptive_t_ci95"][0] for i, m in enumerate(compare_order)]
    upper = [paired[m]["descriptive_t_ci95"][1] - effects[i] for i, m in enumerate(compare_order)]
    ax2.axvline(0, color=COLORS["neutral"], linewidth=0.8)
    for i, model in enumerate(compare_order):
        ax2.errorbar(effects[i], i, xerr=[[lower[i]], [upper[i]]], fmt="o", color=COLORS[model], ecolor=COLORS[model], capsize=3, markersize=5)
    ax2.set_yticks(y, [DISPLAY[m] for m in compare_order])
    ax2.invert_yaxis()
    ax2.set_xlabel("Paired Δ mAP from Visual GRU")
    ax2.grid(axis="x", color="#D9DEE5", linewidth=0.6)
    ax2.text(-0.18, 1.03, "b", transform=ax2.transAxes, weight="bold", fontsize=8)
    ax2.text(0.0, 1.03, "Mean difference and descriptive 95% CI", transform=ax2.transAxes, weight="bold", fontsize=7)
    boot = next(row for row in evidence["paired_game_cluster_bootstrap"]["results"] if row["comparison"] == "late minus visual")
    ax2.text(
        0.98,
        0.83,
        f"Late−visual game bootstrap 95% CI\n[{boot['game_cluster_bootstrap_ci95_low']:+.4f}, {boot['game_cluster_bootstrap_ci95_high']:+.4f}]",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=5.4,
        color=COLORS["dark"],
    )
    save_figure(fig, "Fig2")


def figure3(rows, evidence) -> None:
    source_rob = []
    def add_values(condition, record):
        for seed, value in zip(record.get("seeds", [2026, 2027, 2028, 2029, 2030]), record["event_mAP@10s"]["values"]):
            source_rob.append({"condition": condition, "seed": seed, "mAP_10s": value})

    late = evidence["main_models"]["late"]
    add_values("Time-bounded", late)
    tfidf = next(iter(evidence["tfidf_sensitivity"].values()))
    add_values("TF-IDF", tfidf)
    for condition, key in [("+10-s delay", "event_mAP@10s_text_delay_10s"), ("+20-s delay", "event_mAP@10s_text_delay_20s")]:
        values = evidence["late_text_delay"][key]["values"]
        for seed, value in zip(late["seeds"], values):
            source_rob.append({"condition": condition, "seed": seed, "mAP_10s": value})
    for condition, record in [("Remove text", evidence["zero_text_inference"]["late"]), ("Retrain zero", evidence["text_controls"]["late_zero"]["event_mAP@10s"]), ("Shift 60 s", evidence["text_controls"]["late_shift60"]["event_mAP@10s"]), ("Future +10 s", next(iter(evidence["future_text_control"].values()))["event_mAP@10s"])]:
        for seed, value in zip(late["seeds"], record["values"]):
            source_rob.append({"condition": condition, "seed": seed, "mAP_10s": value})
    write_csv(SRC / "Fig3a_source_data.csv", ["condition", "seed", "mAP_10s"], source_rob)
    class_source = evidence["class_effects_late_minus_visual"]
    write_csv(SRC / "Fig3b_source_data.csv", ["class", "late_minus_visual", "late_mean", "visual_mean"], class_source)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.08, 4.15), gridspec_kw={"width_ratios": [1.05, 1.5]}, constrained_layout=True)
    conditions = ["Time-bounded", "TF-IDF", "+10-s delay", "+20-s delay", "Remove text", "Retrain zero", "Shift 60 s", "Future +10 s"]
    cy = np.arange(len(conditions))
    means=[]; sds=[]
    for condition in conditions:
        vals=[r["mAP_10s"] for r in source_rob if r["condition"]==condition]
        means.append(np.mean(vals)); sds.append(np.std(vals,ddof=1))
    colors=[COLORS["late"]]*4+[COLORS["neutral"]]*3+[COLORS["text"]]
    ax.errorbar(means, cy, xerr=sds, fmt="none", ecolor=COLORS["dark"], capsize=2.5, linewidth=0.9)
    ax.scatter(means,cy,c=colors,s=30,zorder=3,edgecolor="white",linewidth=0.5)
    ax.set_yticks(cy, ["Primary", "TF-IDF", "+10-s delay", "+20-s delay", "Remove text", "Retrain zero", "Shift 60 s", "Future +10 s"])
    ax.invert_yaxis()
    ax.set_xlabel("Event mAP within ±10 s")
    ax.axvline(evidence["main_models"]["visual"]["event_mAP@10s"]["mean"], color=COLORS["visual"], linestyle="--", linewidth=0.9)
    ax.text(-0.16, 1.03, "a", transform=ax.transAxes, weight="bold", fontsize=8)
    ax.text(0.0, 1.03, "Text representation and timing controls", transform=ax.transAxes, weight="bold", fontsize=7)

    diffs = {r["class"]: r["late_minus_visual"] for r in class_source}
    sorted_classes = sorted(diffs, key=lambda c: diffs[c])
    yy = np.arange(len(sorted_classes))
    colors = [COLORS["late"] if diffs[c] >= 0 else COLORS["neutral"] for c in sorted_classes]
    ax2.barh(yy, [diffs[c] for c in sorted_classes], color=colors, height=0.68)
    ax2.axvline(0, color=COLORS["dark"], linewidth=0.8)
    ax2.set_yticks(yy, sorted_classes)
    ax2.set_xlabel("Class ΔAP: Late fusion − Visual GRU")
    ax2.grid(axis="x", color="#D9DEE5", linewidth=0.6)
    ax2.text(-0.20, 1.03, "b", transform=ax2.transAxes, weight="bold", fontsize=8)
    ax2.text(0.0, 1.03, "Class-specific effects", transform=ax2.transAxes, weight="bold", fontsize=7)
    save_figure(fig, "Fig3")


def main() -> None:
    figure1()
    if all((ART / name).exists() for name in ["experiment_results.json", "final_revision_evidence.json"]):
        rows, evidence = load_results()
        figure2(rows, evidence)
        figure3(rows, evidence)


if __name__ == "__main__":
    main()
