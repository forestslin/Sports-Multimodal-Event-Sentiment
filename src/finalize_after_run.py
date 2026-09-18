"""Finish statistical and official-evaluator checks after all runs complete."""

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts_causal_v2"
expected = 35
while True:
    path = ART / "experiment_results.json"
    checkpoints = list((ART / "checkpoints").glob("*_epoch20.pt")) if (ART / "checkpoints").exists() else []
    if path.exists():
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            rows = []
        if len(rows) == expected and len(checkpoints) >= expected:
            break
    time.sleep(30)

subprocess.run([sys.executable, str(ROOT / "src" / "summarize_results.py")], cwd=ROOT.parent, check=True)
groups = defaultdict(list)
for row in rows:
    groups[row["model"]].append(row)
means = {name: sum(r["event_mAP@10s"] for r in values) / len(values) for name, values in groups.items()}
best_model = max(means, key=means.get)
best_seed_row = min(groups[best_model], key=lambda r: abs(r["event_mAP@10s"] - means[best_model]))
subprocess.run(
    [sys.executable, str(ROOT / "src" / "official_crosscheck.py"), "--model", best_model, "--seed", str(int(best_seed_row["seed"])), "--epochs", "20"],
    cwd=ROOT.parent,
    check=True,
)
subprocess.run([sys.executable, str(ROOT / "src" / "robustness_and_timing.py")], cwd=ROOT.parent, check=True)
subprocess.run([sys.executable, str(ROOT / "src" / "make_figures.py")], cwd=ROOT.parent, check=True)
(ART / "experiment_pipeline_complete.json").write_text(
    json.dumps({"status": "complete", "best_model_by_mean_map10": best_model, "representative_seed": int(best_seed_row["seed"]), "mean_map10": means[best_model]}, indent=2),
    encoding="utf-8",
)
