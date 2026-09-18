"""Continue the reproducible experiment automatically after data transfer finishes."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT.parent / "f"
ARTIFACTS = ROOT / "artifacts"
LOG = ROOT / "logs" / "experiment-run.log"


def count_features() -> int:
    return sum(1 for _ in FEATURES.rglob("*.npy"))


manifest = json.loads((ARTIFACTS / "feature_download_manifest.json").read_text(encoding="utf-8")) if (ARTIFACTS / "feature_download_manifest.json").exists() else {"files_requested": 460}
expected = int(manifest["files_requested"])
with LOG.open("a", encoding="utf-8") as handle:
    while count_features() < expected:
        handle.write(f"waiting for features: {count_features()}/{expected}\n")
        handle.flush()
        time.sleep(30)
    handle.write(f"features ready: {expected}/{expected}; starting models\n")
    handle.flush()
    process = subprocess.run(
        [sys.executable, str(ROOT / "src" / "run_experiments.py"), "--epochs", "2"],
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    handle.write(f"experiment exit code: {process.returncode}\n")
