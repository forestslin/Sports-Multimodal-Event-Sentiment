"""Download only feature tensors for the pre-registered matched cohort.

The cohort is fixed by ``modality_coverage.json`` before any training code is
run.  Each selected game receives exactly its two public 2-fps PCA-512 feature
files; raw match video is neither requested nor used.
"""

import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
coverage = json.loads((ROOT / "artifacts" / "modality_coverage.json").read_text(encoding="utf-8"))
games = coverage["splits"]["train"]["games"] + coverage["splits"]["valid"]["games"]
patterns = []
for game in games:
    path = game.replace("\\", "/")
    patterns.extend([f"{path}/1_ResNET_TF2_PCA512.npy", f"{path}/2_ResNET_TF2_PCA512.npy"])

# Keep this path short: the Hub cache appends a long revision hash and several
# official game titles exceed Windows' legacy 260-character limit otherwise.
target = ROOT.parent / "f"
target.mkdir(parents=True, exist_ok=True)
# huggingface_hub 1.31 can fail to create a nested cache parent on Windows for
# deeply nested game names.  Pre-create the exact cache parents before transfer.
for pattern in patterns:
    (target / ".cache" / "huggingface" / "download" / Path(pattern).parent).mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id="SoccerNet/SN-Features",
    repo_type="dataset",
    revision="resnet-tf2-pca512",
    local_dir=target,
    allow_patterns=patterns,
    # The short target path above keeps Hub's nested cache below the Windows path
    # limit; four transfers provide a practical download rate and resume safely.
    max_workers=4,
)
manifest = {
    "repository": "SoccerNet/SN-Features",
    "revision": "resnet-tf2-pca512",
    "feature": "ResNET_TF2_PCA512 at 2 fps",
    "selected_games": {"train": len(coverage["splits"]["train"]["games"]), "valid": len(coverage["splits"]["valid"]["games"])},
    "files_requested": len(patterns),
}
(ROOT / "artifacts" / "feature_download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
