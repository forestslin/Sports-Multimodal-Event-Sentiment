"""Fetch the single documented English Echoes Arrow dataset file."""

from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path(__file__).resolve().parents[1] / "data" / "SN-echoes"
root.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(
    repo_id="SoccerNet/SN-echoes",
    repo_type="dataset",
    filename="whisper_v1_en/1.0.0/soccer_net_echoes_hf_dataset-train.arrow",
    local_dir=root,
)
print(path)
