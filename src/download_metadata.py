"""Download only the public metadata needed to define a reproducible study split.

The script deliberately does not request SoccerNet video files.  The study uses
the published 2-fps ResNet PCA features and time-aligned SoccerNet-Echoes ASR
transcripts, both public dataset artifacts.
"""

from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def fetch(repo_id: str, target: Path, patterns: list[str] | None = None) -> None:
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target,
        allow_patterns=patterns,
    )


if __name__ == "__main__":
    # Official event labels are compact and establish the train/validation split.
    fetch("SoccerNet/SN-Labels", DATA / "SN-Labels")
    # Echoes consists of several ASR variants.  Retaining the English Whisper v1
    # release keeps the text modality defined and computationally manageable.
    fetch("SoccerNet/SN-echoes", DATA / "SN-echoes", ["whisper_v1_en/**"])
