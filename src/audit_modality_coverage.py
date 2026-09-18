"""Report the matched real-data cohort before downloading feature tensors."""

import json
from collections import Counter
from pathlib import Path

import pyarrow.ipc as ipc
from SoccerNet.utils import getListGames

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "SN-Labels"
ECHOES = ROOT / "data" / "SN-echoes" / "whisper_v1_en" / "1.0.0" / "soccer_net_echoes_hf_dataset-train.arrow"
OUT = ROOT / "artifacts"


def main() -> None:
    with ECHOES.open("rb") as handle:
        table = ipc.open_stream(handle).read_all().select(["game"])
    echo_games = {entry["game"].rsplit("/", 1)[0].replace("/", "\\") for entry in table.to_pylist()}
    labels = {str(p.parent.relative_to(LABELS)).replace("/", "\\") for p in LABELS.rglob("Labels-v2.json")}
    result: dict[str, object] = {"echoes_match_games": len(echo_games), "splits": {}}
    for split in ("train", "valid"):
        official = set(getListGames(split))
        both = sorted(official & echo_games & labels)
        result["splits"][split] = {
            "official_games": len(official),
            "with_echoes": len(official & echo_games),
            "with_labels": len(official & labels),
            "matched_vision_text_label": len(both),
            "games": both,
        }
    OUT.mkdir(exist_ok=True)
    (OUT / "modality_coverage.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({x: {k: v for k, v in y.items() if k != "games"} for x, y in result["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
