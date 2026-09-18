"""Audit official SoccerNet event labels before any model fitting."""

import json
from collections import Counter
from pathlib import Path

from SoccerNet.utils import getListGames


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "SN-Labels"
OUT = ROOT / "artifacts"


def label_file(game: str) -> Path | None:
    return LABEL_INDEX.get(game)


def main() -> None:
    global LABEL_INDEX
    # Index once: repeatedly walking the large repository tree makes this audit
    # unnecessarily slow on Windows.
    LABEL_INDEX = {
        str(path.parent.relative_to(LABELS)).replace("/", "\\"): path
        for path in LABELS.rglob("Labels-v2.json")
    }
    inventory: dict[str, object] = {"splits": {}, "classes": {}}
    classes: Counter[str] = Counter()
    for split in ("train", "valid"):
        games = getListGames(split)
        present = []
        annotations = 0
        for game in games:
            path = label_file(game)
            if not path:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("annotations", [])
            present.append(game)
            annotations += len(rows)
            classes.update(row["label"] for row in rows)
        inventory["splits"][split] = {
            "official_games": len(games),
            "labelled_games_downloaded": len(present),
            "annotations": annotations,
            "games": present,
        }
    inventory["classes"] = dict(sorted(classes.items()))
    OUT.mkdir(exist_ok=True)
    (OUT / "label_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(json.dumps({k: {x: y for x, y in v.items() if x != 'games'} for k, v in inventory['splits'].items()}, indent=2))
    print("classes", len(classes), sorted(classes))


if __name__ == "__main__":
    main()
