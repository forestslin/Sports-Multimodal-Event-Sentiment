"""Compare retained and excluded official games for observable selection differences."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from SoccerNet.utils import getListGames


ROOT = Path(__file__).resolve().parents[1]
LABEL_ROOT = ROOT / "data" / "SN-Labels"
ART = ROOT / "artifacts_causal_v2"


def label_counts(games: set[str]) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    available = 0
    for game in games:
        path = LABEL_ROOT.joinpath(*game.split("\\")) / "Labels-v2.json"
        if not path.exists():
            continue
        available += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts.update(row.get("label", "UNKNOWN") for row in payload.get("annotations", []))
    return counts, available


def composition(games: set[str]) -> dict[str, dict[str, int]]:
    leagues = Counter()
    seasons = Counter()
    league_seasons = Counter()
    for game in games:
        parts = game.split("\\")
        league = parts[0]
        season = parts[1]
        leagues[league] += 1
        seasons[season] += 1
        league_seasons[f"{league}/{season}"] += 1
    return {
        "leagues": dict(sorted(leagues.items())),
        "seasons": dict(sorted(seasons.items())),
        "league_seasons": dict(sorted(league_seasons.items())),
    }


def total_variation(left: Counter[str], right: Counter[str]) -> float | None:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total == 0 or right_total == 0:
        return None
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total) for key in keys)


def main() -> None:
    coverage = json.loads((ART / "modality_coverage.json").read_text(encoding="utf-8"))["splits"]
    output = {"note": "Source-language metadata are unavailable in whisper_v1_en and therefore cannot be audited.", "splits": {}}
    for split in ("train", "valid"):
        official = set(getListGames(split))
        retained = set(coverage[split]["games"])
        excluded = official - retained
        retained_labels, retained_label_games = label_counts(retained)
        excluded_labels, excluded_label_games = label_counts(excluded)
        retained_composition = composition(retained)
        excluded_composition = composition(excluded)
        output["splits"][split] = {
            "official_games": len(official),
            "retained_games": len(retained),
            "excluded_games": len(excluded),
            "retained_labelled_games": retained_label_games,
            "excluded_labelled_games": excluded_label_games,
            "retained_composition": retained_composition,
            "excluded_composition": excluded_composition,
            "retained_label_counts": dict(sorted(retained_labels.items())),
            "excluded_label_counts": dict(sorted(excluded_labels.items())),
            "retained_annotations_per_labelled_game": sum(retained_labels.values()) / retained_label_games if retained_label_games else None,
            "excluded_annotations_per_labelled_game": sum(excluded_labels.values()) / excluded_label_games if excluded_label_games else None,
            "label_distribution_total_variation": total_variation(retained_labels, excluded_labels),
            "league_distribution_total_variation": total_variation(
                Counter(retained_composition["leagues"]), Counter(excluded_composition["leagues"])
            ),
        }
    (ART / "selection_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({s: {k: v for k, v in d.items() if not isinstance(v, dict)} for s, d in output["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
