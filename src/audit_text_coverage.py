"""Distinguish transcript, in-vocabulary-token, and nonzero-vector coverage."""

from __future__ import annotations

import json

import numpy as np

from tgre_experiment import ARTIFACTS, Corpus, observation_time


def main() -> None:
    corpus = Corpus(ARTIFACTS / "modality_coverage.json")
    output = {}
    for split in ("train", "valid"):
        bins = transcript_bins = token_bins = 0
        segments = 0
        for game in corpus.games[split]:
            for half in (1, 2):
                data = corpus.half(game, half)
                bins += len(data.text)
                token_bins += int(np.count_nonzero(data.text.sum(axis=1)))
                rows = corpus.echoes.get((game, half), [])
                segments += len(rows)
                ends = np.asarray([row[1] for row in rows], dtype=float)
                cutoffs = observation_time(np.arange(len(data.text)))
                if ends.size:
                    upper = np.searchsorted(ends, cutoffs, side="right")
                    lower = np.searchsorted(ends, cutoffs - 20.0, side="right")
                    transcript_bins += int(np.count_nonzero(upper > lower))
        output[split] = {
            "bins": bins,
            "segments": segments,
            "bins_with_completed_segment_in_20s_window": transcript_bins,
            "segment_coverage_fraction": transcript_bins / bins,
            "bins_with_nonzero_256_term_vector": token_bins,
            "nonzero_vector_fraction": token_bins / bins,
        }
    (ARTIFACTS / "text_coverage_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
