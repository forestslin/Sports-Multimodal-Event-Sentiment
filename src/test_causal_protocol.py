"""Focused checks for the causal-v2 temporal protocol."""

import numpy as np

from tgre_experiment import (
    BIN_SECONDS,
    causal_text_matrix,
    observation_time,
    targets_for_half,
    windowed,
)


def main() -> None:
    assert observation_time(0) == 2.0
    assert observation_time(3) == 8.0

    vocab = {"goal": 0, "late": 1}
    segments = [
        (0.2, 1.9, "goal"),
        (1.8, 2.1, "late"),
    ]
    text = causal_text_matrix(segments, 3, vocab)
    assert text[0, 0] > 0 and text[0, 1] == 0, "2-s cutoff admitted future-ending ASR"
    assert text[1, 1] > 0, "4-s cutoff failed to admit completed ASR"
    symmetric = causal_text_matrix(segments, 3, vocab, past_lookback=10.0, future_access=10.0)
    assert symmetric[0, 1] > 0, "symmetric control failed to admit future-ending ASR"

    target, retained = targets_for_half(5, [(0.0, 0), (1.9, 0), (2.0, 1), (2.1, 2)])
    assert target[0, 0] == 1 and target[0, 1] == 1
    assert target[1, 2] == 1
    assert len(retained) == 4

    values = np.arange(6, dtype=np.float32)[:, None]
    windows = windowed(values, np.array([0, 1, 5]))[:, :, 0]
    assert np.array_equal(windows[0], np.zeros(6, dtype=np.float32))
    assert windows[1, -1] == 1 and windows[1].max() <= 1
    assert windows[2, -1] == 5 and windows[2].max() <= 5

    print("causal-v2 protocol checks passed")


if __name__ == "__main__":
    main()
