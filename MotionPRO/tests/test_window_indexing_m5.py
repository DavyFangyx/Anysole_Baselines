"""M5 window indexing tests.

Train/eval index complete, fully valid windows only; the tail that cannot
fill a window never enters training.  Test mode (full-session inference
export) keeps the final partial window so it can be zero-padded with
valid=0 — no loss mask is invented and no window ever spans a fake frame.

Run directly (touch_gait env):

    python tests/test_window_indexing_m5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MOTIONPRO_ROOT = Path(__file__).resolve().parents[1]
if str(MOTIONPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONPRO_ROOT))

from lib.dataset.image_pressure import window_ranges  # noqa: E402


def test_train_complete_windows_only():
    invalid = np.zeros(45, dtype=np.uint8)
    windows = window_ranges(45, 20, invalid, "train")
    assert windows == [(0, 20), (20, 40)], "tail frames 40..44 must be excluded"
    assert all(right - left == 20 for left, right in windows)


def test_eval_complete_windows_only():
    invalid = np.zeros(45, dtype=np.uint8)
    assert window_ranges(45, 20, invalid, "eval") == [(0, 20), (20, 40)]


def test_test_mode_full_session_coverage():
    invalid = np.zeros(45, dtype=np.uint8)
    windows = window_ranges(45, 20, invalid, "test")
    assert windows == [(0, 20), (20, 40), (40, 45)], (
        "test mode keeps the partial tail for valid-masked export padding")
    assert windows[-1][1] - windows[-1][0] == 5


def test_invalid_windows_are_skipped():
    invalid = np.zeros(45, dtype=np.uint8)
    invalid[25] = 1  # inside the second complete window
    assert window_ranges(45, 20, invalid, "train") == [(0, 20)]
    # test mode: the invalid complete window is dropped but the valid
    # partial tail window is still exported with the valid mask.
    assert window_ranges(45, 20, invalid, "test") == [(0, 20), (40, 45)]
    invalid[41] = 1  # now the partial tail is invalid too
    assert window_ranges(45, 20, invalid, "train") == [(0, 20)]
    assert window_ranges(45, 20, invalid, "test") == [(0, 20)]
    invalid[25] = 0  # only the tail is invalid: complete windows unaffected
    assert window_ranges(45, 20, invalid, "train") == [(0, 20), (20, 40)]
    assert window_ranges(45, 20, invalid, "test") == [(0, 20), (20, 40)]


def test_no_window_spans_fake_frames():
    rng = np.random.default_rng(0)
    invalid = (rng.random(137) < 0.15).astype(np.uint8)
    for mode in ("train", "eval", "test"):
        for left, right in window_ranges(137, 20, invalid, mode):
            assert not invalid[left:right].any(), f"{mode} window spans invalid frames"


if __name__ == "__main__":
    for fn in (test_train_complete_windows_only, test_eval_complete_windows_only,
               test_test_mode_full_session_coverage, test_invalid_windows_are_skipped,
               test_no_window_spans_fake_frames):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_window_indexing_m5: all passed")
