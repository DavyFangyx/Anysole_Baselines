"""M3 alignment tests on one canonical session.

The adapter must consume the shared aligned frame grid as-is: no t_us /
visual-offset interpretation, no new 40 Hz grid, and every private file keeps
the shared frame id.  The historical systematic ~11-frame shift between
pressure and foot height must be gone (median |lag| over dense windows <= 3).

Run directly (touch_gait env):

    python tests/test_shared_frame_alignment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AnysoleWorkspace.tool.adapters.MotionPRO.adapter import (  # noqa: E402
    _context,
    _foot_signal,
    load_shared_session,
    soft_f6_contact,
)

SESSION_ID = "S5091"
ADAPTER_SOURCE = (REPO_ROOT / "AnysoleWorkspace/tool/adapters/MotionPRO/"
                  "adapter.py").read_text(encoding="utf-8")


def _best_lag(a: np.ndarray, b: np.ndarray, maxlag: int = 25) -> int:
    a = (a - a.mean()) / max(a.std(), 1e-9)
    b = (b - b.mean()) / max(b.std(), 1e-9)
    correlations = []
    for lag in range(-maxlag, maxlag + 1):
        x, y = (a[lag:], b[:len(a) - lag]) if lag >= 0 else (a[:lag], b[-lag:])
        correlations.append(np.corrcoef(x, y)[0, 1])
    return int(np.argmax(correlations) - maxlag)


def test_adapter_does_not_reinterpret_time():
    # The adapter consumes the shared grid directly; it must not read t_us,
    # visual offsets, or rebuild a 40 Hz grid.
    assert "t_us" not in ADAPTER_SOURCE
    assert "visual_start" not in ADAPTER_SOURCE
    assert "align_meta" not in ADAPTER_SOURCE


def test_shared_frame_id_consistency():
    session = load_shared_session(SESSION_ID)
    ctx = _context(session)
    contact = soft_f6_contact(ctx)
    n = len(session["frames"]["frame_id"])
    assert ctx["n"] == n
    assert contact.shape[0] == n
    assert np.asarray(session["pressure"]["left48"]).shape[0] == n
    frame_id = np.asarray(session["frames"]["frame_id"])
    assert np.array_equal(frame_id, np.arange(n, dtype=frame_id.dtype))


def test_no_systematic_pressure_height_shift():
    session = load_shared_session(SESSION_ID)
    ctx = _context(session)
    valid = np.asarray(session["frames"]["valid"], dtype=bool)
    for side, key in (("left", "left48"), ("right", "right48")):
        height, _ = _foot_signal(ctx, side)
        sums = np.asarray(session["pressure"][key], dtype=np.float64).sum(axis=1)
        lags = []
        for start in range(0, len(sums) - 100, 10):
            window = slice(start, start + 100)
            if not valid[window].all():
                continue
            if sums[window].std() < 1e-9 or height[window].std() < 1e-9:
                continue
            lags.append(_best_lag(sums[window], -height[window]))
        assert lags, f"{side}: no measurable windows"
        median = float(np.median(lags))
        assert abs(median) <= 3, (
            f"{side}: median pressure/height lag {median} frames — the "
            f"historical ~11-frame systematic shift is still present")
        print(f"  {side}: median lag {median} over {len(lags)} windows")


if __name__ == "__main__":
    try:
        load_shared_session(SESSION_ID)
    except FileNotFoundError as exc:
        raise SystemExit(f"skip: {exc}")
    for fn in (test_adapter_does_not_reinterpret_time,
               test_shared_frame_id_consistency,
               test_no_systematic_pressure_height_shift):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_shared_frame_alignment: all passed")
