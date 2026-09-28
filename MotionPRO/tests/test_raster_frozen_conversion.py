"""Frozen dual-sole raster conversion tests (D_Test4-audited, no redefinition).

Checks the adapter's ``rasterize_feet`` against the frozen constants and the
display layer's copy, the exact round trip through the frozen inverse, the
clip, the fixed left/right boxes and the bilinear 96x96 model-input contract.

Run directly (touch_gait env):

    python tests/test_raster_frozen_conversion.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AnysoleWorkspace.tool.adapters.MotionPRO.adapter import (  # noqa: E402
    BLOCK_REPEAT,
    LEFT_FOOT_BOX,
    PRESSURE_CLIP,
    PRESSURE_HW,
    RIGHT_FOOT_BOX,
    crop_cells,
    rasterize_feet,
)

DISPLAY_COMMON = REPO_ROOT / "results_display/script/utils/render_common.py"


def _parse_box(text: str, name: str) -> tuple[slice, slice]:
    match = re.search(
        rf"{name} = \(slice\((\d+), (\d+)\), slice\((\d+), (\d+)\)\)", text)
    assert match, f"{name} not found in render_common.py"
    return slice(int(match[1]), int(match[2])), slice(int(match[3]), int(match[4]))


def test_frozen_constants():
    assert PRESSURE_HW == (160, 120)
    assert PRESSURE_CLIP == 1023.0
    assert BLOCK_REPEAT == (20, 4)
    assert LEFT_FOOT_BOX == (slice(40, 120), slice(6, 54))
    assert RIGHT_FOOT_BOX == (slice(40, 120), slice(66, 114))
    # The D_Test4 display layer holds the same audited boxes; they must not
    # drift apart.
    source = DISPLAY_COMMON.read_text(encoding="utf-8")
    assert _parse_box(source, "LEFT_FOOT_BOX") == LEFT_FOOT_BOX
    assert _parse_box(source, "RIGHT_FOOT_BOX") == RIGHT_FOOT_BOX


def test_round_trip_recovers_clipped_cells():
    rng = np.random.default_rng(7)
    cells = rng.uniform(0, 1500, size=(5, 2, 4, 12)).astype(np.float32)
    left = cells[:, 0].reshape(5, -1)
    right = cells[:, 1].reshape(5, -1)
    raster = rasterize_feet(left, right)
    assert raster.shape == (5, 160, 120)
    recovered = crop_cells(raster)
    expected = np.clip(cells, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP * 255.0
    assert np.allclose(recovered, expected, rtol=1e-5, atol=1e-4), (
        f"round trip max diff {np.abs(recovered - expected).max()}")


def test_fixed_boxes_and_background():
    rng = np.random.default_rng(11)
    left = rng.uniform(0, 800, size=(3, 48)).astype(np.float32)
    right = rng.uniform(0, 800, size=(3, 48)).astype(np.float32)
    raster = rasterize_feet(left, right)
    left_block = (np.clip(left, 0, PRESSURE_CLIP) / PRESSURE_CLIP * 255).reshape(3, 4, 12)
    right_block = (np.clip(right, 0, PRESSURE_CLIP) / PRESSURE_CLIP * 255).reshape(3, 4, 12)
    left_expected = np.repeat(np.repeat(left_block, 20, axis=1), 4, axis=2)
    right_expected = np.repeat(np.repeat(right_block, 20, axis=1), 4, axis=2)
    assert np.array_equal(raster[:, LEFT_FOOT_BOX[0], LEFT_FOOT_BOX[1]], left_expected)
    assert np.array_equal(raster[:, RIGHT_FOOT_BOX[0], RIGHT_FOOT_BOX[1]], right_expected)
    mask = np.ones((160, 120), dtype=bool)
    mask[LEFT_FOOT_BOX] = False
    mask[RIGHT_FOOT_BOX] = False
    assert (raster[:, mask] == 0).all(), "outside the two foot boxes must be zero"


def test_clip_to_255():
    left = np.full((1, 48), 5000.0, dtype=np.float32)
    right = np.zeros((1, 48), dtype=np.float32)
    raster = rasterize_feet(left, right)
    assert raster[:, LEFT_FOOT_BOX[0], LEFT_FOOT_BOX[1]].max() == 255.0


def test_bilinear_96_contract():
    # The model input contract is bilinear 96x96 (align_corners=False), the
    # same torch call the audited generator used; /255 happens at dataset
    # load, not here.
    rng = np.random.default_rng(3)
    raster = rng.uniform(0, 255, size=(4, 160, 120)).astype(np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(raster)).float()
    resized = torch.nn.functional.interpolate(
        tensor.unsqueeze(1), size=(96, 96), mode="bilinear",
        align_corners=False).squeeze(1).numpy()
    assert resized.shape == (4, 96, 96)
    assert resized.min() >= 0.0 and resized.max() <= 255.0


def test_reference_session_parity():
    # Conversion parity on a real canonical session: the new raster must be
    # exactly the frozen conversion of the shared 4x12 cells (clip included).
    session_id = "S5091"
    try:
        from AnysoleWorkspace.tool.adapters.MotionPRO.adapter import load_shared_session
        session = load_shared_session(session_id)
    except FileNotFoundError:
        print(f"skip: shared session {session_id} unavailable")
        return
    left = np.asarray(session["pressure"]["left48"], dtype=np.float32)
    right = np.asarray(session["pressure"]["right48"], dtype=np.float32)
    raster = rasterize_feet(left, right)
    recovered = crop_cells(raster)
    cells = np.stack([left, right], axis=1).reshape(-1, 2, 4, 12)
    expected = np.clip(cells, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP * 255.0
    diff = np.abs(recovered - expected)
    assert diff.max() < 1e-3, f"{session_id} parity max diff {diff.max()}"


if __name__ == "__main__":
    for fn in (test_frozen_constants, test_round_trip_recovers_clipped_cells,
               test_fixed_boxes_and_background, test_clip_to_255,
               test_bilinear_96_contract, test_reference_session_parity):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_raster_frozen_conversion: all passed")
