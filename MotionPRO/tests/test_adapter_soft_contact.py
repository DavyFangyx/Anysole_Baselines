"""MotionPRO private soft-f6 model input tests on one canonical session.

Verifies: (T,10) contact with only columns 6/7 non-zero, values restricted to
{0.05,0.30,0.70,0.95}, binarized soft values equal the motion_f6 hard states,
shared frame id consistency, and the required artifact.json declaration.

Run directly (touch_gait env):

    python tests/test_adapter_soft_contact.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AnysoleWorkspace.tool.adapters.MotionPRO.adapter import (  # noqa: E402
    ADAPTER_VERSION,
    CONTACT_COLS,
    F6_SOFT,
    LEFT_COL,
    RIGHT_COL,
    _context,
    _f6_pipeline,
    build_one,
    load_shared_session,
    model_input_dir,
    soft_f6_contact,
)

SESSION_ID = "S5091"


def _session():
    try:
        return load_shared_session(SESSION_ID)
    except FileNotFoundError as exc:
        raise SystemExit(f"skip: {exc}")


def test_value_set_and_columns():
    session = _session()
    ctx = _context(session)
    contact = soft_f6_contact(ctx)
    assert contact.shape == (ctx["n"], CONTACT_COLS)
    assert contact.dtype == np.float32
    allowed = {round(float(v), 2) for v in F6_SOFT.values()}
    for column in range(CONTACT_COLS):
        values = {round(float(v), 2) for v in np.unique(contact[:, column])}
        if column in (LEFT_COL, RIGHT_COL):
            assert values <= allowed, f"column {column} has {values}"
            assert values, f"column {column} is empty"
        else:
            assert values == {0.0}, f"column {column} must be all zero, got {values}"


def test_soft_binary_matches_motion_f6():
    session = _session()
    ctx = _context(session)
    contact = soft_f6_contact(ctx)
    states, _ = _f6_pipeline(ctx)
    # Binarizing the four soft levels must reproduce the motion_f6 hard
    # decision per foot (the frozen F6 semantics).
    for column in (LEFT_COL, RIGHT_COL):
        hard = (contact[:, column] > 0.5).astype(np.int8)
        assert np.array_equal(hard, states[:, (0, 1)[column - LEFT_COL]])


def test_model_input_files_and_frame_id():
    session = _session()
    out_dir = build_one(SESSION_ID)
    assert out_dir == model_input_dir(SESSION_ID)
    assert ADAPTER_VERSION in out_dir.parts
    pressure = np.load(out_dir / "pressure.npz")["pressure"]
    contact = np.load(out_dir / "contact.npy")
    frame_id = np.load(out_dir / "frame_id.npy")
    shared_frame_id = np.asarray(session["frames"]["frame_id"])
    assert pressure.shape == (len(shared_frame_id), 320, 120)
    assert contact.shape == (len(shared_frame_id), 10)
    assert np.array_equal(frame_id, shared_frame_id), "frame id must match shared facts"


def test_artifact_contact_declaration():
    out_dir = build_one(SESSION_ID)
    artifact = json.loads((out_dir / "artifact.json").read_text(encoding="utf-8"))
    contact = artifact["parameters"]["contact"]
    assert contact["physical_file"] == "contact.npy"
    assert contact["semantic_role"] == "motionpro_private_soft_f6_foot_loss_weight"
    assert contact["value_set"] == [0.05, 0.30, 0.70, 0.95]
    assert contact["formal_metric_input"] is False
    assert contact["native_diagnostic_only"] is True
    raster = artifact["parameters"]["raster"]
    assert raster["semantic"] == "virtual_tactile_carpet"
    assert raster["shape"] == [320, 120]
    assert raster["value"]["model_entry"].startswith("consumer-side bilinear 96x96")


if __name__ == "__main__":
    for fn in (test_value_set_and_columns, test_soft_binary_matches_motion_f6,
               test_model_input_files_and_frame_id, test_artifact_contact_declaration):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_adapter_soft_contact: all passed")
