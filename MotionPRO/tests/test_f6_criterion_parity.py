"""F6 criterion reuse evidence: MotionPRO adapter vs AnySole contact_adapter.

Both implementations follow the frozen F6 design (fix_plan_v2.md F6a).  This
test compares the computed motion_f6 states, pressure_f6 loaded values and
the four-level soft values on one canonical session — the MotionPRO private
labels must be exactly the same criterion output, not a re-derived variant.
(Read-only computation; no AnySole label file is read or written.)

Run directly (touch_gait env):

    python tests/test_f6_criterion_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AnysoleWorkspace.tool.adapters.MotionPRO.adapter import (  # noqa: E402
    _context as motionpro_context,
    _f6_pipeline as motionpro_pipeline,
    soft_f6_contact,
)
from anysole.data.contact_adapter import (  # noqa: E402
    _context as anyspace_context,
    _f6_pipeline as anyspace_pipeline,
)

SESSION_ID = "S5091"


def test_states_and_loaded_match():
    from AnysoleWorkspace.tool.adapters.MotionPRO.adapter import load_shared_session
    session = load_shared_session(SESSION_ID)
    ctx_m = motionpro_context(session)
    ctx_a = anyspace_context(session)
    states_m, loaded_m = motionpro_pipeline(ctx_m)
    states_a, loaded_a = anyspace_pipeline(ctx_a)
    assert np.array_equal(states_m, states_a), "motion_f6 states differ"
    assert np.array_equal(loaded_m, loaded_a), "pressure_f6 loaded differ"


def test_soft_values_match():
    from AnysoleWorkspace.tool.adapters.MotionPRO.adapter import load_shared_session
    session = load_shared_session(SESSION_ID)
    contact = soft_f6_contact(motionpro_context(session))
    states_a, loaded_a = anyspace_pipeline(anyspace_context(session))
    expected = np.zeros((contact.shape[0], 2), dtype=np.float32)
    for column in range(2):
        state, loaded = states_a[:, column], loaded_a[:, column]
        values = np.zeros(len(state), dtype=np.float32)
        values[state == 1] = np.where(loaded[state == 1], 0.95, 0.70)
        values[state == 0] = np.where(loaded[state == 0], 0.30, 0.05)
        expected[:, column] = values
    assert np.array_equal(contact[:, 6], expected[:, 0]), "left soft values differ"
    assert np.array_equal(contact[:, 7], expected[:, 1]), "right soft values differ"


if __name__ == "__main__":
    for fn in (test_states_and_loaded_match, test_soft_values_match):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_f6_criterion_parity: all passed")
