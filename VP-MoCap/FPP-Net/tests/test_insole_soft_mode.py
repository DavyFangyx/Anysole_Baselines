"""#3 evidence: getVertsPress(soft=...) keeps the native result bit-for-bit.

``soft=False`` (the only mode any caller uses today) must reproduce the
upstream binary propagation exactly; ``soft=True`` is the extension that
propagates per-cell soft labels as a per-vertex mean without the final
binarization.  The equivalence is checked against a verbatim copy of the
upstream implementation, on both binary and real-valued grids.

Run directly:

    python tests/test_insole_soft_mode.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FPP_ROOT = Path(__file__).resolve().parents[1]
if str(FPP_ROOT) not in sys.path:
    sys.path.insert(0, str(FPP_ROOT))

from lib.Dataset.InsoleModule import InsoleModule  # noqa: E402


def native_get_verts_press(module, contact_label):
    """Upstream implementation, copied verbatim from the pristine snapshot."""
    left_press = contact_label[0]
    left_smpl = np.zeros([module.footIdsL.shape[0]], dtype=np.float32)
    for i in range(module.footIdsL.shape[0]):
        ids = module.footIdsL[i]
        if str(ids) in module.insole2smplL.keys():
            tmp = module.insole2smplL[str(ids)]
            _data = left_press[tmp[0], tmp[1]]
            if _data.shape[0] != 0:
                left_smpl[i] = np.sum(_data, axis=0)
    right_press = contact_label[1]
    right_smpl = np.zeros([module.footIdsR.shape[0]], dtype=np.float32)
    for i in range(module.footIdsR.shape[0]):
        ids = module.footIdsR[i]
        if str(ids) in module.insole2smplR.keys():
            tmp = module.insole2smplR[str(ids)]
            _data = right_press[tmp[0], tmp[1]]
            if _data.shape[0] != 0:
                right_smpl[i] = np.sum(_data, axis=0)
    smpl_cont = np.stack([left_smpl, right_smpl])
    smpl_cont[smpl_cont > 0.5] = 1
    return smpl_cont


def _binary_grid(seed):
    rng = np.random.default_rng(seed)
    return (rng.random((2, 31, 11)) < 0.12).astype(np.float32)


def test_default_mode_matches_native_binary_grids():
    module = InsoleModule()
    for seed in range(5):
        grid = _binary_grid(seed)
        got = module.getVertsPress(grid)
        want = native_get_verts_press(module, grid)
        assert got.shape == (2, 96) == want.shape
        assert np.array_equal(got, want), f"binary grid seed={seed} diverged"
        assert set(np.unique(got)) <= {0.0, 1.0}


def test_default_mode_matches_native_on_real_values():
    module = InsoleModule()
    rng = np.random.default_rng(7)
    grid = rng.random((2, 31, 11)).astype(np.float32) * 0.4
    got = module.getVertsPress(grid)
    want = native_get_verts_press(module, grid)
    assert np.array_equal(got, want), "non-binary input must still binarize"


def test_soft_mode_is_the_unbinarized_mean():
    module = InsoleModule()
    grid = _binary_grid(3)
    soft = module.getVertsPress(grid, soft=True)
    assert soft.shape == (2, 96)
    # per-vertex mean over the cells assigned by insole2smpl, independently
    for foot, press, ids_ls, mapping in (
            (0, grid[0], module.footIdsL, module.insole2smplL),
            (1, grid[1], module.footIdsR, module.insole2smplR)):
        for i in range(ids_ls.shape[0]):
            key = str(ids_ls[i])
            if key not in mapping:
                continue
            rows, cols = mapping[key]
            cells = press[rows, cols]
            expect = np.mean(cells) if cells.shape[0] else 0.0
            assert np.isclose(soft[foot, i], expect, atol=1e-6), (
                f"foot {foot} vertex {i} soft mean mismatch")
    assert soft.max() <= 1.0, "soft values stay in the label range"
    # binary grid + mean: a single active cell is diluted, not binarised
    assert not np.array_equal(soft, native_get_verts_press(module, grid))


if __name__ == "__main__":
    for fn in (test_default_mode_matches_native_binary_grids,
               test_default_mode_matches_native_on_real_values,
               test_soft_mode_is_the_unbinarized_mean):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_insole_soft_mode: all passed")
