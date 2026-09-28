"""#6 evidence: keypoints are joined by frame id, not by array position.

The pristine window read ``keypoints/%03d.npy % (frame_idx + i)`` — a purely
positional read of the adapter tree.  The ported code names the file with the
canonical six-digit frame id and refuses to use a sidecar whose stored
``frame_id`` differs from the requested one, so a non-positional id map can
never silently shift the temporal window.

The real adapter tree is exercised end to end as well (a full item load), so
the test covers the shipped data, not only the fixture.

Run directly:

    python tests/test_frame_id_join.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

FPP_ROOT = Path(__file__).resolve().parents[1]
if str(FPP_ROOT) not in sys.path:
    sys.path.insert(0, str(FPP_ROOT))

from lib.config.config import config_cont  # noqa: E402
from lib.Dataset.PressDataset.PED_tempKPCont import ContDataset  # noqa: E402

CONFIG = FPP_ROOT / "configs/temporalKPSMPLCont_series5_mlp.yaml"
N_JOINTS = 26
N_FRAMES = 8
CENTRE = 4
WEIGHT = 12345.0


def _cfg(datadir, tv_fn, tactile_root):
    cfg = config_cont()
    cfg.load(str(CONFIG))
    cfg = cfg.get_cfg()
    cfg.defrost()
    cfg.dataset.datadir = str(datadir)
    cfg.dataset.tv_fn = str(tv_fn)
    cfg.dataset.tactile_root = str(tactile_root)
    cfg.dataset.aug.is_aug = False
    cfg.freeze()
    return cfg


def _fixture(tmp, stored_frame_offset=0, omit_frame_id=False):
    """A minimal tree: public insole sidecars + keypoint sidecars with ids."""
    root = Path(tmp)
    seq = root / "datadir/20260808/S10/S1011"
    (seq / "keypoints").mkdir(parents=True)
    np.save(root / "datadir/20260808/sub_info.npy", {"S10": {"weight": WEIGHT}})
    rng = np.random.default_rng(3)
    for frame in range(N_FRAMES):
        data = {
            "keypoints": (rng.random((N_JOINTS, 2)) * 500 + 100).astype(np.float32),
            "keypoint_scores": np.full((N_JOINTS,), 0.9, dtype=np.float32),
        }
        if not omit_frame_id:
            data["frame_id"] = frame + stored_frame_offset
        np.save(seq / "keypoints" / ("%06d.npy" % frame), data)

    tactile = root / "tactile/20260808/S10/S1011"
    (tactile / "insole").mkdir(parents=True)
    for frame in range(N_FRAMES):
        np.save(tactile / "insole" / ("%06d.npy" % frame),
                (rng.random((2, 31, 11)) * 500).astype(np.float32))
    np.save(tactile / "frame_id.npy", np.arange(N_FRAMES))

    split = root / "split.npy"
    np.save(split, {"test": {"20260808": {"S10": {"S1011":
                 ["%06d.npy" % CENTRE]}}}})
    return _cfg(root / "datadir", split, root / "tactile")


def test_window_is_read_by_six_digit_frame_id():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _fixture(tmp)
        dataset = ContDataset(cfg.dataset, "test")
        assert dataset.images_fn == ["20260808/S10/S1011/%06d" % CENTRE]
        item = dataset[0]
        assert item["frame_id"] == CENTRE
        assert item["keypoints"].shape == (5, N_JOINTS, 3)
        # the sidecar files really are the six-digit names
        names = sorted(p.name for p in
                       (Path(tmp) / "datadir/20260808/S10/S1011/keypoints").iterdir())
        assert names == ["%06d.npy" % i for i in range(N_FRAMES)], names


def test_sidecar_id_mismatch_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _fixture(tmp, stored_frame_offset=1000)  # every id shifted
        dataset = ContDataset(cfg.dataset, "test")
        try:
            dataset[0]
        except ValueError as err:
            assert "frame id mismatch" in str(err), err
            # the window starts at centre-2, so the first mismatch is 2 -> 1002
            assert "stored 1002 != requested 2" in str(err), err
            return
        raise AssertionError("a shifted keypoint id map must not be accepted")


def test_sidecar_without_frame_id_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _fixture(tmp, omit_frame_id=True)
        dataset = ContDataset(cfg.dataset, "test")
        try:
            dataset[0]
        except ValueError as err:
            assert "stored -1" in str(err), err
            return
        raise AssertionError("a sidecar without frame_id must not be accepted")


def test_real_adapter_tree_joins_the_window():
    """A real item: the five read sidecars must all carry the requested ids."""
    cfg = config_cont()
    cfg.load(str(CONFIG))
    cfg = cfg.get_cfg()
    dataset = ContDataset(cfg.dataset, "test")
    item_at = N_FRAMES  # any listed window; the split is the adapter's
    date, sub, seq, centre = dataset.images_fn[item_at].split("/")
    keypoints_root = Path(cfg.dataset.datadir) / date / sub / seq / "keypoints"
    for offset in range(-2, 3):
        sidecar = keypoints_root / ("%06d.npy" % (int(centre) + offset))
        assert sidecar.is_file(), sidecar
        assert int(np.load(sidecar, allow_pickle=True).item()["frame_id"]) \
            == int(centre) + offset
    item = dataset[item_at]
    assert item["frame_id"] == int(centre)
    assert item["keypoints"].shape == (5, N_JOINTS, 3)


if __name__ == "__main__":
    for fn in (test_window_is_read_by_six_digit_frame_id,
               test_sidecar_id_mismatch_is_rejected,
               test_sidecar_without_frame_id_is_rejected,
               test_real_adapter_tree_joins_the_window):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_frame_id_join: all passed")
