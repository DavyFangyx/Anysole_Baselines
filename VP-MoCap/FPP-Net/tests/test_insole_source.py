"""#4 evidence: the insole comes from the single public MMVP 31x11 tree.

The pristine dataset read a private, per-model insole copy under the adapter
tree (``<datadir>/<date>/<sub>/<seq>/insole/<frame>.npy``, wrapped in a dict).
That tree no longer carries insole data; the ported dataset reads the public
``shared://representations/tactile/mmvp_31x11/v1`` representation instead,
validates the ``(2,31,11)`` shape, and refuses to address a session whose
``frame_id.npy`` is not positional ``0..n-1``.

The fixture builds both a public tree and a decoy private file holding a
different value, so the assertion is about *which* address the code reads,
not only about the numbers it returns.

Run directly:

    python tests/test_insole_source.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

FPP_ROOT = Path(__file__).resolve().parents[1]
if str(FPP_ROOT) not in sys.path:
    sys.path.insert(0, str(FPP_ROOT))

from lib.config.config import config_cont  # noqa: E402
from lib.Dataset.PressDataset.PED_tempKPCont import ContDataset  # noqa: E402

CONFIG = FPP_ROOT / "configs/temporalKPSMPLCont_series5_mlp.yaml"
N_JOINTS = 26
N_FRAMES = 6
WEIGHT = 12345.0
DECOY = 999.0  # stands in for the retired private copy's payload


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


def _insole_grid(seed):
    rng = np.random.default_rng(seed)
    return (rng.random((2, 31, 11)) * 500).astype(np.float32)


def _write_keypoints(seq_dir):
    """Keypoints for every frame, carrying the canonical frame id."""
    rng = np.random.default_rng(5)
    for frame in range(N_FRAMES):
        data = {
            "keypoints": (rng.random((N_JOINTS, 2)) * 500 + 100).astype(np.float32),
            "keypoint_scores": np.full((N_JOINTS,), 0.9, dtype=np.float32),
            "frame_id": frame,
        }
        for name in ("%03d.npy" % frame, "%06d.npy" % frame):
            kp_dir = seq_dir / "keypoints"
            kp_dir.mkdir(parents=True, exist_ok=True)
            np.save(kp_dir / name, data)


def _fixture(tmp, frames, frame_ids=None, shape=(2, 31, 11), private_decoy=True):
    root = Path(tmp)
    datadir = root / "datadir/20260808/S10/S1011"
    datadir.mkdir(parents=True)
    np.save(root / "datadir/20260808/sub_info.npy",
            {"S10": {"weight": WEIGHT}})
    _write_keypoints(datadir)
    if private_decoy:
        decoy_dir = datadir / "insole"
        decoy_dir.mkdir()
        for frame in range(N_FRAMES):
            np.save(decoy_dir / ("%06d.npy" % frame),
                    {"insole": np.full((2, 31, 11), DECOY, dtype=np.float32)})

    tactile = root / "tactile/20260808/S10/S1011"
    (tactile / "insole").mkdir(parents=True)
    grids = {}
    for frame in range(N_FRAMES):
        grid = _insole_grid(frame)
        np.save(tactile / "insole" / ("%06d.npy" % frame), grid)
        grids[frame] = grid
    ids = np.arange(N_FRAMES) if frame_ids is None else frame_ids
    np.save(tactile / "frame_id.npy", ids)

    split = root / "split.npy"
    np.save(split, {"test": {"20260808": {"S10": {"S1011":
                 ["%06d.npy" % f for f in frames]}}}})
    return _cfg(root / "datadir", split, root / "tactile"), grids


def test_item_reads_the_public_tree_not_a_private_copy():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, grids = _fixture(tmp, [2, 3])
        dataset = ContDataset(cfg.dataset, "test")
        assert str(dataset.tactile_root).startswith(str(Path(tmp) / "tactile"))
        # the split tree names the window centre by frame id; the item must
        # read the public file at exactly that id
        frame = int(dataset.images_fn[0].split("/")[-1])
        item = dataset[0]

        module = dataset.insole_module
        want = np.concatenate(
            [module.sigmoidNorm(grids[frame], WEIGHT)[0],
             module.sigmoidNorm(grids[frame], WEIGHT)[1]], axis=1)
        got = item["insole"].numpy()
        assert item["insole"].shape == (484,)
        assert np.allclose(got, want[module.maskImg], atol=1e-6), (
            "the item must carry the public-grid normalization")

        decoy = module.sigmoidNorm(np.full((2, 31, 11), DECOY, np.float32), WEIGHT)
        decoy = np.concatenate([decoy[0], decoy[1]], axis=1)
        assert not np.allclose(got, decoy[module.maskImg]), (
            "the retired private copy must not be the value source")

        assert item["keypoints"].shape == (5, N_JOINTS, 3)
        assert item["contact_label"].shape == (484,)
        assert item["contact_smpl"].shape == (192,)
        assert set(np.unique(item["contact_smpl"].numpy())) <= {0.0, 1.0}


def test_real_tree_address_is_the_shared_representation():
    """The shipped config must point at the public 31x11 tree."""
    cfg = config_cont()
    cfg.load(str(CONFIG))
    cfg = cfg.get_cfg()
    assert cfg.dataset.tactile_root.endswith(
        "representations/tactile/mmvp_31x11/v1"), cfg.dataset.tactile_root
    tactile_root = Path(cfg.dataset.tactile_root)
    dataset = ContDataset(cfg.dataset, "test")
    assert Path(dataset.tactile_root) == tactile_root
    # a real session: the public insole is a raw (2,31,11) float32 array
    date, sub, seq, frame = dataset.images_fn[0].split("/")
    public = np.load(tactile_root / date / sub / seq / "insole"
                     / ("%06d.npy" % int(frame)))
    assert public.shape == (2, 31, 11) and public.dtype == np.float32
    ids = np.load(tactile_root / date / sub / seq / "frame_id.npy")
    assert np.array_equal(ids, np.arange(ids.shape[0]))


def _expect(exc, fn, needle):
    try:
        fn()
    except exc as err:
        assert needle in str(err), f"unexpected message: {err}"
        return
    raise AssertionError(f"{exc.__name__} not raised ({needle})")


def test_missing_frame_id_sidecar_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, _ = _fixture(tmp, [0])
        (Path(tmp) / "tactile/20260808/S10/S1011/frame_id.npy").unlink()
        _expect(FileNotFoundError,
                lambda: ContDataset(cfg.dataset, "test")[0],
                "no frame_id.npy")


def test_non_positional_frame_ids_are_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, _ = _fixture(tmp, [0], frame_ids=np.arange(N_FRAMES) + 5)
        _expect(ValueError,
                lambda: ContDataset(cfg.dataset, "test")[0],
                "not positional")


def test_frame_outside_the_public_session_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, _ = _fixture(tmp, [N_FRAMES + 2])
        _expect(IndexError,
                lambda: ContDataset(cfg.dataset, "test")[0],
                "outside the public representation")


def test_wrong_public_shape_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, _ = _fixture(tmp, [0])
        np.save(Path(tmp) / "tactile/20260808/S10/S1011/insole/000000.npy",
                np.zeros((2, 31, 10), dtype=np.float32))
        _expect(ValueError,
                lambda: ContDataset(cfg.dataset, "test")[0],
                "expected (2,31,11)")


if __name__ == "__main__":
    for fn in (test_item_reads_the_public_tree_not_a_private_copy,
               test_real_tree_address_is_the_shared_representation,
               test_missing_frame_id_sidecar_is_rejected,
               test_non_positional_frame_ids_are_rejected,
               test_frame_outside_the_public_session_is_rejected,
               test_wrong_public_shape_is_rejected):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_insole_source: all passed")
