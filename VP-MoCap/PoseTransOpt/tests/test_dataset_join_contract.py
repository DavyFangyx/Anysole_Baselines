"""PoseTransOpt dataset join/trim contract tests (registration #1-#4).

The adapter-joined session package is the single alignment authority
(join_manifest.json): every stream is loaded on shared frame ids, fake/invalid
frames are dropped by the adapter, one shared keep mask applies the per-segment
end trim (upstream ``[2:-2]``) to every modality exactly once, and Savitzky-Golay
filtering never crosses a fake gap.

Run directly:

    /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python \
        tests/test_dataset_join_contract.py

The native model env is ``mmvp`` (torch + icecream + open3d + human_body_prior);
``touch_gait`` lacks icecream/open3d, so this file installs minimal stubs for
them (the depth/point-cloud math is patched out: it is not under test here).

Real-session assertions run against
``model-input://PoseTransOpt/adapter_v1/20260804/S5/S5011`` when it exists and
report a skip otherwise.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np

POSETRANSOPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = POSETRANSOPT_ROOT.parents[2]
WORKSPACE = Path(os.environ.get("ANYSOLE_WORKSPACE", REPO_ROOT / "AnysoleWorkspace"))
if str(POSETRANSOPT_ROOT) not in sys.path:
    sys.path.insert(0, str(POSETRANSOPT_ROOT))


def _install_light_stubs():
    """Provide the native-env extras that the slim test env may lack."""
    for name in ("icecream", "open3d"):
        try:
            __import__(name)
        except ModuleNotFoundError:
            module = types.ModuleType(name)
            if name == "icecream":
                module.ic = lambda *args, **kwargs: None
            sys.modules[name] = module


_install_light_stubs()

from lib.dataset import dataset_mmvp  # noqa: E402
from lib.dataset.dataset_mmvp import (  # noqa: E402
    TRIM_END,
    Dataset,
    _frame_index,
    _savgol_per_segment,
    _trimmed_segments,
)

REAL_SESSION = WORKSPACE / "model_inputs/PoseTransOpt/adapter_v1/20260804/S5/S5011"


def _no_point_cloud(*args, **kwargs):
    """Stub the scene unprojection: the join/trim contract needs no depth math."""
    return np.zeros((8, 8, 3), dtype=np.float64), None


dataset_mmvp.depth_to_pointcloud = _no_point_cloud


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _save_sidecar(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, payload, allow_pickle=True)


def _base_cfg(base: Path, method=None, **task_overrides):
    task = {
        "input_path_base": str(base),
        "scene_rgbd": str(base / "template_scene_rgbd.npy"),
        "image_width": 1624,
        "image_height": 1240,
        "focal_length": 1394.0,
        "depth_scale": 1.0,
        "scale": 0.97,
        "thres_contact": 15,
        "fps": 40.0,
        "max_frames": 0,
    }
    task.update(task_overrides)
    method_cfg = {"pose_filter": True, "kp_filter": True, "kp_score": True}
    method_cfg.update(method or {})
    return {"task": task, "method": method_cfg}


def _build_session(base: Path, npz_frames, joined_frames, segments_frames,
                   contact_keep_prob=0.5, seed=0):
    """Write a synthetic adapter_v1 session package for one subject.

    ``npz_frames``/``joined_frames`` are frame id lists; ``segments_frames``
    follows the adapter convention of *inclusive* frame-id ends (the reporting
    view ``segments_frames``), while the ``segments`` field it writes is the
    half-open index-space view the dataset consumes.  The CLIFF npz carries one
    row per canonical frame (its own frame_id), the keypoint/contact sidecars
    only the joined frames.
    """
    rng = np.random.default_rng(seed)
    base.mkdir(parents=True, exist_ok=True)
    n = len(npz_frames)
    pose = np.zeros((n, 72), dtype=np.float32)
    pose[:, 0] = np.asarray(npz_frames, dtype=np.float32)  # marker: row -> frame id
    shape = np.zeros((n, 10), dtype=np.float32)
    shape[:, 0] = np.asarray(npz_frames, dtype=np.float32)  # marker for beta mean
    global_t = np.zeros((n, 3), dtype=np.float32)
    global_t[:, 0] = np.asarray(npz_frames, dtype=np.float32)  # marker for retrieval
    np.savez(base / "CLIFF_results.npz", frame_id=np.asarray(npz_frames, dtype=np.int64),
             pose=pose, shape=shape, global_t=global_t,
             valid=np.ones(n, dtype=np.uint8))

    for index, frame_id in enumerate(joined_frames):
        # keypoint x marks the segment boundary: constant per segment, so a
        # cross-gap filter would visibly bleed.
        segment = 0 if index < len(joined_frames) // 2 else 1
        keypoints = np.zeros((26, 2), dtype=np.float32)
        keypoints[:, 0] = 0.0 if segment == 0 else 1000.0
        keypoints[:, 1] = float(frame_id)
        _save_sidecar(base / "keypoints" / f"{frame_id:06d}.npy",
                      {"frame_id": frame_id, "keypoints": keypoints,
                       "keypoint_scores": np.ones(26, dtype=np.float32)})

        prob = (rng.random((2, 96)) < contact_keep_prob).astype(np.float32)
        _save_sidecar(base / "pred_contact_smpl" / f"{frame_id:06d}.npy",
                      {"frame_id": frame_id,
                       "contact_smpl": {"pred": prob, "gt": (prob > 0.5).astype(np.float32)}})

    index_segments = []
    cursor = 0
    for start, end in segments_frames:
        span = end - start + 1  # inclusive frame-id ends, as the adapter writes
        index_segments.append([cursor, cursor + span])
        cursor += span
    final_frames = []
    for cursor_start, cursor_end in index_segments:
        final_frames.extend(joined_frames[cursor_start + TRIM_END:cursor_end - TRIM_END])
    manifest = {
        "adapter_version": "adapter_v1",
        "session_id": "SYNTH",
        "cliff_npz": str(base / "CLIFF_results.npz"),
        "joined_frames": list(joined_frames),
        "segments": index_segments,
        "segments_frames": [list(seg) for seg in segments_frames],
        "final_frames": list(final_frames),
        "trim": {"frames_per_segment_end": TRIM_END},
    }
    (base / "join_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------
# #1 frame-id retrieval
# --------------------------------------------------------------------------

def test_frame_index_lookup_and_mismatch():
    npz_frame = np.arange(10, 100, dtype=np.int64)
    query = np.array([10, 11, 55, 99], dtype=np.int64)
    assert _frame_index(npz_frame, query).tolist() == [0, 1, 45, 89]
    for bad in (np.array([9]), np.array([100]), np.array([55, 100])):
        try:
            _frame_index(npz_frame, bad)
        except ValueError:
            continue
        raise AssertionError(f"missing frame id {bad} must raise")


def test_missing_frame_id_in_npz_raises(tmp_path):
    base = tmp_path / "S1"
    # the joined grid contains frame 20, the CLIFF npz has no row for it
    _build_session(base, npz_frames=[f for f in range(40) if f != 20],
                   joined_frames=list(range(40)), segments_frames=[[0, 39]])
    try:
        Dataset(_base_cfg(base, max_frames=8))
    except ValueError as error:
        assert "joined frame ids do not match" in str(error), str(error)
        return
    raise AssertionError("a joined frame id absent from the CLIFF npz must raise")


# --------------------------------------------------------------------------
# #3 shared keep mask + per-segment trim
# --------------------------------------------------------------------------

def test_trimmed_segments_index_math():
    assert _trimmed_segments([[0, 20], [20, 45]]) == [[0, 16], [16, 37]]
    assert _trimmed_segments([[0, 4]]) == [[0, 0]]


def test_final_frames_mismatch_raises(tmp_path):
    base = tmp_path / "S2"
    manifest = _build_session(base, npz_frames=list(range(60)),
                              joined_frames=list(range(0, 40)),
                              segments_frames=[[0, 39]])
    manifest["final_frames"] = manifest["final_frames"][:-1]  # 35 instead of 36
    (base / "join_manifest.json").write_text(json.dumps(manifest))
    try:
        Dataset(_base_cfg(base, max_frames=8))
    except ValueError as error:
        assert "final_frames" in str(error), str(error)
        return
    raise AssertionError("post-trim frame ids must match the adapter final_frames")


def test_single_keep_mask_aligns_every_modality(tmp_path):
    base = tmp_path / "S3"
    joined = list(range(10, 70))
    manifest = _build_session(base, npz_frames=list(range(0, 80)),
                              joined_frames=joined, segments_frames=[[10, 69]])
    dataset = Dataset(_base_cfg(base, method={"pose_filter": False}))
    expected = np.asarray(manifest["final_frames"], dtype=np.int64)
    # one trim, applied once, to every modality
    assert dataset.frame_ids.tolist() == expected.tolist()
    for name in ("pose", "betas", "trans_origin", "target_halpe_fil",
                 "target_halpe_score", "contact", "vertex_contact"):
        assert getattr(dataset, name).shape[0] == expected.size, name
    assert dataset.contact.shape == (expected.size, 4)
    assert dataset.vertex_contact.shape == (expected.size, 2, 96)
    # frame-id retrieval (not glob order + [2:-2]): trans_origin row i is the
    # CLIFF row of frame_ids[i], and the contact sidecar row i is that frame too
    assert np.array_equal(dataset.trans_origin[:, 0], expected.astype(np.float32))
    for index, frame_id in enumerate(expected.tolist()):
        payload = np.load(base / "pred_contact_smpl" / f"{frame_id:06d}.npy",
                          allow_pickle=True).item()
        assert payload["frame_id"] == frame_id
        assert dataset.frame_ids[index] == frame_id
    # the native positional mapping is provably different (no silent 2-frame shift
    # between the pose stream and the contact stream)
    assert not np.array_equal(dataset.frame_ids, np.asarray(joined[:expected.size]))


# --------------------------------------------------------------------------
# #2 beta mean over this sequence's rows only
# --------------------------------------------------------------------------

def test_beta_mean_uses_joined_rows_only(tmp_path):
    base = tmp_path / "S4"
    npz_frames = list(range(0, 60))
    joined = list(range(20, 60))  # first 20 rows are bystander-only frames
    _build_session(base, npz_frames=npz_frames, joined_frames=joined,
                   segments_frames=[[20, 59]])
    dataset = Dataset(_base_cfg(base, method={"pose_filter": False, "kp_filter": False}))
    npz = np.load(base / "CLIFF_results.npz")
    npz_wide_mean = float(npz["shape"].mean(axis=0)[0])
    final_mean = float(npz["shape"][20 + TRIM_END:60 - TRIM_END].mean(axis=0)[0])
    got = float(dataset.betas[0, 0])
    # mean over the rows this sequence actually loads (joined 20..59, trimmed
    # 22..57), never over the npz-wide set that still contains bystander rows
    assert abs(got - final_mean) < 1e-4, (got, final_mean)
    assert abs(got - npz_wide_mean) > 1e-3, (got, npz_wide_mean)
    assert np.allclose(dataset.betas, dataset.betas[0])


# --------------------------------------------------------------------------
# #4 savgol per contiguous segment
# --------------------------------------------------------------------------

def test_savgol_does_not_cross_fake_gap():
    values = np.concatenate([np.zeros(30), np.full(30, 1000.0)])
    segments = [(0, 30), (30, 60)]
    per_segment = _savgol_per_segment(values, 15, 3, segments)
    whole_array = dataset_mmvp.savgol_filter(values, 15, 3, axis=0)
    # inside each segment the filter is the native one
    assert np.allclose(per_segment[:30], dataset_mmvp.savgol_filter(values[:30], 15, 3))
    assert np.allclose(per_segment[30:], dataset_mmvp.savgol_filter(values[30:], 15, 3))
    # a cross-gap filter would drag the last 14 samples of segment 1 upward
    assert not np.allclose(per_segment, whole_array)
    assert np.allclose(per_segment[:30], 0.0), per_segment[:5]
    assert np.allclose(per_segment[30:], 1000.0), per_segment[-5:]


def test_savgol_short_segment_raises():
    try:
        _savgol_per_segment(np.zeros(20), 17, 5, [(0, 10), (10, 20)])
    except ValueError as error:
        assert "savgol window" in str(error), str(error)
        return
    raise AssertionError("a segment shorter than the savgol window must raise")


def test_keypoint_filter_is_per_segment(tmp_path):
    base = tmp_path / "S5"
    joined = list(range(0, 24)) + list(range(100, 124))  # one fake gap
    _build_session(base, npz_frames=list(range(0, 40)) + list(range(100, 140)),
                   joined_frames=joined,
                   segments_frames=[[0, 23], [100, 123]])
    dataset = Dataset(_base_cfg(base, method={"pose_filter": False, "kp_filter": True, "kp_score": False}))
    assert dataset.segments == [[0, 20], [20, 40]]
    assert dataset.target_halpe_fil.shape == (40, 26, 2)
    # segment 0 stays at 0, segment 1 stays at 1000: no leak across the gap
    assert np.allclose(dataset.target_halpe_fil[:20, :, 0], 0.0), dataset.target_halpe_fil[17:21, 0, 0]
    assert np.allclose(dataset.target_halpe_fil[20:, :, 0], 1000.0), dataset.target_halpe_fil[18:23, 0, 0]
    whole_array = dataset_mmvp.savgol_filter(
        np.concatenate([np.zeros(20), np.full(20, 1000.0)]), 15, 3)
    assert not np.allclose(dataset.target_halpe_fil[:, 0, 0], whole_array)
    assert np.all(dataset.target_halpe_score == 1.0)


def test_pose_filter_is_per_segment(tmp_path):
    base = tmp_path / "S6"
    joined = list(range(0, 24)) + list(range(100, 124))
    _build_session(base, npz_frames=list(range(0, 40)) + list(range(100, 140)),
                   joined_frames=joined,
                   segments_frames=[[0, 23], [100, 123]])
    dataset = Dataset(_base_cfg(base, method={"pose_filter": True, "kp_filter": False}))
    assert dataset.segments == [[0, 20], [20, 40]]
    assert dataset.pose.shape == (40, 24, 3, 3)
    assert dataset.trans_origin.shape == (40, 3)
    # NOTE: the synthetic global_t has one non-zero column, so this only checks
    # the shape/no-crash contract of the per-segment trans filter (windows 17/5,
    # 17/5, 17/3 as in the native loader).
    assert np.isfinite(dataset.trans_origin).all()
    assert np.isfinite(dataset.pose).all()


# --------------------------------------------------------------------------
# real T1 session (same assertion battery on adapter_v1 output)
# --------------------------------------------------------------------------

def test_real_session_join_contract():
    if not (REAL_SESSION / "join_manifest.json").is_file():
        print(f"  skip: no T1 session at {REAL_SESSION}")
        return
    manifest = json.loads((REAL_SESSION / "join_manifest.json").read_text())
    dataset = Dataset(_base_cfg(REAL_SESSION, method={"pose_filter": False, "kp_filter": False}))
    expected = np.asarray(manifest["final_frames"], dtype=np.int64)
    assert dataset.frame_ids.tolist() == expected.tolist()
    assert dataset.pose.shape[0] == expected.size
    assert dataset.contact.shape == (expected.size, 4)
    assert dataset.vertex_contact.shape == (expected.size, 2, 96)
    npz = np.load(REAL_SESSION / "CLIFF_results.npz", allow_pickle=True)
    npz_frame = np.asarray(npz["frame_id"], dtype=np.int64)
    order = _frame_index(npz_frame, expected)
    assert np.allclose(dataset.trans_origin, npz["global_t"][order], atol=1e-5)
    # #2: beta mean over this sequence's joined rows only (load_pose runs
    # before the shared keep mask, so the mean is taken over the joined grid),
    # never over every row of the mask-bbox CLIFF npz.  The authoritative grid
    # is joined_frames: segments_frames is an inclusive-end reporting view.
    joined = np.asarray(manifest["joined_frames"], dtype=np.int64)
    joined_order = _frame_index(npz_frame, joined)
    joined_mean = float(npz["shape"][joined_order].mean(axis=0)[0])
    npz_wide_mean = float(npz["shape"].mean(axis=0)[0])
    got = float(dataset.betas[0, 0])
    assert abs(got - joined_mean) < 1e-4, (got, joined_mean)
    assert abs(got - npz_wide_mean) > 1e-4, (got, npz_wide_mean)


def test_template_scene_is_readable_container():
    """The adapter writes template_scene_rgbd.npy as an npz container."""
    path = REAL_SESSION / "template_scene_rgbd.npy"
    if not path.is_file():
        print(f"  skip: no template scene at {path}")
        return
    from lib.utils.depth_utils import _load_rgbd

    rgb, depth = _load_rgbd(path)
    assert rgb.shape == (1240, 1624, 3), rgb.shape
    assert depth.shape == (1240, 1624), depth.shape
    assert np.isfinite(np.asarray(depth)).all()


TESTS = (
    test_frame_index_lookup_and_mismatch,
    test_missing_frame_id_in_npz_raises,
    test_trimmed_segments_index_math,
    test_final_frames_mismatch_raises,
    test_single_keep_mask_aligns_every_modality,
    test_beta_mean_uses_joined_rows_only,
    test_savgol_does_not_cross_fake_gap,
    test_savgol_short_segment_raises,
    test_keypoint_filter_is_per_segment,
    test_pose_filter_is_per_segment,
    test_real_session_join_contract,
    test_template_scene_is_readable_container,
)


if __name__ == "__main__":
    import tempfile

    for test in TESTS:
        if test in (test_frame_index_lookup_and_mismatch, test_trimmed_segments_index_math,
                    test_savgol_does_not_cross_fake_gap, test_savgol_short_segment_raises,
                    test_real_session_join_contract, test_template_scene_is_readable_container):
            test()
        else:
            with tempfile.TemporaryDirectory() as folder:
                test(Path(folder))
        print(f"ok: {test.__name__}")
    print("test_dataset_join_contract: all passed")
