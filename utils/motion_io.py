"""Unified motion reader (canonical home: ``Baselines/utils``).

AnySole motion artifacts use the standard SMPL NPZ contract.  BVH remains
supported for Step2Motion and legacy result folders; callers do not need a
format flag because the archive suffix/keys are inspected automatically.

``results_display/script/utils/motion_io.py`` is a re-export shim of this
module so display-side callers keep their import paths.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# Bootstrap the repository root (for the ``anysole`` package) and this
# package's directory regardless of the caller's launch mode.
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parents[2]  # utils/ -> Baselines/ -> gait repo root
for entry in (REPO_ROOT,):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from anysole.data.smpl_io import load_smpl, resolve_smpl_path, smpl_vertices_from_archive_params
from anysole.utils.geometry import fk_pose6d_np
from anysole.types import JOINT_NAMES, SMPL_ROOTS
from Baselines.utils.bvh_aligner import parse_bvh_aligner


# Historical Skeleton3 order used only by Step2Motion/legacy result files.
# Keeping it here (not in the AnySole model package) prevents the AnySole
# model package from acquiring a second motion protocol again.
LEGACY_BVH_NAMES = (
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
)


def joints_to_meters(joints):
    pts = np.asarray(joints, dtype=np.float32)
    return pts * 0.01 if pts.size and float(np.ptp(pts[0], axis=0).max()) > 5.0 else pts


def smpl_yup_to_display(joints):
    """Native SMPL (x,y-up,z) -> renderer/common z-up (x,-z,y)."""
    pts = np.asarray(joints, dtype=np.float64)
    return np.stack((pts[..., 0], -pts[..., 2], pts[..., 1]), axis=-1)


def interp_joints(joints, frame_time, query_t):
    src_t = np.arange(joints.shape[0], dtype=np.float64) * float(frame_time)
    clipped = np.clip(np.asarray(query_t, dtype=np.float64), src_t[0], src_t[-1])
    out = np.empty((len(clipped),) + joints.shape[1:], dtype=np.float64)
    for joint_i in range(joints.shape[1]):
        for axis_i in range(3):
            out[:, joint_i, axis_i] = np.interp(clipped, src_t, joints[:, joint_i, axis_i])
    return out.astype(np.float32)


def detect_motion_format(path: Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".bvh":
        return "bvh"
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported motion file: {path}")
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
    if {"poses", "trans"}.issubset(keys) or {"root_orient", "pose_body", "trans"}.issubset(keys):
        return "smpl"
    raise ValueError(f"Cannot identify motion NPZ format from keys {sorted(keys)}: {path}")


def load_motion(path: Path, query_t: Optional[np.ndarray] = None) -> dict:
    """Load one motion file and return ``joints`` (T,J,3), ``parents``, format."""
    path = Path(path)
    fmt = detect_motion_format(path)
    if fmt == "bvh":
        parsed = parse_bvh_aligner(path, trim_leading_seconds=0.0)
        joints = joints_to_meters(parsed["joints"])
        if query_t is not None:
            joints = joints_to_meters(interp_joints(parsed["joints"], parsed["frame_time"], query_t))
        return {
            "joints": joints,
            "rotations": np.asarray(parsed["rotations"], dtype=np.float32),
            "parents": np.asarray(parsed["parents"], dtype=np.int64),
            "names": tuple(parsed["names"]),
            "format": fmt,
        }
    if fmt == "smpl":
        motion = load_smpl(path, query_t=query_t)
        joints = fk_pose6d_np(
            motion["pose_6d"], motion["trans_m"],
            motion["offsets_m"], motion["parents"],
        )
        return {
            "joints": smpl_yup_to_display(joints),
            "rotations": Rotation.from_rotvec(
                np.asarray(motion["poses_aa"], dtype=np.float64).reshape(-1, 24, 3).reshape(-1, 3)
            ).as_matrix().reshape(-1, 24, 3, 3).astype(np.float32),
            "parents": motion["parents"],
            "names": tuple(JOINT_NAMES),
            "format": fmt,
        }
    raise AssertionError(f"unhandled motion format: {fmt}")


def load_rotations(path: Path, query_t: Optional[np.ndarray] = None) -> dict:
    """Load global rotation matrices and semantic names on the requested grid."""
    path = Path(path)
    if path.suffix.lower() == ".bvh":
        parsed = parse_bvh_aligner(path, trim_leading_seconds=0.0)
        rotations = np.asarray(parsed["local_rotations"], dtype=np.float64)
        if query_t is not None:
            source_t = np.arange(len(rotations), dtype=np.float64) * float(parsed["frame_time"])
            query = np.clip(np.asarray(query_t, dtype=np.float64), source_t[0], source_t[-1])
            sampled = np.empty((len(query),) + rotations.shape[1:], dtype=np.float64)
            for j in range(rotations.shape[1]):
                sampled[:, j] = Slerp(source_t, Rotation.from_matrix(rotations[:, j]))(query).as_matrix()
            rotations = sampled
        return {"rotations": rotations.astype(np.float32), "names": tuple(parsed["names"]), "format": "bvh"}
    with np.load(path, allow_pickle=True) as data:
        if 'joint_rotations' in data:
            rotations = np.asarray(data['joint_rotations'], dtype=np.float32)
            names = tuple(str(v) for v in data['joint_names'])
            if rotations.shape[1:] != (len(names), 3, 3):
                raise ValueError('Malformed canonical joint_rotations: ' + str(path))
            return dict(rotations=rotations, names=names, format='bvh')
        if "poses" in data and "vertices_world" in data:
            poses = np.asarray(data["poses"], dtype=np.float64).reshape(-1, 24, 3)
            rotations = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
            if "joint_names" in data:
                names = tuple(
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in np.asarray(data["joint_names"]).reshape(-1)
                )
            else:
                names = tuple(JOINT_NAMES)
            return {"rotations": rotations.astype(np.float32), "names": names, "format": "smpl"}
    motion = load_smpl(path, query_t=query_t)
    rotations = Rotation.from_rotvec(
        np.asarray(motion["poses_aa"], dtype=np.float64).reshape(-1, 24, 3).reshape(-1, 3)
    ).as_matrix().reshape(-1, 24, 3, 3)
    return {"rotations": rotations.astype(np.float32), "names": tuple(motion["joint_names"]), "format": "smpl"}


def _smpl_vertices(motion: dict) -> np.ndarray:
    """Run the canonical neutral SMPL surface model for one loaded archive."""
    return smpl_vertices_from_archive_params(
        motion["poses_aa"], motion["model_trans_m"], motion["betas"]
    )


def load_surface(path: Path, query_t: Optional[np.ndarray] = None) -> dict | None:
    """Load vertices and rotations when the prediction has an SMPL surface contract.

    BVH-only outputs return ``None``: surface-only metrics are then explicitly
    unsupported rather than approximated from joints.
    """
    path = Path(path)
    if path.suffix.lower() != ".npz":
        return None
    with np.load(path, allow_pickle=True) as data:
        if "vertices_world" in data:
            vertices = np.asarray(data["vertices_world"], dtype=np.float32)
            rotations = np.asarray(data["poses"], dtype=np.float32).reshape(-1, 24, 3) if "poses" in data else None
            return {"vertices": vertices, "rotations": rotations}
        keys = set(data.files)
        is_smpl = {"poses", "trans"}.issubset(keys) or {"root_orient", "pose_body", "trans"}.issubset(keys)
    if not is_smpl:
        return None
    motion = load_smpl(path, query_t=query_t)
    return {
        "vertices": smpl_yup_to_display(_smpl_vertices(motion)),
        "rotations": np.asarray(motion["poses_aa"], dtype=np.float32).reshape(-1, 24, 3),
    }


def load_session_gt(seq_dir: Path, n_frames: int, fps: float = 40.0) -> dict:
    meta = json.loads((Path(seq_dir) / "align_meta.json").read_text())
    t_grid = float(meta["visual_start_s"]) + np.arange(n_frames, dtype=np.float64) / float(fps)
    t_mocap = t_grid - float(meta["offset_s"])
    try:
        smpl = resolve_smpl_path(meta, tuple(SMPL_ROOTS))
        result = load_motion(smpl, query_t=t_mocap)
        result["source_path"] = str(smpl)
    except FileNotFoundError:
        # Legacy/Step2Motion sessions may only have BVH.  This fallback is
        # confined to the display reader; AnySole training/inference never
        # takes it.
        bvh = Path(str(meta.get("bvh_path", ""))).expanduser()
        if not bvh.is_file():
            raise
        result = load_motion(bvh, query_t=t_mocap)
        result["source_path"] = str(bvh)
    # Extra keys let mesh renderers reuse the exact same time grid/archive:
    # ``format`` (smpl/bvh) already comes from load_motion.
    result["t_mocap"] = t_mocap
    return result
