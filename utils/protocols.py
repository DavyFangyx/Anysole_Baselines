"""Data protocols shared by the baseline evaluation side.

Native motion protocols (评估整改任务 01 §4.1, common19 deleted):

- ``smpl24``: the native SMPL-24 skeleton.  The joint names/parents come from
  ``anysole.types`` (the main model's own protocol module) so there is exactly
  one SMPL-24 table in the repository.
- ``bvh23``: the native BVH-23 Skeleton3 semantic skeleton (23 joints; the
  five EndSite markers of the raw files are dropped by ``gt_loading.bvh_joints``).
  The table lives here — BVH is the baseline-side format, and it must not leak
  into the AnySole model package.

``smpl24-native`` / ``bvh23-native`` are the protocol identifiers recorded in
results; the bare tokens are the internal branch keys.  ``pressure`` appears
in the registry for V2T rows.
"""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]  # utils/ -> Baselines/ -> repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from anysole.types import JOINT_NAMES, JOINT_PARENTS
from anysole.utils.metrics import pelvis_align

PROTOCOLS = ("smpl24", "bvh23")
PROTOCOL_LABELS = {"smpl24": "smpl24-native", "bvh23": "bvh23-native"}


def normalize_protocol(protocol: str) -> str:
    value = str(protocol or "").lower().replace("_", "-")
    if value.startswith("smpl"):
        return "smpl24"
    if value.startswith("bvh"):
        return "bvh23"
    if value.startswith("pressure"):
        return "pressure"
    raise ValueError(f"unsupported protocol {protocol!r} (expected smpl24/bvh23/pressure)")


# SMPL-24: single source of truth is the main model's protocol module.
SMPL_JOINT_NAMES = tuple(JOINT_NAMES)
SMPL_JOINT_PARENTS = tuple(int(p) for p in JOINT_PARENTS)

# BVH-23 Skeleton3 semantic order (verified against Step2Motion BVH exports,
# e.g. S10103_gen.bvh).
BVH_JOINT_NAMES = (
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
)

# Skeleton3 hierarchy of the native BVH-23: shoulders hang from Spine3; the
# five EndSite markers (children of Head/LeftHand/RightHand/LeftToeBase/
# RightToeBase) are dropped by bvh_joints().  Prefer the parents returned by
# bvh_joints() (parsed from the file) over this table.
BVH_JOINT_PARENTS = (
    -1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 4, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21,
)

# Pelvis alignment per protocol: SMPL-24 pelvis is the mean of left/right hip
# (native indices 1/2); BVH-23 pelvis is the single Hips joint (index 0).
SMPL_PELVIS_INDICES = (1, 2)
BVH_PELVIS_INDICES = (0,)
PROTOCOL_PELVIS = {"smpl24": SMPL_PELVIS_INDICES, "bvh23": BVH_PELVIS_INDICES}

# Foot joints for the public joint-based foot_sliding_mm (semantically
# matched: SMPL ankle <-> BVH Foot, SMPL foot <-> BVH ToeBase).
PROTOCOL_FOOT_NAMES = {
    "smpl24": ("left_ankle", "right_ankle", "left_foot", "right_foot"),
    "bvh23": ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"),
}

# Root-orientation forward axis per protocol (both pred and GT rotations of
# one protocol live in the same frame, so the two must match):
#   smpl24: rotations are native SMPL (y-up, body forward +Z)        -> axis 2
#   bvh23:  rotations are the z-up display frame, body forward -Y
#           (verified 2026-09-26 against SMPL GT heading on S13013:
#           median circular diff 4.4 deg; the common 180 deg sign flip
#           cancels exactly in the circular-difference error formulas) -> axis 1
PROTOCOL_FORWARD_AXIS = {"smpl24": 2, "bvh23": 1}

SMPL_NATIVE_EDGES = tuple(
    (int(i), int(p)) for i, p in enumerate(SMPL_JOINT_PARENTS) if int(p) >= 0
)
BVH_NATIVE_EDGES = tuple(
    (i, p) for i, p in enumerate(BVH_JOINT_PARENTS) if p >= 0
)


def native_edges(names_or_protocol, parents=None) -> list[tuple[int, int]]:
    """Native skeleton stick edges as (child, parent) index pairs.

    With ``parents`` (e.g. from :func:`Baselines.utils.gt_loading.bvh_joints`)
    the edges are derived from the parsed file; otherwise the built-in SMPL-24
    tree or the Skeleton3 hierarchy is used.  The common19 edge set no longer
    exists; display layouts and GT panels consume this interface instead.
    """
    if parents is not None:
        return [
            (int(i), int(p))
            for i, p in enumerate(np.asarray(parents, dtype=np.int64))
            if int(p) >= 0
        ]
    if isinstance(names_or_protocol, str):
        protocol = normalize_protocol(names_or_protocol)
        return list(SMPL_NATIVE_EDGES if protocol == "smpl24" else BVH_NATIVE_EDGES)
    names = tuple(str(n) for n in np.asarray(names_or_protocol).reshape(-1))
    if set(BVH_JOINT_NAMES).issubset(names):
        return list(BVH_NATIVE_EDGES)
    if set(SMPL_JOINT_NAMES).issubset(names):
        return list(SMPL_NATIVE_EDGES)
    raise ValueError(f"unrecognized skeleton names: {names}")


def native_frame_mpjpe_mm(
    pred_native: np.ndarray,
    gt_native: np.ndarray,
    *,
    pelvis_indices: tuple[int, ...] | None = None,
    protocol: str | None = None,
) -> np.ndarray:
    """Per-frame pelvis-relative MPJPE in mm over one native skeleton.

    No joint selection happens here — the caller passes a prediction and its
    paired GT already in the same native joint order.  ``pelvis_indices``
    defaults from ``protocol`` (same definition as the scalar ``metrics()``).
    """
    if pelvis_indices is None:
        if protocol is None:
            raise ValueError("native_frame_mpjpe_mm needs pelvis_indices or protocol")
        pelvis_indices = PROTOCOL_PELVIS[normalize_protocol(protocol)]
    pred = np.asarray(pred_native, dtype=np.float64)
    gt = np.asarray(gt_native, dtype=np.float64)
    n = min(pred.shape[0], gt.shape[0])
    p_aligned, _ = pelvis_align(pred[:n], pelvis_indices=pelvis_indices)
    g_aligned, _ = pelvis_align(gt[:n], pelvis_indices=pelvis_indices)
    return np.linalg.norm(p_aligned - g_aligned, axis=-1).mean(axis=-1) * 1000.0
