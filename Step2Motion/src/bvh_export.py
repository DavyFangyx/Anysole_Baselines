from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation


DEFAULT_JOINT_NAMES = [
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "Spine",
    "Spine1",
    "Spine3",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
]


def joint_names_for_count(n_joints: int) -> list[str]:
    if n_joints == len(DEFAULT_JOINT_NAMES):
        return list(DEFAULT_JOINT_NAMES)
    return ["Hips"] + ["Joint_%02d" % i for i in range(1, n_joints)]


def _safe_joint_names(names: list[str]) -> list[str]:
    safe = []
    used: set[str] = set()
    for index, name in enumerate(names):
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip()) or "Joint_%02d" % index
        original = value
        suffix = 1
        while value in used:
            value = "%s_%d" % (original, suffix)
            suffix += 1
        used.add(value)
        safe.append(value)
    return safe


def write_bvh(
    path: str | Path,
    joint_names: list[str],
    parents: np.ndarray,
    offsets: np.ndarray,
    local_rotations_wxyz: np.ndarray,
    root_positions: np.ndarray,
    frame_time: float,
) -> None:
    """Write a self-contained BVH from the current dataset skeleton."""
    parents = np.asarray(parents, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.float64)
    rotations = np.asarray(local_rotations_wxyz, dtype=np.float64)
    root_positions = np.asarray(root_positions, dtype=np.float64)
    n_joints = len(parents)
    if offsets.shape != (n_joints, 3):
        raise ValueError("offsets must have shape (%d, 3), got %s" % (n_joints, offsets.shape))
    if rotations.ndim != 3 or rotations.shape[1:] != (n_joints, 4):
        raise ValueError("rotations must have shape (frames, %d, 4), got %s" % (n_joints, rotations.shape))
    if root_positions.shape != (rotations.shape[0], 3):
        raise ValueError("root_positions must have shape (%d, 3), got %s" % (rotations.shape[0], root_positions.shape))
    if len(joint_names) != n_joints:
        raise ValueError("joint_names has %d entries for %d joints" % (len(joint_names), n_joints))
    if frame_time <= 0:
        raise ValueError("frame_time must be positive")

    roots = np.flatnonzero(parents < 0).tolist()
    if len(roots) != 1:
        raise ValueError("BVH requires exactly one root, got %d" % len(roots))
    root = roots[0]
    children: list[list[int]] = [[] for _ in range(n_joints)]
    for child, parent in enumerate(parents.tolist()):
        if parent < 0:
            continue
        if parent >= n_joints or parent == child:
            raise ValueError("invalid parent %d for joint %d" % (parent, child))
        children[parent].append(child)

    names = _safe_joint_names(joint_names)
    hierarchy = ["HIERARCHY"]
    channel_order: list[int] = []

    def emit_joint(joint: int, depth: int) -> None:
        indent = "\t" * depth
        kind = "ROOT" if joint == root else "JOINT"
        hierarchy.append("%s%s %s" % (indent, kind, names[joint]))
        hierarchy.append("%s{" % indent)
        off = offsets[joint]
        hierarchy.append("%s\tOFFSET %.9f %.9f %.9f" % (indent, off[0], off[1], off[2]))
        if joint == root:
            hierarchy.append(
                "%s\tCHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation" % indent
            )
        else:
            hierarchy.append("%s\tCHANNELS 3 Xrotation Yrotation Zrotation" % indent)
        channel_order.append(joint)
        if children[joint]:
            for child in children[joint]:
                emit_joint(child, depth + 1)
        else:
            hierarchy.extend(
                [
                    "%s\tEnd Site" % indent,
                    "%s\t{" % indent,
                    "%s\t\tOFFSET 0.000000000 0.000000000 0.000000000" % indent,
                    "%s\t}" % indent,
                ]
            )
        hierarchy.append("%s}" % indent)

    emit_joint(root, 0)
    if len(channel_order) != n_joints:
        missing = sorted(set(range(n_joints)) - set(channel_order))
        raise ValueError("skeleton contains joints disconnected from the root: %s" % missing)

    xyzw = np.concatenate([rotations[..., 1:], rotations[..., :1]], axis=-1)
    # Standard BVH channel convention: rotations are applied in channel order
    # (X then Y then Z = Rx@Ry@Rz).  scipy 'zyx' returns the angles ordered
    # (z, y, x), so the triple is reversed before writing; writing 'xyz'
    # values made the public evaluator (bvh_aligner) reconstruct different
    # rotations (measured ~38 mm mean world error on S13073).
    eulers = SciRotation.from_quat(xyzw.reshape(-1, 4)).as_euler("zyx", degrees=True)
    eulers = eulers[..., ::-1]
    eulers = eulers.reshape(rotations.shape[0], n_joints, 3)
    motion = ["MOTION", "Frames: %d" % rotations.shape[0], "Frame Time: %.9f" % frame_time]
    for frame in range(rotations.shape[0]):
        values: list[float] = []
        for joint in channel_order:
            if joint == root:
                values.extend(root_positions[frame].tolist())
            values.extend(eulers[frame, joint].tolist())
        motion.append(" ".join("%.9f" % value for value in values))

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(hierarchy + motion) + "\n")
