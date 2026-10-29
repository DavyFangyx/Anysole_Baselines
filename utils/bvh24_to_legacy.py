"""BVH24 (job 2026-10-24) -> legacy BVH-23 channel conversion.

The 2026-10-24 upstream job exports ``motion_smpl24_blender_world_m.bvh``
(SMPL-24 hierarchy, Blender +Z up/+Y forward, metres, ZXY degrees, full world
root motion) next to the SMPL npz.  Consumers that still speak the legacy
Step2Motion BVH convention (23 CMU joints, y-up, cm, YXZ degrees, 72
channels) call :func:`parse_legacy_channels`, which returns that exact
contract from either file kind:

* legacy files (``ROOT Hips``, 23 joints) parse verbatim;
* BVH24 files are converted: FK in the Blender frame, basis change to the
  SMPL y-up frame, positional remap onto the 23 CMU joints, re-encoding as
  legacy channels (root pos/rotation + per-joint YXZ local eulers).

The remap is positional.  Spine1 splits the rigid spine1->spine2 bone at its
midpoint and carries identity local rotation; every other legacy joint maps
onto a real SMPL landmark (LeftFoot <- left_ankle, LeftToeBase <- left_foot,
LeftShoulder <- left_collar, LeftHand <- left_wrist, etc.).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp

LEGACY_JOINT_NAMES = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
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
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
]
LEGACY_PARENTS = np.asarray(
    [-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 4, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21],
    dtype=np.int64,
)
N_LEGACY = len(LEGACY_JOINT_NAMES)

# smpl_to_blender from conversion_manifest.coordinate_contract; symmetric and
# self-inverse, so the same matrix maps Blender vectors back to SMPL.
SMPL2BLENDER = np.asarray([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

# legacy joint -> ((smpl joint name, weight), ...) for world positions.
_POS_MAP: dict[str, tuple[tuple[str, float], ...]] = {
    "Hips": (("pelvis", 1.0),),
    "Spine": (("spine1", 1.0),),
    "Spine1": (("spine1", 0.5), ("spine2", 0.5)),
    "Spine2": (("spine2", 1.0),),
    "Spine3": (("spine3", 1.0),),
    "Neck": (("neck", 1.0),),
    "Head": (("head", 1.0),),
    "LeftShoulder": (("left_collar", 1.0),),
    "LeftArm": (("left_shoulder", 1.0),),
    "LeftForeArm": (("left_elbow", 1.0),),
    "LeftHand": (("left_wrist", 1.0),),
    "RightShoulder": (("right_collar", 1.0),),
    "RightArm": (("right_shoulder", 1.0),),
    "RightForeArm": (("right_elbow", 1.0),),
    "RightHand": (("right_wrist", 1.0),),
    "LeftUpLeg": (("left_hip", 1.0),),
    "LeftLeg": (("left_knee", 1.0),),
    "LeftFoot": (("left_ankle", 1.0),),
    "LeftToeBase": (("left_foot", 1.0),),
    "RightUpLeg": (("right_hip", 1.0),),
    "RightLeg": (("right_knee", 1.0),),
    "RightFoot": (("right_ankle", 1.0),),
    "RightToeBase": (("right_foot", 1.0),),
}

# legacy joint -> smpl joint whose global rotation it carries; None means
# identity local rotation (synthetic Spine1, leaf Hands).
_ROT_SRC: dict[str, str | None] = {
    "Hips": "pelvis",
    "Spine": "spine1",
    "Spine1": None,
    "Spine2": "spine2",
    "Spine3": "spine3",
    "Neck": "neck",
    "Head": "head",
    "LeftShoulder": "left_collar",
    "LeftArm": "left_shoulder",
    "LeftForeArm": "left_elbow",
    "LeftHand": "left_wrist",
    "RightShoulder": "right_collar",
    "RightArm": "right_shoulder",
    "RightForeArm": "right_elbow",
    "RightHand": "right_wrist",
    "LeftUpLeg": "left_hip",
    "LeftLeg": "left_knee",
    "LeftFoot": "left_ankle",
    "LeftToeBase": "left_foot",
    "RightUpLeg": "right_hip",
    "RightLeg": "right_knee",
    "RightFoot": "right_ankle",
    "RightToeBase": "right_foot",
}


def bvh_kind(path: str | Path) -> str:
    """Return 'legacy' or 'bvh24' from the hierarchy root name."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^\s*ROOT\s+(\S+)", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"BVH file has no ROOT joint: {path}")
    root = match.group(1)
    if root == "Hips":
        return "legacy"
    if root == "pelvis":
        return "bvh24"
    raise ValueError(f"Unrecognized BVH root joint {root!r} in {path}")


def remap_joints_smpl_to_legacy(
    joints: np.ndarray, names: tuple | list
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Positional SMPL-24 -> legacy-23 remap, frame-agnostic.

    ``joints`` is (T, >=24, 3) with per-joint names (SMPL snake_case) in any
    rigid frame (SMPL y-up or display z-up); the mapping is a linear
    combination of joint positions, so it commutes with any frame rotation.
    Returns (T, 23, 3) legacy joints, the legacy names and parents.
    """
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError(f"joint array must be (T,J,3), got {joints.shape}")
    idx = {str(n): i for i, n in enumerate(names)}
    missing = sorted({smpl for entries in _POS_MAP.values() for smpl, _ in entries} - set(idx))
    if missing:
        raise ValueError(f"missing SMPL joints: {missing}")
    out = np.zeros((joints.shape[0], N_LEGACY, 3), dtype=np.float64)
    for k, name in enumerate(LEGACY_JOINT_NAMES):
        for smpl_name, weight in _POS_MAP[name]:
            out[:, k] += weight * joints[:, idx[smpl_name]]
    return out, tuple(LEGACY_JOINT_NAMES), LEGACY_PARENTS.copy()


def remap_rotations_smpl_to_legacy(
    rotations: np.ndarray, names: tuple | list
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Global rotations (T,J,3,3) SMPL-24 -> legacy-23, frame-agnostic.

    Per :data:`_ROT_SRC`: every legacy joint carries its mapped SMPL joint's
    global rotation; synthetic Spine1 inherits its parent's (identity local).
    """
    rotations = np.asarray(rotations)
    if rotations.ndim != 4 or rotations.shape[2:] != (3, 3):
        raise ValueError(f"rotations must be (T,J,3,3), got {rotations.shape}")
    idx = {str(n): i for i, n in enumerate(names)}
    missing = sorted({src for src in _ROT_SRC.values() if src} - set(idx))
    if missing:
        raise ValueError(f"missing SMPL joints: {missing}")
    out = np.zeros((rotations.shape[0], N_LEGACY, 3, 3), dtype=np.float64)
    for k, name in enumerate(LEGACY_JOINT_NAMES):
        src = _ROT_SRC[name]
        if src is None:
            parent = LEGACY_PARENTS[k]
            out[:, k] = out[:, parent] if parent >= 0 else rotations[:, idx["pelvis"]]
        else:
            out[:, k] = rotations[:, idx[src]]
    return out, tuple(LEGACY_JOINT_NAMES)


def parse_legacy_channels(path: str | Path) -> tuple[np.ndarray, float, np.ndarray]:
    """Return ``(motion (T,72), frame_time, offsets_cm (23,3))`` in legacy layout.

    Legacy files parse verbatim; BVH24 files are converted (see module
    docstring).  Both kinds satisfy the consumers' contract: 72 channels,
    23 joints in :data:`LEGACY_JOINT_NAMES` order, y-up cm positions and
    offsets, YXZ degree eulers.
    """
    if bvh_kind(path) == "legacy":
        return _parse_legacy(Path(path))
    return _parse_bvh24(Path(path))


def _parse_legacy(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    """Verbatim legacy parser (mirrors the pre-BVH24 parse implementations)."""
    text = path.read_text(errors="replace").splitlines()
    offsets = []
    names = []
    channels: list[list[str]] = []
    joint_i = -1
    for line in text:
        stripped = line.strip()
        if stripped.startswith("ROOT ") or stripped.startswith("JOINT "):
            names.append(stripped.split()[1])
            joint_i += 1
        elif stripped.startswith("OFFSET") and joint_i >= 0 and len(offsets) == joint_i:
            offsets.append([float(v) for v in stripped.split()[1:4]])
        elif stripped.startswith("CHANNELS") and joint_i >= 0 and len(channels) == joint_i:
            parts = stripped.split()
            channels.append(parts[2 : 2 + int(parts[1])])
        if stripped.startswith("MOTION"):
            break
    # The downstream contract is YXZ degrees; other channel orders (e.g. the
    # model-exported gen.bvh files, which declare XYZ) must not be silently
    # misread here.
    if channels:
        if channels[0][3:] != ["Yrotation", "Xrotation", "Zrotation"] or any(
            ch != ["Yrotation", "Xrotation", "Zrotation"] for ch in channels[1:]
        ):
            raise ValueError(
                f"BVH channel order is not the legacy YXZ contract: {path} "
                f"(root {channels[0][3:]})"
            )
    motion_idx = next(i for i, line in enumerate(text) if line.startswith("MOTION"))
    n_frames = int(text[motion_idx + 1].split(":")[1])
    frame_time = float(text[motion_idx + 2].split(":")[1])
    motion = np.asarray(
        [list(map(float, line.split())) for line in text[motion_idx + 3 : motion_idx + 3 + n_frames]],
        dtype=np.float64,
    )
    if motion.shape != (n_frames, 72):
        raise ValueError("Unexpected BVH motion shape %s in %s" % (motion.shape, path))
    if tuple(names) != tuple(LEGACY_JOINT_NAMES):
        raise ValueError("Unexpected BVH joint names in %s: %s" % (path, names))
    if len(offsets) != N_LEGACY:
        raise ValueError("Unexpected BVH joint count %d in %s" % (len(offsets), path))
    return motion, frame_time, np.asarray(offsets, dtype=np.float64)


# SMPL-24 canonical joint order (npz ``joint_names``) and hierarchy.
SMPL_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
)
SMPL_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)


def _bvh24_structure(path: Path) -> tuple[list[str], np.ndarray, list[int], np.ndarray, float]:
    """Parse the BVH24 hierarchy and motion channels (Blender frame, DFS order)."""
    text = path.read_text(errors="replace").splitlines()
    names: list[str] = []
    offsets: list[np.ndarray] = []
    channels: list[list[str]] = []
    parents: list[int] = []
    stack: list[int] = []
    offset_filled: list[bool] = []
    pending: str | None = None
    for line in text:
        stripped = line.strip()
        if stripped.startswith("ROOT ") or stripped.startswith("JOINT "):
            pending = stripped.split()[1]
        elif stripped.startswith("End Site"):
            pending = "__end__"
        elif stripped == "{":
            if pending is None:
                raise ValueError(f"Malformed hierarchy in {path}: joint block without name")
            if pending == "__end__":
                stack.append(-1)  # marker; no joint arrays
            else:
                parents.append(stack[-1] if stack else -1)
                names.append(pending)
                offsets.append(np.zeros(3, dtype=np.float64))
                channels.append([])
                offset_filled.append(False)
                stack.append(len(names) - 1)
            pending = None
        elif stripped == "}":
            if stack:
                stack.pop()
        elif stripped.startswith("OFFSET") and stack and stack[-1] >= 0:
            # fill-once guard: the End Site OFFSET inside a joint block must
            # not overwrite the joint's own offset
            if not offset_filled[stack[-1]]:
                offsets[stack[-1]] = np.asarray(stripped.split()[1:4], dtype=np.float64)
                offset_filled[stack[-1]] = True
        elif stripped.startswith("CHANNELS") and stack and stack[-1] >= 0:
            parts = stripped.split()
            channels[stack[-1]] = parts[2 : 2 + int(parts[1])]
        if stripped.startswith("MOTION"):
            break
    if len(names) != 24:
        raise ValueError(f"Expected 24 joints in {path}, got {len(names)}")
    if channels[0] != ["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"]:
        raise ValueError(f"Unexpected root channels in {path}: {channels[0]}")
    if any(ch != ["Zrotation", "Xrotation", "Yrotation"] for ch in channels[1:]):
        raise ValueError(f"Unexpected joint channels in {path}")

    motion_idx = next(i for i, line in enumerate(text) if line.startswith("MOTION"))
    n_frames = int(text[motion_idx + 1].split(":")[1])
    frame_time = float(text[motion_idx + 2].split(":")[1])
    motion = np.asarray(
        [list(map(float, line.split())) for line in text[motion_idx + 3 : motion_idx + 3 + n_frames]],
        dtype=np.float64,
    )
    expected = 6 + 3 * (len(names) - 1)
    if motion.shape != (n_frames, expected):
        raise ValueError("Unexpected BVH motion shape %s in %s" % (motion.shape, path))
    offsets_b = np.stack(offsets)  # (24,3) metres, Blender frame
    return names, offsets_b, parents, motion, frame_time


def _bvh24_world(
    names: list[str], offsets_b: np.ndarray, parents: list[int], motion: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FK in the Blender frame, then basis change to SMPL y-up (DFS order)."""
    n_frames = motion.shape[0]
    R_local = np.zeros((n_frames, 24, 3, 3), dtype=np.float64)
    R_local[:, 0] = SciRotation.from_euler("ZXY", motion[:, 3:6], degrees=True).as_matrix()
    R_local[:, 1:] = (
        SciRotation.from_euler("ZXY", motion[:, 6:].reshape(-1, 3), degrees=True)
        .as_matrix()
        .reshape(n_frames, 23, 3, 3)
    )
    R_glob = np.zeros_like(R_local)
    pos_b = np.zeros((n_frames, 24, 3), dtype=np.float64)
    pos_b[:, 0] = motion[:, :3]
    rest_b = np.zeros((24, 3), dtype=np.float64)
    R_rest = np.broadcast_to(np.eye(3), (24, 3, 3)).copy()
    for j in range(24):
        p = parents[j]
        if p < 0:
            R_glob[:, j] = R_local[:, j]
        else:
            R_glob[:, j] = R_glob[:, p] @ R_local[:, j]
            pos_b[:, j] = pos_b[:, p] + np.einsum("nij,j->ni", R_glob[:, p], offsets_b[j])
            rest_b[j] = rest_b[p] + R_rest[p] @ offsets_b[j]
    pos_s = np.einsum("ij,nkj->nki", SMPL2BLENDER, pos_b)
    R_glob_s = np.einsum("ik,njkl,lm->njim", SMPL2BLENDER, R_glob, SMPL2BLENDER)
    rest_s = np.einsum("ij,kj->ki", SMPL2BLENDER, rest_b)
    return pos_s, R_glob_s, rest_s


def parse_bvh24_smpl(path: str | Path, query_t: np.ndarray | None = None) -> dict:
    """SMPL-24 native world data from a BVH24 file.

    Returns ``joints`` (T,24,3) world positions in the SMPL y-up frame
    (npz ``joint_names`` order), ``global_rotations`` (T,24,3,3), ``offsets``
    (24,3) rest-pose bone vectors (y-up metres), ``parents`` and
    ``frame_time``.  ``query_t`` resamples the motion onto the given times
    (positions linear, rotations SLERP; clipped to the source span).
    """
    names, offsets_b, parents, motion, frame_time = _bvh24_structure(Path(path))
    pos_s, R_glob_s, rest_s = _bvh24_world(names, offsets_b, parents, motion)
    order = [names.index(name) for name in SMPL_JOINT_NAMES]
    joints = pos_s[:, order]
    rotations = R_glob_s[:, order]
    rest = rest_s[order]
    offsets = np.zeros((24, 3), dtype=np.float64)
    for k in range(1, 24):
        offsets[k] = rest[k] - rest[SMPL_PARENTS[k]]
    if query_t is not None:
        query_t = np.asarray(query_t, dtype=np.float64)
        src_t = np.arange(motion.shape[0], dtype=np.float64) * frame_time
        clipped = np.clip(query_t, src_t[0], src_t[-1])
        joints = np.stack(
            [np.interp(clipped, src_t, joints[:, j, a]) for j in range(24) for a in range(3)],
            axis=1,
        ).reshape(len(clipped), 24, 3)
        if motion.shape[0] == 1:
            rotations = np.broadcast_to(rotations[0], (len(clipped), 24, 3, 3)).copy()
        else:
            out = np.empty((len(clipped), 24, 3, 3), dtype=np.float64)
            for j in range(24):
                slerp = Slerp(src_t, SciRotation.from_matrix(rotations[:, j]))
                out[:, j] = slerp(clipped).as_matrix()
            rotations = out
    return {
        "joints": joints,
        "global_rotations": rotations,
        "offsets": offsets,
        "parents": SMPL_PARENTS.copy(),
        "names": SMPL_JOINT_NAMES,
        "frame_time": frame_time,
    }


def _parse_bvh24(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    names, offsets_b, parents, motion, frame_time = _bvh24_structure(path)
    pos_s, R_glob_s, rest_s = _bvh24_world(names, offsets_b, parents, motion)
    n_frames = motion.shape[0]
    name_idx = {name: i for i, name in enumerate(names)}
    for name in _POS_MAP:
        for smpl_name, _ in _POS_MAP[name]:
            if smpl_name not in name_idx:
                raise ValueError(f"{path}: missing SMPL joint {smpl_name!r}")

    pos_legacy = np.zeros((n_frames, N_LEGACY, 3), dtype=np.float64)
    rest_legacy = np.zeros((N_LEGACY, 3), dtype=np.float64)
    R_glob_legacy = np.zeros((n_frames, N_LEGACY, 3, 3), dtype=np.float64)
    for k, name in enumerate(LEGACY_JOINT_NAMES):
        for smpl_name, weight in _POS_MAP[name]:
            src = name_idx[smpl_name]
            pos_legacy[:, k] += weight * pos_s[:, src]
            rest_legacy[k] += weight * rest_s[src]
        rot_src = _ROT_SRC[name]
        if rot_src is None:
            parent = LEGACY_PARENTS[k]
            R_glob_legacy[:, k] = (
                R_glob_legacy[:, parent] if parent >= 0 else R_glob_s[:, 0]
            )
        else:
            R_glob_legacy[:, k] = R_glob_s[:, name_idx[rot_src]]

    offsets_cm = np.zeros((N_LEGACY, 3), dtype=np.float64)
    for k in range(1, N_LEGACY):
        offsets_cm[k] = (rest_legacy[k] - rest_legacy[LEGACY_PARENTS[k]]) * 100.0

    root_euler = SciRotation.from_matrix(R_glob_legacy[:, 0]).as_euler("YXZ", degrees=True)
    local_eulers = np.zeros((n_frames, N_LEGACY - 1, 3), dtype=np.float64)
    for k in range(1, N_LEGACY):
        parent = LEGACY_PARENTS[k]
        local = np.einsum("nij,njk->nik", np.transpose(R_glob_legacy[:, parent], (0, 2, 1)), R_glob_legacy[:, k])
        local_eulers[:, k - 1] = SciRotation.from_matrix(local).as_euler("YXZ", degrees=True)

    motion_legacy = np.concatenate(
        [pos_legacy[:, 0] * 100.0, root_euler, local_eulers.reshape(n_frames, -1)],
        axis=1,
    )
    return motion_legacy, frame_time, offsets_cm
