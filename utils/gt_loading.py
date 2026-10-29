"""GT loading and prediction-archive reading on the canonical session grid.

Every loader samples GT at shared ``frames.npz:mocap_time_s``. Those clocks
already include accepted session corrections; manifest offsets can be older
and must not be applied a second time. Prediction archives use canonical
frame ids, while generated BVH has a separate local clock.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from pathlib import Path
import sys
from typing import Any

try:
    from AnysoleWorkspace.tool.workspace import resolve_uri as _resolve_uri
except ImportError:  # standalone use without the AnysoleWorkspace package on path
    _resolve_uri = None

_ROOT = Path(__file__).resolve().parents[2]  # utils/ -> Baselines/ -> repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from anysole.types import JOINT_NAMES
from Baselines.utils.motion_io import (
    LEGACY_BVH_NAMES,
    load_motion,
    load_rotations,
    load_surface,
)
from Baselines.utils.protocols import PROTOCOL_LABELS, normalize_protocol

REPO_ROOT = _ROOT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_list(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    try:
        out = json.loads(str(value))
        return [int(x) for x in out]
    except (ValueError, TypeError, json.JSONDecodeError):
        return []


def load_split_ids(split_csv: Path, split: str) -> list[str]:
    """Read the project-owned canonical split table in row order."""
    ids: list[str] = []
    with Path(split_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None or split not in rows.fieldnames:
            raise ValueError(f"{split_csv} missing split column {split!r}")
        for row in rows:
            value = (row.get(split) or "").strip()
            if value:
                ids.append(value)
    if not ids:
        raise ValueError(f"canonical split {split!r} is empty in {split_csv}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"canonical split {split!r} contains duplicate session ids")
    return ids


def read_manifest(path: Path, split: str, split_csv: Path | None = None,
                  evaluable_only: bool = True) -> list[dict[str, str]]:
    """Read manifest metadata using the canonical split table when supplied.

    The manifest is only session metadata. Train/val/test membership must be
    supplied by the canonical split CSV.
    """
    opener = path.open(encoding="utf-8-sig")
    if path.suffix.lower() in (".jsonl", ".json"):
        rows = [json.loads(line) for line in opener if line.strip()]
    else:
        rows = list(csv.DictReader(opener))
    by_id = {str(row["session_id"]): row for row in rows if row.get("session_id")}
    if split_csv is not None:
        ids = load_split_ids(split_csv, split)
        missing = [sid for sid in ids if sid not in by_id]
        if missing:
            raise ValueError(
                f"canonical split {split!r} contains sessions absent from {path}: {missing}"
            )
        selected = [by_id[sid] for sid in ids]
        if evaluable_only:
            selected = [
                row for row in selected
                if parse_list(row.get("valid_frame_indices"))
                or not ("valid_frame_indices" in row)
            ]
        return selected
    raise ValueError("split_csv is required; manifest has no split membership fields")


def resolve_repo_path(value: str, base: Path) -> Path:
    # Canonical URIs (raw://smpl/..., raw://bvh/..., work://, shared://, ...)
    # entered the manifest in 2026-10; delegate to the workspace resolver.
    if "://" in str(value) and _resolve_uri is not None:
        try:
            return _resolve_uri(value)
        except Exception:
            pass  # fall through to the legacy path handling below
    p = Path(value).expanduser()
    if p.is_absolute() and p.exists():
        return p
    marker = "/mocap_ori_bvh/"
    normalized = p.as_posix()
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        matches = sorted((REPO_ROOT / "AnysoleWorkspace/raw/bvh").glob("*/mocap_ori_bvh/" + suffix))
        if matches:
            return matches[0]
    for candidate in (base / p, REPO_ROOT / p):
        if candidate.exists():
            return candidate
    return p if p.is_absolute() else REPO_ROOT / p


def shared_session_frames(row: dict) -> dict:
    """Shared facts are the time authority; manifest offsets may predate corrections."""
    explicit = row.get('shared_frames_path')
    if explicit:
        path = Path(explicit)
    else:
        camera = str(row.get('camera') or 'cam3')
        if not camera.startswith('cam'):
            camera = 'cam' + camera
        candidates = sorted((REPO_ROOT / 'AnysoleWorkspace/shared/facts/sessions' / camera).glob(
            '*/*/' + str(row['session_id']) + '/frames.npz'))
        if len(candidates) != 1:
            raise ValueError('Need exactly one shared frames.npz for ' + str(row['session_id']))
        path = candidates[0]
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]).copy() for key in
                ('frame_id', 'mocap_time_s', 'visual_time_s', 'valid', 'fake')}
    n = len(data['frame_id'])
    if n <= 0 or not np.array_equal(data['frame_id'], np.arange(n)):
        raise ValueError(str(path) + ': frame ids must be 0..N-1')
    if row.get('n_frames') is not None and int(float(row['n_frames'])) != n:
        raise ValueError(str(path) + ': manifest/shared frame counts differ')
    for key in ('mocap_time_s', 'visual_time_s', 'valid', 'fake'):
        if data[key].shape != (n,):
            raise ValueError(str(path) + ': malformed ' + key)
    for key in ('mocap_time_s', 'visual_time_s'):
        if not np.isfinite(data[key]).all() or np.any(np.diff(data[key]) <= 0):
            raise ValueError(str(path) + ': nonfinite or nonmonotonic ' + key)
        if row.get('target_fps') and not np.allclose(np.diff(data[key]), 1.0/float(row['target_fps']), atol=1e-7, rtol=0):
            raise ValueError(str(path) + ': shared clock disagrees with FPS')
    data['source_path'] = str(path)
    return data


def session_time_grid(row: dict) -> np.ndarray:
    return shared_session_frames(row)['mocap_time_s']


def bvh_joints(path: Path, row: dict[str, str] | None = None) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Native BVH-23 world joints (+ names + parents) from a BVH file.

    BVH files carry the 23 Skeleton3 semantic joints plus EndSite markers
    (zero-rotation terminal nodes).  The native protocol is the 23 semantic
    joints: EndSite channels are dropped and parent indices remapped.
    Without ``row`` the raw BVH is returned as-is.  With ``row`` the motion is
    queried at shared ``frames.npz:mocap_time_s``.
    """
    query_t = session_time_grid(row) if row is not None else None
    loaded = load_motion(path, query_t=query_t)
    if loaded["format"] != "bvh":
        raise ValueError(f"expected legacy BVH, got {loaded['format']}: {path}")
    joints = np.asarray(loaded["joints"], dtype=np.float32)
    names = tuple(str(n) for n in np.asarray(loaded["names"]).reshape(-1))
    index = {name: i for i, name in enumerate(names)}
    missing = [name for name in LEGACY_BVH_NAMES if name not in index]
    if missing:
        raise ValueError(f"BVH missing Skeleton3 joints: {missing}: {path}")
    order = [index[name] for name in LEGACY_BVH_NAMES]
    joints = joints[:, order]
    old_to_new = {old: new for new, old in enumerate(order)}
    raw_parents = np.asarray(loaded["parents"], dtype=np.int64)
    parents = np.asarray(
        [
            old_to_new[int(raw_parents[old])] if int(raw_parents[old]) >= 0 else -1
            for old in order
        ],
        dtype=np.int64,
    )
    return joints, tuple(LEGACY_BVH_NAMES), parents


def _decode_names(values) -> tuple[str, ...]:
    return tuple(v.decode("utf-8") if isinstance(v, bytes) else str(v)
                 for v in np.asarray(values).reshape(-1).tolist())


def _infer_protocol(arr: np.ndarray, names: tuple[str, ...] | None) -> tuple[str, tuple[str, ...]]:
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"joint array must be (T,J,3), got {arr.shape}")
    if names is not None:
        if len(names) != arr.shape[1]:
            raise ValueError(f"joint_names has {len(names)} entries for J={arr.shape[1]}")
        if set(JOINT_NAMES).issubset(names):
            return "smpl24", names
        if set(LEGACY_BVH_NAMES).issubset(names):
            return "bvh23", names
        raise ValueError(f"unrecognized joint_names protocol: {names}")
    if arr.shape[1] == len(JOINT_NAMES):
        return "smpl24", tuple(JOINT_NAMES)
    if arr.shape[1] == len(LEGACY_BVH_NAMES):
        return "bvh23", tuple(LEGACY_BVH_NAMES)
    raise ValueError(
        f"cannot infer motion protocol from J={arr.shape[1]}; include joint_names"
    )


def array_from_file(path: Path) -> tuple[np.ndarray, np.ndarray | None, tuple[str, ...], str]:
    """Load (T,J,3) world joints + optional valid mask + joint names + protocol.

    NPZ files follow the unified ``eval_motion`` contract
    (``joint_xyz_world`` / ``joint_names`` / ``valid_mask``) or the SMPL NPZ
    motion contract (``poses``/``trans``).  ``*.bvh`` files are parsed as
    native BVH-23 (Skeleton3, EndSite channels dropped).  Returned joints are
    in the z-up display frame; the protocol identifier is the recorded
    ``smpl24-native`` / ``bvh23-native`` label.
    """
    if path.suffix.lower() == ".bvh":
        joints, names, _parents = bvh_joints(path)
        return joints, None, names, "bvh23-native"
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            # The explicit unified contract wins over native SMPL parameter
            # fields that may be carried in the same archive for trajectory
            # and mesh readers.  AnySole archives contain both forms.
            if "joint_xyz_world" in data:
                arr = np.asarray(data["joint_xyz_world"])
                mask = np.asarray(data["valid_mask"]).reshape(-1) if "valid_mask" in data else None
                names = _decode_names(data["joint_names"]) if "joint_names" in data else None
                protocol, names = _infer_protocol(arr, names)
                return arr, mask, names, PROTOCOL_LABELS[protocol]
            if ({"poses", "trans"}.issubset(data.files)
                    or {"root_orient", "pose_body", "trans"}.issubset(data.files)):
                motion = load_motion(path)
                return motion["joints"], None, tuple(motion["names"]), "smpl24-native"
            keys = list(data.keys())
            candidates = ("joints_world", "pred_joints", "joints", "xyz", "pose")
            key = next((k for k in candidates if k in data), None)
            if key is None:
                raise ValueError(f"no joint array in {path}; keys={keys}")
            arr = np.asarray(data[key])
            mask = np.asarray(data["valid_mask"]).reshape(-1) if "valid_mask" in data else None
            names = _decode_names(data["joint_names"]) if "joint_names" in data else None
            protocol, names = _infer_protocol(arr, names)
            return arr, mask, names, PROTOCOL_LABELS[protocol]
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray):
        protocol, names = _infer_protocol(obj, None)
        return obj, None, names, PROTOCOL_LABELS[protocol]
    if isinstance(obj, dict):
        for key in ("joint_xyz_world", "joints_world", "pred_joints", "joints", "xyz"):
            if key in obj:
                arr = np.asarray(obj[key])
                names_in = _decode_names(obj["joint_names"]) if "joint_names" in obj else None
                protocol, names = _infer_protocol(arr, names_in)
                mask = np.asarray(obj.get("valid_mask")).reshape(-1) if "valid_mask" in obj else None
                return arr, mask, names, PROTOCOL_LABELS[protocol]
    raise ValueError(f"unsupported prediction object: {path}")


def protocol_gt(row: dict[str, str], protocol: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """GT joints in the prediction's native protocol, in native joint order."""
    protocol = normalize_protocol(protocol)
    if protocol == "bvh23":
        path = resolve_repo_path(row["bvh_path"], REPO_ROOT)
        joints, names, _parents = bvh_joints(path, row)
        if "pelvis" in names:
            # Farm BVH content is now the BVH24 export; remap onto the native
            # BVH-23 layout only here (GT for the legacy protocol).
            from Baselines.utils.bvh24_to_legacy import remap_joints_smpl_to_legacy

            joints, names, _parents = remap_joints_smpl_to_legacy(joints, names)
        return joints, names
    if protocol == "smpl24":
        path = resolve_repo_path(row["smpl_path"], REPO_ROOT)
        n = int(float(row["n_frames"]))
        fps = float(row.get("target_fps") or 40.0)
        query_t = session_time_grid(row)
        motion = load_motion(path, query_t=query_t)
        return motion["joints"], tuple(motion["names"])
    raise ValueError(f"unsupported protocol {protocol!r}")


def protocol_gt_surface(row: dict[str, str], protocol: str) -> dict | None:
    if normalize_protocol(protocol) != "smpl24":
        return None
    path = resolve_repo_path(row["smpl_path"], REPO_ROOT)
    n = int(float(row["n_frames"]))
    fps = float(row.get("target_fps") or 40.0)
    query_t = session_time_grid(row)
    return load_surface(path, query_t=query_t)


def bvh23_rotation_view(data: dict) -> dict:
    """Drop BVH EndSite nodes and reorder rotations to the 23-joint BVH-23 order."""
    names = tuple(str(n) for n in np.asarray(data["names"]).reshape(-1))
    if "pelvis" in names:
        # Farm BVH content is now the 2026-10-24 BVH24 export; remap its
        # SMPL-24 global rotations onto the native BVH-23 layout (same frame).
        from Baselines.utils.bvh24_to_legacy import remap_rotations_smpl_to_legacy

        rotations, names = remap_rotations_smpl_to_legacy(np.asarray(data["rotations"]), names)
        return {**data, "rotations": rotations, "names": names}
    order = [data['names'].index(name) for name in LEGACY_BVH_NAMES]
    return {**data, 'rotations': np.asarray(data['rotations'])[:, order],
            'names': tuple(LEGACY_BVH_NAMES)}


def protocol_gt_rotations(row: dict[str, str], protocol: str) -> dict:
    protocol = normalize_protocol(protocol)
    if protocol == "bvh23":
        path = resolve_repo_path(row["bvh_path"], REPO_ROOT)
        n = int(float(row.get("n_frames") or 0))
        fps = float(row.get("target_fps") or 40.0)
        query_t = session_time_grid(row)
        data = load_rotations(path, query_t=query_t)
        # BVH EndSite nodes are not semantic joints in the public BVH-23 protocol.
        return bvh23_rotation_view(data)
    path = resolve_repo_path(row["smpl_path"], REPO_ROOT)
    n = int(float(row["n_frames"]))
    fps = float(row.get("target_fps") or 40.0)
    query_t = session_time_grid(row)
    return load_rotations(path, query_t=query_t)


def find_prediction(root: Path, session_id: str, pattern: str | None = None, config_id: str | None = None) -> Path | None:
    if pattern:
        try:
            candidate = root / pattern.format(session_id=session_id, config_id=config_id)
        except (KeyError, IndexError):
            candidate = None
        if candidate is not None and candidate.is_file():
            return candidate
    for ext in (".npz", ".npy", ".bvh"):
        exact = root / f"{session_id}{ext}"
        if exact.is_file():
            return exact
    matches = sorted(p for p in root.rglob(f"*{session_id}*") if p.suffix.lower() in (".npz", ".npy", ".bvh"))
    if not matches:
        return None
    # E2-4 variant rule (2026-10-06): never pick the first rglob hit silently.
    # Prefer the plain per-session file, then clip variants in _cN order, and
    # log which file was chosen so a variant pick is visible in evaluation.log.
    plain = [p for p in matches if not re.search(r"_c\d+", p.stem)]
    chosen = (plain or matches)[0]
    if chosen != matches[0]:
        logging.warning("find_prediction %s: %d candidates; variant pick: %s (plain first, then _cN order)",
                        session_id, len(matches), chosen)
    return chosen


def valid_mask(row: dict[str, str], n: int) -> np.ndarray:
    if row.get("valid_frame_indices"):
        keep = np.zeros(n, dtype=bool)
        keep[np.asarray([i for i in parse_list(row["valid_frame_indices"]) if 0 <= i < n], dtype=int)] = True
        return keep
    mask = np.ones(n, dtype=bool)
    for i in parse_list(row.get("fake_frame_indices")):
        if 0 <= i < n:
            mask[i] = False
    return mask


def archive_meta(path: Path) -> dict[str, str]:
    """Read scalar provenance/string fields of one NPZ without loading arrays."""
    if path.suffix.lower() != ".npz":
        return {}
    with np.load(path, allow_pickle=False) as data:
        out = {}
        for key in data.files:
            arr = data[key]
            if arr.ndim == 0 or (arr.ndim == 1 and arr.shape[0] == 1 and arr.dtype.kind in "UOS"):
                value = arr.reshape(-1)[0]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if isinstance(value, str) and not value.startswith("{"):
                    out[key] = str(value)
        return out
