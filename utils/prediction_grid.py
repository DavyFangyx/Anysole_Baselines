"""Place generated BVH on canonical frame ids, never on the raw mocap clock."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from Baselines.utils.motion_io import load_motion, load_rotations


def _semantic_joints(loaded: dict) -> tuple[np.ndarray, tuple[str, ...]]:
    """Drop BVH EndSite pseudo-joints; keep every semantic joint as-is."""
    names = tuple(str(n) for n in np.asarray(loaded["names"]).reshape(-1))
    keep = [i for i, n in enumerate(names) if "EndSite" not in n]
    if not keep:
        raise ValueError("BVH has no semantic joints: " + repr(names))
    return np.asarray(loaded["joints"])[:, keep], tuple(names[i] for i in keep)


def bvh_prediction_on_grid(path, row):
    path = Path(path)
    sidecar = path.with_suffix('.frames.npz')
    if not sidecar.is_file():
        raise ValueError('Generated BVH has no canonical frame map: ' + str(sidecar) +
                         '; rerun fixed inference. Raw mocap timestamps cannot index a local BVH.')
    points, names = _semantic_joints(load_motion(path))
    rotations = load_rotations(path)
    order = [rotations['names'].index(name) for name in names]
    with np.load(sidecar, allow_pickle=False) as z:
        ids = np.asarray(z['frame_indices'])
        keep = np.asarray(z['valid_mask'], dtype=bool)
        if str(z['session_id']) != str(row['session_id']):
            raise ValueError('Prediction frame map belongs to a different session')
    n = int(float(row['n_frames']))
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or len(ids) != len(points) or keep.shape != ids.shape
            or np.any(ids < 0) or np.any(ids >= n) or np.any(np.diff(ids) <= 0)):
        raise ValueError('Invalid/duplicated frame ids in generated BVH sidecar')
    full = np.zeros((n, len(names), 3), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    local_rot = np.broadcast_to(np.eye(3, dtype=np.float32), (n, len(names), 3, 3)).copy()
    full[ids] = points
    mask[ids] = keep
    local_rot[ids] = rotations['rotations'][:, order]
    return full, mask, names, local_rot
