"""Table 3/4 metrics for MotionPRO test evaluation."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anysole.utils.metrics import (  # noqa: E402
    foot_sliding_joints,
    mean_point_error,
    pa_mpjpe,
    pelvis_align as canonical_pelvis_align,
    root_trajectory_metrics,
    rotation_error_degrees,
    shape_vertex_std,
    temporal_metrics,
    windowed_world_mpjpe,
)

PELVIS_JOINT = 0
HEIGHT_AXIS = 1
CONTACT_HEIGHT_THRESH = 0.05
# Canonical joint-based foot_sliding_mm uses the protocol's four foot joints:
# SMPL left/right ankle and left/right foot.
FOOT_JOINT_IDS_SMPL = (7, 8, 10, 11)
MM_SCALE = 1000.0
JITTER_SCALE = 1000.0
DEFAULT_FPS = 40.0
NAN = float('nan')

# MotionPRO's private table 3/4 columns.  Frozen set; every value comes from
# the canonical anysole/utils/metrics.py formulas (no local redefinition).
# The canonical module adds root_orientation_* and dropped shape_vertex_std_mm
# from the public set — this table keeps its historical columns; the two new
# canonical metrics are covered by the public evaluator instead.
METRIC_NAMES = (
    "mpjpe_mm",
    "pa_mpjpe_mm",
    "mpjae_deg",
    "pve_mm",
    "shape_vertex_std_mm",
    "root_ate_mm",
    "root_rte_percent",
    "w_mpjpe100_mm",
    "wa_mpjpe100_mm",
    "accel_error_m_s2",
    "jitter_pred_m_s3",
    "jitter_gt_m_s3",
    "foot_sliding_mm",
    "WBCE",
)
CSV_FIELDS = ('session_id', 'n_valid_frames') + METRIC_NAMES


def _as_bool(mask, n: int) -> np.ndarray:
    if mask is None:
        return np.ones((n,), dtype=bool)
    out = np.asarray(mask).reshape(-1).astype(bool)
    if out.shape[0] != n:
        raise ValueError(f'valid mask length {out.shape[0]} != {n}')
    return out


def _nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return NAN
    return float(values.mean())


def pelvis_align(points: np.ndarray) -> np.ndarray:
    return points - points[:, PELVIS_JOINT:PELVIS_JOINT + 1]


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Return R, t, scale mapping src -> dst."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f'umeyama expects (N,3) pairs, got {src.shape} and {dst.shape}')
    n = src.shape[0]
    if n == 0:
        return np.eye(3), np.zeros(3), 1.0
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
        S = S.copy()
        S[-1] *= -1
    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    scale = 1.0
    if with_scale and var_src > 0:
        scale = float(np.sum(S) / var_src)
    t = mu_dst - scale * (R @ mu_src)
    return R, t, scale


def apply_similarity(points: np.ndarray, R, t, scale: float = 1.0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return scale * points @ R.T + t


def first_valid_indices(valid: np.ndarray, n: int = 2) -> np.ndarray:
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return idx
    return idx[:n]


def last_valid_index(valid: np.ndarray) -> Optional[int]:
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return None
    return int(idx[-1])


def _accel(points: np.ndarray, fps: float) -> np.ndarray:
    acc = np.full_like(points, np.nan, dtype=np.float64)
    if points.shape[0] < 3:
        return acc
    acc[1:-1] = (points[:-2] - 2.0 * points[1:-1] + points[2:]) * (fps ** 2)
    return acc


def _triple_valid(valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(valid, dtype=bool)
    if valid.shape[0] < 3:
        return out
    out[1:-1] = valid[:-2] & valid[1:-1] & valid[2:]
    return out


def _mean_l2(diff: np.ndarray, mask: np.ndarray) -> float:
    if mask is None:
        kept = diff.reshape(-1, diff.shape[-1])
    else:
        kept = diff[mask]
    if kept.size == 0:
        return NAN
    return float(np.mean(np.linalg.norm(kept.reshape(-1, kept.shape[-1]), axis=-1)))


def compute_session_metrics(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    pred_verts: Optional[np.ndarray] = None,
    gt_verts: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
    fps: float = DEFAULT_FPS,
    session_id: str = '',
    pred_rotations: Optional[np.ndarray] = None,
    gt_rotations: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    pred_j = np.asarray(pred_joints, dtype=np.float64)
    gt_j = np.asarray(gt_joints, dtype=np.float64)
    if pred_j.shape != gt_j.shape or pred_j.ndim != 3 or pred_j.shape[-1] != 3:
        raise ValueError(f'joints must be (T,J,3), got {pred_j.shape} and {gt_j.shape}')
    n_frames = pred_j.shape[0]
    keep = _as_bool(valid, n_frames)
    n_valid = int(keep.sum())
    out = {
        'session_id': session_id,
        'n_valid_frames': float(n_valid),
    }
    for name in METRIC_NAMES:
        out[name] = NAN
    if n_valid == 0:
        return out

    times = np.arange(n_frames, dtype=np.float64) / float(fps)
    pred_valid = pred_j[keep]
    gt_valid = gt_j[keep]
    times_valid = times[keep]
    pred_aligned, _ = canonical_pelvis_align(pred_valid)
    gt_aligned, _ = canonical_pelvis_align(gt_valid)
    out['mpjpe_mm'] = float(mean_point_error(pred_aligned, gt_aligned, scale=1000.0).mean())
    try:
        out['pa_mpjpe_mm'] = float(pa_mpjpe(pred_valid, gt_valid).mean())
    except ValueError:
        pass
    root = root_trajectory_metrics(pred_valid[:, PELVIS_JOINT], gt_valid[:, PELVIS_JOINT])
    out['root_ate_mm'] = root['root_ate_mm']
    out['root_rte_percent'] = root['root_rte_percent']
    try:
        out['w_mpjpe100_mm'], out['wa_mpjpe100_mm'] = windowed_world_mpjpe(
            pred_valid, gt_valid, window=100
        )
    except ValueError:
        pass
    temporal = temporal_metrics(pred_valid, gt_valid, times_valid)
    for key in ('accel_error_m_s2', 'jitter_pred_m_s3', 'jitter_gt_m_s3'):
        out[key] = float(temporal[key])
    if pred_rotations is not None and gt_rotations is not None:
        out['mpjae_deg'] = float(rotation_error_degrees(
            np.asarray(pred_rotations)[keep], np.asarray(gt_rotations)[keep]
        ).mean())
    if pred_j.shape[1] >= 12:
        # Canonical public foot_sliding_mm: joint-based over the four SMPL
        # foot joints.  (The vertex variant is a SMPL-only diagnostic kept
        # under its own name in the canonical module.)
        out['foot_sliding_mm'], _ = foot_sliding_joints(
            pred_j[keep][:, FOOT_JOINT_IDS_SMPL],
            gt_j[keep][:, FOOT_JOINT_IDS_SMPL],
            times_valid,
        )

    if pred_verts is not None and gt_verts is not None:
        pred_v = np.asarray(pred_verts, dtype=np.float64)
        gt_v = np.asarray(gt_verts, dtype=np.float64)
        if pred_v.shape != gt_v.shape or pred_v.ndim != 3 or pred_v.shape[-1] != 3:
            raise ValueError(f'verts must be (T,V,3), got {pred_v.shape} and {gt_v.shape}')
        _, pred_v_aligned = canonical_pelvis_align(pred_valid, pred_v[keep])
        _, gt_v_aligned = canonical_pelvis_align(gt_valid, gt_v[keep])
        out['pve_mm'] = float(mean_point_error(pred_v_aligned, gt_v_aligned, scale=1000.0).mean())
        out['shape_vertex_std_mm'] = shape_vertex_std(pred_v[keep])

        ground = float(gt_v[keep, :, HEIGHT_AXIS].min())
        contact = (gt_v[:, :, HEIGHT_AXIS] - ground) < CONTACT_HEIGHT_THRESH
        contact[~keep] = False
        if np.any(contact):
            out['WBCE'] = float(np.mean(np.abs(pred_v[contact, HEIGHT_AXIS] - ground))) * MM_SCALE
    else:
        pred_v = None
        gt_v = None

    return out


def summarize_metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    valid_rows = [row for row in rows if int(row.get('n_valid_frames', 0)) > 0]
    summary = {
        'session_id': 'OVERALL',
        'n_valid_frames': float(sum(int(row['n_valid_frames']) for row in valid_rows)),
    }
    for name in METRIC_NAMES:
        summary[name] = NAN
    if not valid_rows:
        return summary

    weights = np.array([row['n_valid_frames'] for row in valid_rows], dtype=np.float64)
    for name in METRIC_NAMES:
        values = np.array([row[name] for row in valid_rows], dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        if name == 'root_rte_percent':
            summary[name] = float(values[finite].mean())
        else:
            w = weights[finite]
            if w.sum() <= 0:
                continue
            summary[name] = float(np.average(values[finite], weights=w))
    return summary


def _fmt(value: float) -> str:
    if value is None or not math.isfinite(value):
        return 'nan'
    return f'{value:.6f}'


def write_metrics_files(rows: Sequence[Mapping[str, float]], result_dir: Path, fps: float = DEFAULT_FPS):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    all_rows = list(rows) + [summarize_metrics(rows)]
    csv_path = result_dir / 'test_metrics.csv'
    log_path = result_dir / 'test_metrics.log'
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in all_rows:
            writer.writerow({
                'session_id': row['session_id'],
                'n_valid_frames': int(row['n_valid_frames']),
                **{name: _fmt(row[name]) for name in METRIC_NAMES},
            })
    lines = [
        'MotionPRO test metrics',
        f'eval_fps: {fps}',
        'units: *_mm/WBCE in mm; mpjae_deg in degrees; root_rte_percent in %; '
        'accel_error_m_s2 in m/s^2; jitter_*_m_s3 in m/s^3',
        '',
    ]
    header = f"{'session_id':<16}{'n_valid':>8}" + ''.join(f'{name:>12}' for name in METRIC_NAMES)
    lines.append(header)
    for row in all_rows:
        line = f"{str(row['session_id']):<16}{int(row['n_valid_frames']):>8}"
        line += ''.join(f'{_fmt(row[name]):>12}' for name in METRIC_NAMES)
        lines.append(line)
    log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return csv_path, log_path
