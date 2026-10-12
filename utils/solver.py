"""Protocol-aware metric assembly for the baseline evaluation side.

This module contains zero metric formulas: every formula comes from the single
implementation in ``anysole/utils/metrics.py`` (评估整改任务 01 §4).  What
lives here is the assembly layer — protocol dispatch, necessary-input handling
and the V2T nine-metric solver — shared by the baseline evaluation entry
(``Baselines/utils/evaluate.py``) and, through the re-exports in
``results_display/script/utils/compare_core.py``, by the comparison layer.

Contact metrics are deliberately absent (评估整改任务 01 §4.4): no motion
model trains and exports an explicit contact prediction, so formal
``contact_f1`` is missing (—) for all of them.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]  # utils/ -> Baselines/ -> repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from scipy.ndimage import zoom

from anysole.types import JOINT_NAMES
from anysole.utils.metrics import (
    MOTION_METRIC_NAMES,
    foot_sliding_joints,
    foot_sliding_vertices,
    matrix_rotation_error_degrees,
    mean_point_error,
    pa_mpjpe,
    pelvis_align,
    root_orientation_drift_deg,
    root_orientation_error_deg,
    root_trajectory_metrics,
    rotation_error_degrees,
    shape_vertex_std,
    temporal_metrics,
    windowed_world_mpjpe,
)
from Baselines.utils.motion_io import LEGACY_BVH_NAMES
from Baselines.utils.protocols import (
    PROTOCOL_FOOT_NAMES,
    PROTOCOL_PELVIS,
    normalize_protocol,
)

# V2T metric hierarchy (2026-09-27): three pressure levels (grid / force /
# CoP) plus the contact level.  Each level keeps one L2 representative plus
# one shape/global statistic in the brief; L1 errors and derived means are
# leaf keys — computed and stored in detail files, excluded from the brief.
# The grid level additionally keeps ``T_mse`` (= T_rmse^2) as a leaf so the
# name matches FPP-Net's native ``pressure MSE``.  The contact level
# (phase 2) carries the reconstruction quality (``contact_smpl_mse``
# [brief] / ``contact_smpl_bce`` [leaf]) on the continuous per-vertex SMPL
# contact maps: pred from the FPP-Net contact head, gt from the press2Cont
# th=0.5 binary vertex contact (E3-A ruling; provenance fields
# ``contact_gt_source`` / ``contact_gt_threshold`` / ``pixel_weight_revision``
# in the archive).  ``contact_f1`` is deliberately NOT part of the formal set:
# its binary GT now exists, but AnySole has no contact head, so the key would
# be permanently single-row — stays diagnostic (E3-Q5).
V2T_LEVELS = {
    "grid": ("T_corr", "T_rmse", "T_mse", "T_mae"),
    "force": ("pressure_force_rmse", "pressure_force_r2", "pressure_force_mae"),
    "cop": ("pressure_cop_error_left", "pressure_cop_error_right",
            "pressure_cop_error_mean"),
    "contact": ("contact_smpl_mse", "contact_smpl_bce"),
}
V2T_BRIEF_KEYS = (
    "T_corr", "T_rmse",
    "pressure_force_rmse", "pressure_force_r2",
    "pressure_cop_error_left", "pressure_cop_error_right",
    "contact_smpl_mse",
)
V2T_LEAF_KEYS = ("T_mse", "T_mae", "pressure_force_mae", "pressure_cop_error_mean",
                 "contact_smpl_bce")
V2T_METRICS = V2T_BRIEF_KEYS + V2T_LEAF_KEYS
PRESSURE_GRID_SHAPE = (31, 11)
# Public comparison metrics per family.  ``shape_vertex_std_mm`` /
# ``foot_sliding_vertex_mm`` are SMPL-only diagnostics computed when surface
# data exists but are not public comparison metrics.
METRICS = MOTION_METRIC_NAMES + V2T_METRICS

# Cross-model comparison file schemas.  v1 (common19) is superseded and must
# never be read back as a current result.
COMPARISON_SCHEMA_V2 = "mmvp_native_metrics_v2"
LEGACY_SCHEMA_V1 = "mmvp_common_metrics_v1"


def canonical_pressure_feet(values: np.ndarray) -> np.ndarray:
    """Return normalized pressure as ``(T,2,31,11)`` for V2T metrics."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 96:
        grids = values.reshape(-1, 2, 4, 12)
    elif values.ndim == 3 and values.shape[1:] == (31, 22):
        grids = np.stack([values[:, :, :11], values[:, :, 11:]], axis=1)
    elif values.ndim == 4 and values.shape[1] == 2:
        grids = values
    else:
        raise ValueError(f"unsupported V2T pressure array shape {values.shape}")
    if tuple(grids.shape[-2:]) == (4, 12):
        # AnySole 4×12：4=内外侧短轴、12=脚跟到脚尖长轴（PCA 主轴实证：
        # 长轴沿 12 轴 463/531 帧；FPP 31×11 长轴沿 31 轴 670/670 帧）。
        # 先转置再 zoom（12→31 长轴、4→11 短轴），与 FPP 轴约定对齐；
        # 不转置直接 zoom 会把内外侧轴放到 31 长轴上（90° 错位）。
        # pred/gt 同步变换，AnySole 自身指标数值不变（方向无关）。
        grids = zoom(
            grids.transpose(0, 1, 3, 2),
            (1, 1, PRESSURE_GRID_SHAPE[0] / 12, PRESSURE_GRID_SHAPE[1] / 4),
            order=1,
            mode="nearest",
            prefilter=False,
        )
    if tuple(grids.shape[-2:]) != PRESSURE_GRID_SHAPE:
        raise ValueError(
            f"pressure grids must resample to {PRESSURE_GRID_SHAPE}, got {grids.shape[-2:]}"
        )
    return grids


def pressure_metrics(pred: np.ndarray, target: np.ndarray,
                     valid: np.ndarray | None = None) -> dict[str, float]:
    """Pressure reconstruction measures on one common 31x11 foot grid.

    AnySole stores 4x12 cells per foot while FPP-Net stores 31x11 cells per
    foot.  Comparing the flattened native arrays would be dimensionally
    invalid.  The former is bilinearly resampled to 31x11; force and CoP are
    then computed from the same normalized grid for both methods.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    p = canonical_pressure_feet(pred)
    g = canonical_pressure_feet(target)
    n = min(len(p), len(g))
    p, g = p[:n], g[:n]
    keep = np.ones(n, dtype=bool) if valid is None else np.asarray(valid[:n], dtype=bool)
    p, g = p[keep], g[keep]
    if not len(p):
        return {key: float("nan") for key in V2T_METRICS}
    diff = p - g
    frame_corr = []
    for p_frame, g_frame in zip(p, g):
        pv, gv = p_frame.reshape(-1), g_frame.reshape(-1)
        if np.std(pv) > 1e-8 and np.std(gv) > 1e-8:
            frame_corr.append(float(np.corrcoef(pv, gv)[0, 1]))
    corr = float(np.mean(frame_corr)) if frame_corr else float("nan")
    force_p, force_g = p.sum(axis=(-1, -2)), g.sum(axis=(-1, -2))
    force_diff = force_p - force_g
    total_p, total_g = force_p.sum(axis=-1), force_g.sum(axis=-1)
    ss_res = float(np.square(total_p - total_g).sum())
    ss_tot = float(np.square(total_g - total_g.mean()).sum())

    def cop(grid):
        # Both supported layouts are canonicalized to [T,foot,width,length].
        width, length = grid.shape[-2:]
        mass = np.clip(grid, 0.0, None)
        force = mass.sum(axis=(-1, -2))
        x = np.arange(width, dtype=np.float64).reshape(1, 1, width, 1)
        y = np.arange(length, dtype=np.float64).reshape(1, 1, 1, length)
        cx = (mass * x).sum(axis=(-1, -2)) / np.maximum(force, 1e-8) / max(width - 1, 1)
        cy = 1.0 - (mass * y).sum(axis=(-1, -2)) / np.maximum(force, 1e-8) / max(length - 1, 1)
        return np.stack([cx, cy], axis=-1)

    cop_diff = np.linalg.norm(cop(p) - cop(g), axis=-1)
    sq = np.square(diff)
    mse = float(sq.mean())
    return {
        "T_mae": float(np.abs(diff).mean()),
        "T_rmse": float(np.sqrt(mse)),
        "T_mse": mse,
        "T_corr": corr,
        "pressure_force_mae": float(np.abs(force_diff).mean()),
        "pressure_force_rmse": float(np.sqrt(np.square(force_diff).mean())),
        "pressure_force_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "pressure_cop_error_left": float(cop_diff[:, 0].mean()),
        "pressure_cop_error_right": float(cop_diff[:, 1].mean()),
        "pressure_cop_error_mean": float(cop_diff.mean()),
    }


def contact_smpl_metrics(pred: np.ndarray, target: np.ndarray,
                         valid: np.ndarray | None = None) -> dict[str, float]:
    """Per-vertex SMPL contact reconstruction measures (contact level).

    ``pred`` / ``target`` are (T, 2, N_vert) continuous contact maps: pred
    from the FPP-Net contact head, target from the press2Cont th=0.5 binary
    vertex contact (E3-A; provenance in the archive).  MSE is the plain L2
    over all vertices;
    BCE matches FPP-Net's native ``binary_cross_entropy`` training loss
    (clipped logits).  contact_f1 is deliberately absent: it requires a
    binary contact GT that does not exist yet (待处理).
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = min(len(pred), len(target))
    pred, target = pred[:n], target[:n]
    keep = np.ones(n, dtype=bool) if valid is None else np.asarray(valid[:n], dtype=bool)
    pred, target = pred[keep], target[keep]
    if not len(pred):
        return {"contact_smpl_mse": float("nan"), "contact_smpl_bce": float("nan")}
    eps = 1e-7
    clipped = np.clip(pred, eps, 1.0 - eps)
    return {
        "contact_smpl_mse": float(np.square(pred - target).mean()),
        "contact_smpl_bce": float(
            -(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)).mean()
        ),
    }


def v2t_metrics(pred_pressure: np.ndarray, gt_pressure: np.ndarray,
                valid: np.ndarray | None = None,
                contact_smpl_pred: np.ndarray | None = None,
                contact_smpl_gt: np.ndarray | None = None) -> dict[str, float]:
    """Canonical V2T metrics — brief (7) + leaf (5) keys, four levels.

    The brief carries one representative per level: grid L2 (T_rmse) and
    shape correlation (T_corr), force L2 (pressure_force_rmse) and the
    global statistic (pressure_force_r2), per-foot CoP errors, and the
    contact-level L2 (contact_smpl_mse).  Leaf keys (T_mse = T_rmse^2,
    T_mae, pressure_force_mae, pressure_cop_error_mean, contact_smpl_bce)
    are computed and stored in detail files but excluded from the brief.

    Contact keys are computed only when continuous contact maps are given
    (FPP-Net archives); otherwise they are NaN — the capability gate
    (``contact_prediction``) decides whether that is ``not_applicable``.
    Legacy binarized ``contact_pred``/``contact_gt`` archive fields are
    never consumed here.
    """
    out = pressure_metrics(pred_pressure, gt_pressure, valid)
    if contact_smpl_pred is not None and contact_smpl_gt is not None:
        out.update(contact_smpl_metrics(contact_smpl_pred, contact_smpl_gt, valid))
    else:
        out["contact_smpl_mse"] = float("nan")
        out["contact_smpl_bce"] = float("nan")
    pred = np.asarray(pred_pressure)
    gt = np.asarray(gt_pressure)
    n = min(len(pred), len(gt))
    keep = np.ones(n, dtype=bool) if valid is None else np.asarray(valid[:n], dtype=bool)
    out["n_valid_frames"] = int(keep.sum())
    return out


def load_v2t_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"pressure_pred", "pressure_gt", "valid_mask"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"V2T archive missing keys {sorted(missing)}: {path}")
        # contact_pred/contact_gt are legacy binarized diagnostics: their
        # presence is tolerated but they are never returned as formal
        # inputs.  contact_smpl_pred/contact_smpl_gt (continuous per-vertex
        # maps, contact level) are returned when present.
        out = {key: np.asarray(data[key]) for key in required}
        for key in ("contact_smpl_pred", "contact_smpl_gt"):
            if key in data.files:
                out[key] = np.asarray(data[key])
        return out


def _foot_joint_indices(names: tuple[str, ...] | None, protocol: str) -> np.ndarray:
    wanted = PROTOCOL_FOOT_NAMES[protocol]
    if names is not None:
        index = {name: i for i, name in enumerate(names)}
        missing = [name for name in wanted if name not in index]
        if missing:
            raise ValueError(f"{protocol} missing foot joints: {missing}")
        ids = [index[name] for name in wanted]
    else:
        default_names = JOINT_NAMES if protocol == "smpl24" else LEGACY_BVH_NAMES
        index = {name: i for i, name in enumerate(default_names)}
        ids = [index[name] for name in wanted]
    return np.asarray(ids, dtype=np.int64)


def metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    keep: np.ndarray,
    fps: float,
    *,
    protocol: str,
    names: tuple[str, ...] | None = None,
    pred_rotations: np.ndarray | None = None,
    gt_rotations: np.ndarray | None = None,
    pred_vertices: np.ndarray | None = None,
    gt_vertices: np.ndarray | None = None,
    times: np.ndarray | None = None,
    forward_axis: int | None = None,
) -> dict[str, float]:
    """Canonical motion metrics in the prediction's native protocol.

    ``protocol`` selects the native joint convention ("smpl24" / "bvh23";
    the "-native" labels are accepted): pelvis alignment, root trajectory and
    the joint-based foot sliding are protocol-aware.  No common19 cropping
    exists anymore — ``pred``/``gt`` must already be the same native joint
    order, and the metric formulas come from ``anysole.utils.metrics`` only.

    ``root_ate_mm`` is absolute for smpl24 and translation-aligned for bvh23
    (bvh23 has no world anchor by upstream design; per-sequence translation
    alignment, drift point o, 2026-10-08).

    Root orientation needs ``pred_rotations``/``gt_rotations`` whose channel 0
    is the root rotation; ``forward_axis`` is the fixed body-forward axis of
    that rotation frame (see ``protocols.PROTOCOL_FORWARD_AXIS``).  MPJAE
    additionally needs full joint rotations — callers gate it by the
    ``direct_joint_rotations`` capability.  ``pve_mm`` and the surface
    diagnostics appear only when vertices are passed (callers gate by
    ``predicted_surface`` and the archive public-surface contract; GT beta
    never activates this path).  Contact is
    deliberately absent: formal contact_f1 is missing (—) for every current
    model.
    """
    protocol = normalize_protocol(protocol)
    if protocol == "pressure":
        raise ValueError("metrics() is a motion solver; pressure rows use v2t_metrics()")
    n = min(len(pred), len(gt), len(keep))
    pred = np.asarray(pred[:n], dtype=np.float64)
    gt = np.asarray(gt[:n], dtype=np.float64)
    keep = np.asarray(keep[:n], dtype=bool)
    out = {key: float("nan") for key in METRICS}
    out["n_valid_frames"] = int(keep.sum())
    if not keep.any():
        return out
    if times is None:
        times = np.arange(n, dtype=np.float64) / float(fps)
    else:
        times = np.asarray(times[:n], dtype=np.float64)
    p, g, t = pred[keep], gt[keep], times[keep]
    pelvis = PROTOCOL_PELVIS[protocol]
    p_aligned, _ = pelvis_align(p, pelvis_indices=pelvis)
    g_aligned, _ = pelvis_align(g, pelvis_indices=pelvis)
    out["mpjpe_mm"] = float(mean_point_error(p_aligned, g_aligned, scale=1000.0).mean())
    try:
        out["pa_mpjpe_mm"] = float(pa_mpjpe(p, g).mean())
    except ValueError:
        pass
    # 2026-10-08 漂移点 o：bvh23 无世界锚（上游设计，根轨迹 = cumsum(位移) 从原点
    # 出发），root_ate_mm 逐序列平移对齐后比形状（rte/path 不受影响，rte 本就内部
    # 对齐）；smpl24 保留绝对世界 ATE。
    traj = root_trajectory_metrics(p[:, 0], g[:, 0])
    if protocol == "bvh23":
        aligned = p[:, 0] - (p[:, 0] - g[:, 0]).mean(axis=0)
        traj["root_ate_mm"] = float(np.linalg.norm(aligned - g[:, 0], axis=-1).mean() * 1000.0)
    out.update(traj)
    out.pop("root_path_length_m", None)
    try:
        out["w_mpjpe100_mm"], out["wa_mpjpe100_mm"] = windowed_world_mpjpe(p, g, window=100)
    except ValueError:
        pass
    temporal = temporal_metrics(p, g, t)
    for key in ("accel_error_m_s2", "jitter_pred_m_s3", "jitter_gt_m_s3"):
        out[key] = float(temporal[key])
    if pred_rotations is not None and gt_rotations is not None:
        pr = np.asarray(pred_rotations[:n])[keep]
        gr = np.asarray(gt_rotations[:n])[keep]
        if pr.shape[-2:] == (3, 3):
            out["mpjae_deg"] = float(matrix_rotation_error_degrees(pr, gr).mean())
            axis = 2 if forward_axis is None else int(forward_axis)
            out["root_orientation_deg"] = float(
                root_orientation_error_deg(pr[:, 0], gr[:, 0], forward_axis=axis).mean()
            )
            out["root_orientation_drift_deg"] = float(
                root_orientation_drift_deg(pr[:, 0], gr[:, 0], forward_axis=axis).mean()
            )
        else:
            out["mpjae_deg"] = float(rotation_error_degrees(pr, gr).mean())
    # Public joint-based foot sliding on the protocol's four foot joints.
    foot_ids = _foot_joint_indices(names, protocol)
    foot_value, foot_count = foot_sliding_joints(p[:, foot_ids], g[:, foot_ids], t)
    if foot_count:
        out["foot_sliding_mm"] = foot_value
    if pred_vertices is not None and gt_vertices is not None:
        pv = np.asarray(pred_vertices[:n])[keep]
        gv = np.asarray(gt_vertices[:n])[keep]
        _, pv_aligned = pelvis_align(p, pv, pelvis_indices=pelvis)
        _, gv_aligned = pelvis_align(g, gv, pelvis_indices=pelvis)
        out["pve_mm"] = float(mean_point_error(pv_aligned, gv_aligned, scale=1000.0).mean())
        # SMPL-only surface diagnostics (not part of METRICS).
        out["shape_vertex_std_mm"] = shape_vertex_std(pv)
        vertex_foot_value, vertex_foot_count = foot_sliding_vertices(pv, gv, t)
        if vertex_foot_count:
            out["foot_sliding_vertex_mm"] = vertex_foot_value
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {k: float("nan") for k in METRICS}; out["n_valid_frames"] = sum(int(r["n_valid_frames"]) for r in rows)
    for k in METRICS:
        vals = [(float(r[k]), int(r["n_valid_frames"])) for r in rows if math.isfinite(float(r[k]))]
        if vals: out[k] = float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))
    return out
