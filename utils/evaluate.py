"""Baseline evaluation entry: predictions + unified solver -> metrics.

Each baseline's evaluation produces not only its prediction files but also the
corresponding evaluation results (2026-09-26 用户裁定：评估不仅要产出文件还
要有对应的评估结果).  This entry reads the exported predictions, loads the
paired GT on the canonical session grid, computes the metrics with the shared
solver (formulas from ``anysole.utils.metrics`` only) and writes, per split:

    results/baselines/<Model>/metrics/<split>_comparison.json   (schema v2:
        per-session rows + weighted aggregate + protocol/capability/provenance
        metadata + split/manifest hashes)
    results/baselines/<Model>/metrics/<split>_comparison.csv    (per-session
        detail for human reading)

results_display (R_Test2/R_Test4) consumes these files — it is not the place
where baseline metrics are produced.

Run (touch_gait env, from the repo root):

    python -m Baselines.utils.evaluate --model all --split test
    python -m Baselines.utils.evaluate --model step2motion --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]  # utils/ -> Baselines/ -> repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from Baselines.utils.gt_loading import (
    archive_meta,
    array_from_file,
    bvh_joints,
    find_prediction,
    load_split_ids,
    protocol_gt,
    protocol_gt_rotations,
    protocol_gt_surface,
    read_manifest,
    resolve_repo_path,
    valid_mask,
    session_time_grid,
    shared_session_frames,
)
from Baselines.utils.protocols import (
    PROTOCOL_FORWARD_AXIS,
    PROTOCOL_LABELS,
    normalize_protocol,
)
from Baselines.utils.solver import (
    COMPARISON_SCHEMA_V2,
    LEGACY_SCHEMA_V1,
    METRICS,
    aggregate,
    load_v2t_archive,
    metrics,
    v2t_metrics,
)

ROOT = _ROOT
WORKSPACE = ROOT / "AnysoleWorkspace"
RESULTS = ROOT / "results"
DEFAULT_MANIFEST = WORKSPACE / "protocol/manifests" / "session_manifest.csv"
DEFAULT_SPLIT_CSV = WORKSPACE / "protocol/splits" / "default" / "splits.csv"
DEFAULT_REGISTRY = ROOT / "results_display" / "models_modes.yaml"

# Short CLI aliases -> registry entry names.
MODEL_ALIASES = {
    "motionpro": "MotionPRO",
    "pressure_toolkit": "MMVP_pressure_toolkit",
    "vp_mocap": "MMVP_VP-MoCap",
    "step2motion": "Step2Motion",
    "fpp_v2t": "MMVP_FPP-Net",
    "fppnet": "MMVP_FPP-Net",
    "wham": "WHAM",
    "sam3db": "SAM3DB",
    "gvhmr": "GVHMR",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_registry(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
        raw = yaml.safe_load(text)
    except ImportError:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"mode registry must be a mapping: {path}")
    entries: list[dict[str, Any]] = []
    for mode, models in raw.items():
        if not isinstance(models, list):
            raise ValueError(f"registry[{mode}] must be a list: {path}")
        for model in models:
            if isinstance(model, dict):
                entries.append({"mode": str(mode), **model})
    return entries


def _capabilities(entry: dict[str, Any], archive: dict[str, Any] | None = None):
    from Baselines.utils.capabilities import (
        applicable_for,
        denied_capabilities,
        read_capabilities,
    )
    caps = read_capabilities(entry.get("capabilities"))
    denied = denied_capabilities(entry, archive)
    applicable = applicable_for(caps, denied, approved=entry.get("approved", ()))
    return caps, denied, applicable


def _session_time_grid(row: dict[str, str]) -> np.ndarray:
    """Canonical time comes from shared facts, with corrections already applied."""
    return session_time_grid(row)


def _gt_source_path(row: dict[str, str], protocol: str) -> str:
    key = "bvh_path" if protocol == "bvh23" else "smpl_path"
    try:
        return str(resolve_repo_path(row[key], ROOT))
    except (KeyError, ValueError):
        return f"{key}:{row.get(key, '')}"


def _resolve_results_uri(value: str) -> Path:
    if value.startswith("results://"):
        # results://baselines/X/... -> ROOT/results/baselines/X/...
        return ROOT / "results" / value[len("results://"):]
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _model_result_root(prediction_root: Path) -> Path:
    """results/baselines/<Model>/predictions/... -> results/baselines/<Model>."""
    parts = prediction_root.parts
    try:
        idx = parts.index("predictions")
    except ValueError:
        raise ValueError(f"prediction_root must contain a predictions/ level: {prediction_root}")
    return Path(*parts[:idx])


def evaluate_motion_row(entry: dict[str, Any], row: dict[str, str],
                        pred_path: Path | None, surface_metrics: bool,
                        gt_cache: dict, gt_surface_cache: dict,
                        gt_rotation_cache: dict) -> dict[str, Any]:
    """One session of a motion row: native protocol, capability-gated metrics."""
    reasons: dict[str, str] = {}
    base = {
        "model": entry.get("name", ""),
        "session_id": row["session_id"],
        "subject_id": row.get("subject_id", ""),
        "action": row.get("action", ""),
        "split": "placeholder",
        "protocol": "", "gt_source": "",
        "n_valid_frames": 0,
        "mode": entry.get("mode", ""),
        "family": str(entry.get("family") or "baseline"),
        "status": "ok", "reason": "", "metric_reasons": {}, "prediction": str(pred_path or ""),
    }
    base.update({key: float("nan") for key in METRICS})
    if pred_path is None:
        base.update({"status": "missing",
                     "reason": f"no prediction under {entry.get('prediction_root', '')}"})
        return base
    archive_info = archive_meta(pred_path)
    caps, denied, applicable = _capabilities(entry, archive_info)
    try:
        declared = normalize_protocol(str(entry.get("protocol") or ""))
        if declared == "pressure":
            raise ValueError("pressure rows use evaluate_pressure_row")
        if pred_path.suffix.lower() == ".bvh":
            # Generated BVH starts at local time zero, not the raw mocap epoch.
            from Baselines.utils.prediction_grid import bvh_prediction_on_grid
            pred, pred_mask, pred_names, bvh_rotations = bvh_prediction_on_grid(pred_path, row)
            protocol = "bvh23"
        else:
            pred, pred_mask, pred_names, protocol_label = array_from_file(pred_path)
            protocol = normalize_protocol(protocol_label)
            if pred_mask is None:
                raise ValueError(
                    "legacy motion archive lacks the unified "
                    "joint_xyz_world/valid_mask contract"
                )
        if protocol != declared:
            raise ValueError(
                f"declared protocol {declared!r} but prediction is "
                f"{PROTOCOL_LABELS[protocol]!r}"
            )
        expected = int(float(row.get("n_frames") or 0))
        if pred_mask is not None and len(pred_mask) != expected:
            raise ValueError(
                f"unified motion archive mask={len(pred_mask)}; canonical session={expected}"
            )
        gt_key = (row["session_id"], protocol)
        if gt_key not in gt_cache:
            gt_cache[gt_key] = protocol_gt(row, protocol)
        gt, gt_names = gt_cache[gt_key]
        base["protocol"] = PROTOCOL_LABELS[protocol]
        base["gt_source"] = _gt_source_path(row, protocol)
        if not caps.get("joint_positions", False):
            raise ValueError("declared joint_positions=false for a motion row")

        eval_fps = float(row.get("target_fps") or 40.0)
        times = _session_time_grid(row)
        n_eval = min(len(pred), len(gt), expected)
        keep = valid_mask(row, n_eval)
        if pred_mask is not None:
            keep &= pred_mask[:n_eval].astype(bool)

        want_rotations = bool(applicable & {"root_orientation_deg", "root_orientation_drift_deg", "mpjae_deg"})
        pred_rotations = gt_rotations = None
        if want_rotations:
            from Baselines.utils.motion_io import load_rotations
            pred_rotation_data = (dict(rotations=bvh_rotations) if pred_path.suffix.lower() == '.bvh'
                                  else load_rotations(pred_path))
            cache_key = (row["session_id"], protocol)
            gt_rotation_data = gt_rotation_cache.setdefault(
                cache_key, protocol_gt_rotations(row, protocol)
            )
            pred_rotations = np.asarray(pred_rotation_data["rotations"], dtype=np.float64)
            gt_rotations = np.asarray(gt_rotation_data["rotations"], dtype=np.float64)
            # 帧数对齐（2026-10-22 S14012 修复）：旋转/表面通道未随 n_eval
            # 截断时，keep 布尔索引与通道长度不一致（canonical 451 帧 vs
            # S2M 导出 450 帧）→ solver 内 boolean index 崩溃。所有通道
            # 一律截断到 n_eval；上游导出少一帧的旋转通道以末帧补齐
            # （S2M BVH 出口一帧短，属上游容差）。
            if len(pred_rotations) == n_eval - 1:
                pred_rotations = np.concatenate(
                    [pred_rotations, pred_rotations[-1:]], axis=0)
            if len(gt_rotations) == n_eval - 1:
                gt_rotations = np.concatenate(
                    [gt_rotations, gt_rotations[-1:]], axis=0)
            pred_rotations = pred_rotations[:n_eval]
            gt_rotations = gt_rotations[:n_eval]

        pred_vertices = gt_vertices = None
        public_surface = str(archive_info.get("public_surface_metrics", "true")).strip().lower()
        if ("pve_mm" in applicable and surface_metrics
                and public_surface not in {"0", "false", "no"}):
            if protocol != "smpl24":
                reasons["pve_mm"] = "not_applicable: no SMPL surface for bvh23"
            else:
                from Baselines.utils.motion_io import load_surface
                pred_surface = load_surface(pred_path)
                if pred_surface is None:
                    if caps.get("predicted_surface", False):
                        raise ValueError(
                            "declared predicted_surface=true but archive has no "
                            "surface/vertex contract"
                        )
                    reasons["pve_mm"] = "not_computed: no surface in archive"
                else:
                    pred_vertices = pred_surface.get("vertices")
                    if pred_vertices is not None:
                        pred_vertices = pred_vertices[:n_eval]
                    cache_key = (row["session_id"], protocol)
                    if cache_key not in gt_surface_cache:
                        gt_surface_cache[cache_key] = protocol_gt_surface(row, protocol)
                    gt_surface = gt_surface_cache[cache_key]
                    gt_vertices = gt_surface.get("vertices") if gt_surface else None
                    if gt_vertices is not None:
                        gt_vertices = gt_vertices[:n_eval]
        elif "pve_mm" in applicable and public_surface in {"0", "false", "no"}:
            reasons["pve_mm"] = "not_applicable: archive public_surface_metrics=false"
        elif "pve_mm" in applicable:
            reasons["pve_mm"] = "not_computed: --no-surface-metrics"

        values = metrics(
            pred[:n_eval], gt[:n_eval], keep, eval_fps,
            protocol=protocol,
            names=pred_names,
            pred_rotations=pred_rotations,
            gt_rotations=gt_rotations,
            pred_vertices=pred_vertices,
            gt_vertices=gt_vertices,
            times=times,
            forward_axis=PROTOCOL_FORWARD_AXIS[protocol],
        )
        base.update(values)
        from Baselines.utils.capabilities import mask_metrics
        mask_metrics(base, applicable, reasons)
        base["n_valid_frames"] = int(keep.sum())
        for metric in METRICS:
            if not math.isfinite(float(base[metric])) and metric in applicable:
                reasons.setdefault(metric, "missing: solver returned no value")
    except FileNotFoundError as exc:
        base.update({"status": "missing", "reason": str(exc)})
    except Exception as exc:
        base.update({"status": "invalid", "reason": str(exc)})
        base.update({key: float("nan") for key in METRICS})
    base["metric_reasons"] = json.dumps(reasons, ensure_ascii=False, sort_keys=True)
    return base


def evaluate_pressure_row(entry: dict[str, Any], row: dict[str, str],
                          pred_path: Path | None) -> dict[str, Any]:
    """One session of a V2T (pressure) row: brief (6) + leaf (4) pressure keys."""
    caps, denied, applicable = _capabilities(entry)
    reasons: dict[str, str] = {}
    base = {
        "model": entry.get("name", ""),
        "session_id": row["session_id"],
        "subject_id": row.get("subject_id", ""),
        "action": row.get("action", ""),
        "split": "placeholder",
        "protocol": "pressure", "gt_source": "",
        "n_valid_frames": 0,
        "mode": entry.get("mode", ""),
        "family": str(entry.get("family") or "baseline"),
        "status": "ok", "reason": "", "metric_reasons": {}, "prediction": str(pred_path or ""),
    }
    base.update({key: float("nan") for key in METRICS})
    try:
        if pred_path is None:
            raise FileNotFoundError(f"no prediction under {entry.get('prediction_root', '')}")
        archive = load_v2t_archive(pred_path)
        expected = int(float(row.get("n_frames") or 0))
        if len(archive["valid_mask"]) != expected:
            raise ValueError(
                f"standard V2T archive length={len(archive['valid_mask'])}; "
                f"canonical session={expected}"
            )
        values = v2t_metrics(archive["pressure_pred"], archive["pressure_gt"],
                             archive["valid_mask"])
        base.update(values)
        from Baselines.utils.capabilities import mask_metrics
        mask_metrics(base, applicable, reasons)
        reasons["contact_f1"] = "not_applicable: no explicit contact prediction (task 01 §4.4)"
        base["metric_reasons"] = json.dumps(reasons, ensure_ascii=False, sort_keys=True)
        base["gt_source"] = "archive embedded pressure_gt (same session grid)"
    except FileNotFoundError as exc:
        base.update({"status": "missing", "reason": str(exc)})
    except Exception as exc:
        base.update({"status": "invalid", "reason": str(exc)})
    return base


def write_comparison(entry: dict[str, Any], split: str, split_csv: Path,
                     manifest_path: Path, rows: list[dict[str, Any]],
                     summary: dict[str, Any], caps: dict[str, bool], denied: set[str],
                     applicable: set[str], gt_source: str, surface_metrics: bool,
                     force: bool) -> Path:
    """Write ``metrics/<split>_comparison.json`` (schema v2) + per-session CSV."""
    result_root = _model_result_root(_resolve_results_uri(str(entry["prediction_root"])))
    output = result_root / "metrics" / f"{split}_comparison.json"
    if output.is_file() and not force:
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
            if str(existing.get("schema_version")) == COMPARISON_SCHEMA_V2:
                if entry.get('protocol') == 'pressure':
                    print(f"skip existing {output} (--force to overwrite)")
                    return output
                if existing.get('time_axis_policy') != 'shared corrections already applied; manifest offset is not reapplied':
                    raise FileExistsError('Existing metrics use the old/undeclared clock: ' + str(output) +
                                          '; preserve them and use a fresh result root or explicitly --force.')
                expected_hashes = {
                    str(row['session_id']): __import__('hashlib').sha256(
                        Path(shared_session_frames(row)['source_path']).read_bytes()).hexdigest()
                    for row in rows if row.get('session_id') and entry.get('protocol') != 'pressure'
                }
                if existing.get('shared_time_hashes') != expected_hashes:
                    raise FileExistsError('Shared time inputs changed; use a fresh result root or explicitly --force.')
                print(f"skip existing {output} (--force to overwrite)")
                return output
            if str(existing.get("schema_version")) == LEGACY_SCHEMA_V1:
                print(f"[legacy] {output}: schema v1 (common19) superseded, rewriting as v2")
        except FileExistsError:
            raise
        except (ValueError, OSError):
            pass
    metric_rows = []
    for row in rows:
        metric_rows.append({
            key: _json_safe(row.get(key))
            for key in ("session_id", "status", "reason", "n_valid_frames", *METRICS)
        })
    payload = {
        "schema_version": COMPARISON_SCHEMA_V2,
        "model": entry.get("name", ""),
        "display_name": entry.get("display_name", entry.get("name", "")),
        "family": str(entry.get("family") or "baseline"),
        "mode": entry.get("mode", ""),
        "split": split,
        "split_csv": str(split_csv),
        "manifest": str(manifest_path),
        "evaluation_protocol": summary.get("protocol", ""),
        "capabilities": {
            field: bool(caps.get(field, False))
            for field in ("joint_positions", "direct_joint_rotations", "root_translation",
                          "root_rotation", "predicted_surface", "predicted_shape",
                          "pressure_prediction", "contact_prediction")
        },
        "capabilities_denied": sorted(denied),
        "applicable_metrics": sorted(applicable),
        "gt_source": gt_source,
        "surface_metrics": bool(surface_metrics),
        "time_axis_source": "shared/facts/sessions/<camera>/<date>/<subject>/<session>/frames.npz:mocap_time_s",
        "time_axis_policy": "shared corrections already applied; manifest offset is not reapplied",
        "shared_time_hashes": {
            str(row['session_id']): __import__('hashlib').sha256(
                Path(shared_session_frames(row)['source_path']).read_bytes()).hexdigest()
            for row in rows if row.get('session_id') and entry.get('protocol') != 'pressure'
        },
        "evaluation_fps": float(summary.get("eval_fps", 40.0)),
        "units": {
            "*_mm": "millimetres",
            "*_deg": "degrees",
            "root_rte_percent": "percent",
            "accel_error_m_s2": "m/s^2",
            "jitter_pred_m_s3": "m/s^3",
            "jitter_gt_m_s3": "m/s^3",
            "pressure_cop_error_*": "normalized sensor-grid units",
        },
        "summary": {
            key: _json_safe(summary.get(key))
            for key in ("n_valid_frames", *METRICS)
        },
        "n_sessions": int(sum(row.get("status") == "ok" for row in rows)),
        "sessions": metric_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_name(f"{split}_comparison.csv")
    fields = ["model", "session_id", "subject_id", "action", "split", "protocol",
              "gt_source", "mode", "family", "status", "reason", "n_valid_frames",
              *METRICS, "prediction", "metric_reasons"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in fields})
    print(f"wrote {output} (+ {csv_path.name})")
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="all",
                        help="registry name or alias, comma-separated: motionpro / "
                             "pressure_toolkit / vp_mocap / step2motion / fpp_v2t / "
                             "all (default: all)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--sessions", default="",
                        help="comma-separated session IDs to restrict the run")
    parser.add_argument("--surface-metrics", dest="surface_metrics", action="store_true",
                        default=True,
                        help="Compute PVE for surface-capable models (default on).")
    parser.add_argument("--no-surface-metrics", dest="surface_metrics", action="store_false",
                        help="Skip SMPL surface loading (faster; pve_mm rows stay —).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing schema-v2 comparison files.")
    args = parser.parse_args(argv)

    entries = _load_registry(args.registry)
    wanted = {MODEL_ALIASES.get(token.strip().lower(), token.strip().lower())
              for token in str(args.model).split(",") if token.strip()}
    selected = [e for e in entries
                if "all" in wanted or str(e.get("name") or "") in wanted]
    if "all" not in wanted and not selected:
        raise SystemExit(f"no registry entry for --model {args.model!r} "
                         f"(registry: {args.registry})")
    ids = load_split_ids(args.split_csv, args.split)
    rows_all = read_manifest(args.manifest, args.split, split_csv=args.split_csv)
    by_id = {str(r["session_id"]): r for r in rows_all}
    if args.sessions:
        session_filter = {s.strip() for s in args.sessions.split(",") if s.strip()}
        rows_all = [r for r in rows_all if r["session_id"] in session_filter]

    for entry in selected:
        name = entry.get("name", "?")
        root = _resolve_results_uri(str(entry.get("prediction_root") or ""))
        pattern = entry.get("pattern") or None
        print(f"[{name}] split={args.split} sessions={len(rows_all)} root={root}")
        gt_cache: dict = {}
        gt_surface_cache: dict = {}
        gt_rotation_cache: dict = {}
        caps, denied, applicable = _capabilities(entry)
        rows: list[dict[str, Any]] = []
        for row in rows_all:
            pred_path = find_prediction(root, str(row["session_id"]), pattern)
            if str(entry.get("mode")) == "V2T":
                result = evaluate_pressure_row(entry, row, pred_path)
            else:
                result = evaluate_motion_row(
                    entry, row, pred_path, args.surface_metrics,
                    gt_cache, gt_surface_cache, gt_rotation_cache,
                )
            result["split"] = args.split
            rows.append(result)
        ok_rows = [r for r in rows if r["status"] == "ok"]
        summary = aggregate(ok_rows)
        summary.update({
            "model": name,
            "n_sessions": len(ok_rows),
            "protocol": PROTOCOL_LABELS[normalize_protocol(str(entry.get("protocol") or ""))]
            if str(entry.get("protocol") or "").lower() != "pressure" else "pressure",
            "eval_fps": 40.0,
        })
        gt_source = "archive embedded pressure_gt (same session grid)" \
            if str(entry.get("mode")) == "V2T" else "native-protocol paired GT (manifest paths)"
        write_comparison(entry, args.split, args.split_csv, args.manifest,
                         rows, summary, caps, denied, applicable, gt_source,
                         args.surface_metrics, args.force)
        missing = sum(r["status"] == "missing" for r in rows)
        invalid = sum(r["status"] == "invalid" for r in rows)
        print(f"[{name}] ok={len(ok_rows)} missing={missing} invalid={invalid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
