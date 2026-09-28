"""Resolve centralized Anysole workspace paths for MotionPRO.

The canonical URI scheme is the public resolver in
``AnysoleWorkspace/tool/workspace.py``; this module maps the same schemes to
the same roots for MotionPRO-owned code.  ``workspace://`` is kept only as a
deprecated alias for legacy configuration strings.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = Path(os.environ.get("ANYSOLE_WORKSPACE", REPO_ROOT / "AnysoleWorkspace")).expanduser()
RESULTS_ROOT = Path(os.environ.get("ANYSOLE_RESULTS", REPO_ROOT / "results")).expanduser()
DISPLAY_ROOT = Path(os.environ.get("ANYSOLE_RESULTSDISPLAY", REPO_ROOT / "results_display")).expanduser()

# Must stay in sync with the private adapter that writes these inputs:
# AnysoleWorkspace/tool/adapters/MotionPRO/adapter.py ADAPTER_VERSION.
MOTIONPRO_ADAPTER_VERSION = "adapter_v1"

PREFIXES = {
    "model-input://": WORKSPACE_ROOT / "model_inputs",
    "protocol://": WORKSPACE_ROOT / "protocol",
    "shared://": WORKSPACE_ROOT / "shared",
    "asset://": WORKSPACE_ROOT / "assets",
    "work://": WORKSPACE_ROOT / "work",
    "results://": RESULTS_ROOT,
    "display://": DISPLAY_ROOT,
    "workspace://": WORKSPACE_ROOT,  # deprecated alias
}

LEGACY_PREFIXES = {
    "data/exp/checkpoint": RESULTS_ROOT / "MotionPRO/checkpoints",
    "data/exp/result": RESULTS_ROOT / "MotionPRO/metrics",
    "data/exp/viz_compare": DISPLAY_ROOT / "MotionPRO",
    "data/tensorboard": RESULTS_ROOT / "MotionPRO/tensorboard",
    "data/sequences": WORKSPACE_ROOT / "model_inputs" / "MotionPRO" / MOTIONPRO_ADAPTER_VERSION,
    "data/splits": WORKSPACE_ROOT / "protocol/splits/default",
    "data/smpl": WORKSPACE_ROOT / "assets/third_party/smpl",
    "data/cliff_ckpt": WORKSPACE_ROOT / "assets/third_party/MotionPRO/cliff_ckpt",
    "data/mmdetection": WORKSPACE_ROOT / "assets/third_party/MotionPRO/mmdetection",
}


def _resolve_raw_bvh(path: Path) -> Path:
    """Resolve BVHs recorded before the workspace was reorganized."""
    marker = "/mocap_ori_bvh/"
    normalized = path.as_posix()
    if marker not in normalized:
        return path
    suffix = normalized.split(marker, 1)[1]
    matches = sorted((WORKSPACE_ROOT / "raw" / "bvh").glob("*/mocap_ori_bvh/" + suffix))
    return matches[0] if matches else path


def resolve_path(value, base_dir=None) -> str:
    text = str(value or "")
    for prefix, root in PREFIXES.items():
        if text.startswith(prefix):
            return str(root / text[len(prefix):])
    normalized = text.replace("\\", "/").rstrip("/")
    for prefix, root in LEGACY_PREFIXES.items():
        if normalized == prefix or normalized.startswith(prefix + "/"):
            return str(root / normalized[len(prefix):].lstrip("/"))
    path = Path(os.path.expandvars(text)).expanduser()
    if path.is_absolute() or base_dir is None:
        return str(_resolve_raw_bvh(path))
    return str(_resolve_raw_bvh(Path(base_dir) / path))


def sequence_root(cam_id=3) -> Path:
    """Default MotionPRO model input root for one camera."""
    return (WORKSPACE_ROOT / "model_inputs" / "MotionPRO"
            / MOTIONPRO_ADAPTER_VERSION / f"cam{int(cam_id)}")
