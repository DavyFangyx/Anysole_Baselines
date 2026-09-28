"""Resolve canonical AnysoleWorkspace paths for Step2Motion.

Step2Motion data is built by
``AnysoleWorkspace/tool/adapters/Step2Motion/build_gait.py`` under
``model_inputs/Step2Motion/<adapter_version>/``; every path resolver here
delegates to the frozen public resolver in
``AnysoleWorkspace/tool/workspace.py``.  The legacy ``workspace://derived``
scheme and the old ``sources/raw`` BVH fallback are gone.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_GAIT_ROOT = Path(__file__).resolve().parents[3]
if str(_GAIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_GAIT_ROOT))

from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

WORKSPACE_ROOT = Path(os.environ.get("ANYSOLE_WORKSPACE", _GAIT_ROOT / "AnysoleWorkspace")).expanduser()
RESULTS_ROOT = Path(os.environ.get("ANYSOLE_RESULTS", _GAIT_ROOT / "results")).expanduser()
DISPLAY_ROOT = Path(os.environ.get("ANYSOLE_RESULTSDISPLAY", _GAIT_ROOT / "results_display")).expanduser()

# display:// is a Step2Motion-private scheme (results_display); the frozen
# public resolver owns every other canonical URI.
_DISPLAY_ROOT = DISPLAY_ROOT


def resolve_path(value, base_dir=None) -> str:
    text = str(value or "")
    if text.startswith("display://"):
        return str(_DISPLAY_ROOT / text[len("display://"):])
    if "://" in text:
        return str(resolve_uri(text))
    path = Path(os.path.expandvars(text)).expanduser()
    if path.is_absolute():
        return str(path)
    if base_dir is not None:
        return str(Path(base_dir) / path)
    # Relative upstream demo paths (e.g. ``data/step2motion``, ``models``)
    # keep their upstream CWD-relative meaning.
    return str(path)


def normalizer_path(config: dict) -> Path:
    """Canonical normalizer location derived from the configured dataset.

    ``<model_inputs>/Step2Motion/<adapter_version>/<normalizer>/
    normalizer_<normalizer>.pth`` where the adapter version is read from the
    configured ``train_data`` path (URI or resolved form), so a config never
    hardcodes the version twice.
    """
    train = str(config.get("train_data") or "")
    resolved = Path(resolve_path(train))
    root = resolved.resolve().parent.parent  # .../Step2Motion/<adapter_version>
    model_root = (WORKSPACE_ROOT / "model_inputs" / "Step2Motion").resolve()
    if model_root not in root.parents:
        raise ValueError(
            f"train_data {train!r} is not under {model_root}; "
            "cannot derive the canonical normalizer path"
        )
    name = str(config["normalizer"])
    return root / name / f"normalizer_{name}.pth"


def resolve_config_paths(config):
    for key in (
        "prior_train_data", "prior_val_data", "prior_test_data",
        "train_data", "val_data", "test_data", "models_dir", "model_dir",
    ):
        if key in config:
            config[key] = resolve_path(config[key])
    return config
