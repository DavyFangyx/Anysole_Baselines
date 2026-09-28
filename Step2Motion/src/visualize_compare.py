"""Compatibility entry point for Step2Motion evaluation visualizations.

The maintained renderer lives in ``results_display/script`` so it can also be
run independently after predictions have been generated.  ``test.py`` imports
this module from the Step2Motion ``src`` directory, therefore this thin
adapter keeps that existing API and command-line entry point working.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[3]
_RENDERER = _REPO_ROOT / "results_display" / "script" / "r_test1_visualize_step2motion.py"
if not _RENDERER.is_file():
    raise ImportError("Step2Motion visualization renderer not found: %s" % _RENDERER)

_spec = importlib.util.spec_from_file_location("_step2motion_visualize_renderer", _RENDERER)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load Step2Motion visualization renderer")

_saved_path = list(sys.path)
_saved_modules = {}
try:
    # The renderer resolves ``utils`` (results_display/script) and ``anysole``
    # (repo root) exactly as it does when run standalone; restore sys.path
    # afterwards so its own sys.path edits (archived Step2Motion src for
    # bvh_export) never shadow imports in the importing process.
    sys.path.insert(0, str(_RENDERER.parent))
    sys.path.insert(0, str(_REPO_ROOT))
    # ``src/test.py`` already imports Step2Motion's own ``utils.py`` under the
    # name ``utils``, which would shadow the results_display package the
    # renderer imports from.  Swap the package in for the exec, restore after.
    _utils_pkg_path = _RENDERER.parent / "utils"
    if str(_utils_pkg_path) not in (
            getattr(sys.modules.get("utils"), "__path__", []) or []):
        _saved_modules["utils"] = sys.modules.get("utils")
        _utils_spec = importlib.util.spec_from_file_location(
            "utils", _utils_pkg_path / "__init__.py",
            submodule_search_locations=[str(_utils_pkg_path)])
        _utils_pkg = importlib.util.module_from_spec(_utils_spec)
        sys.modules["utils"] = _utils_pkg
        _utils_spec.loader.exec_module(_utils_pkg)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
except ImportError as exc:
    # Visualization is a post-processing convenience.  Keep model evaluation
    # usable when optional rendering dependencies (for example OpenCV) are not
    # installed in the training environment.
    _module = None
    _missing_dependency = getattr(exc, "name", str(exc))
finally:
    sys.path[:] = _saved_path
    for _name, _orig in _saved_modules.items():
        sys.modules[_name] = _orig

DEFAULT_VIZ_DIR = (
    _module.DEFAULT_VIZ_DIR
    if _module is not None
    else _REPO_ROOT / "results_display" / "Test1_visualization" / "Step2Motion" / "gait_model"
)


def visualize_from_exports(*args, **kwargs):
    if _module is None:
        print(
            "Skipping Step2Motion visualization: optional dependency %s is not installed."
            % _missing_dependency
        )
        return {"skipped": True, "reason": "missing dependency: %s" % _missing_dependency}
    return _module.visualize_from_exports(*args, **kwargs)


def main() -> None:
    if _module is None:
        raise SystemExit(
            "Step2Motion visualization requires optional dependency %s; "
            "model evaluation itself can run without it." % _missing_dependency
        )
    _module.main()


if __name__ == "__main__":
    main()
