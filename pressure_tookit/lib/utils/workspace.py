"""Resolve canonical Anysole workspace URIs.

Delegates to the frozen public resolver (``AnysoleWorkspace/tool/workspace.py``)
for raw:// protocol:// shared:// model-input:// work:// asset:// results://.
The historical ``workspace://derived`` scheme is removed; toolkit code must
consume the canonical model_inputs/work trees.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

GAIT_ROOT = Path(__file__).resolve().parents[4]

if str(GAIT_ROOT) not in sys.path:
    sys.path.insert(0, str(GAIT_ROOT))
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

WORKSPACE_ROOT = Path(
    os.environ.get("ANYSOLE_WORKSPACE", GAIT_ROOT / "AnysoleWorkspace")).expanduser()
RESULTS_ROOT = Path(
    os.environ.get("ANYSOLE_RESULTS", GAIT_ROOT / "results")).expanduser()
DISPLAY_ROOT = Path(
    os.environ.get("ANYSOLE_RESULTSDISPLAY", GAIT_ROOT / "results_display")).expanduser()


def resolve_path(value) -> str:
    text = str(value or "")
    if text.startswith("display://"):
        return str(DISPLAY_ROOT / text[len("display://"):])
    return str(resolve_uri(text))
