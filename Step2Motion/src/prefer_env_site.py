"""Prefer conda-env packages over ~/.local.

Python 3.8 cannot import newer pymotion from user site-packages because
that copy uses tuple[T, T] annotations. test.py / train.py must import
this module before any pymotion import.
"""
from __future__ import annotations

import sys


def prefer_env_site() -> None:
    kept = []
    for path in sys.path:
        normalized = path.replace("\\", "/")
        if "/.local/lib/python" in normalized and normalized.rstrip("/").endswith("site-packages"):
            continue
        kept.append(path)
    sys.path[:] = kept
    for name in list(sys.modules):
        if name == "pymotion" or name.startswith("pymotion."):
            del sys.modules[name]


prefer_env_site()
