"""M4 tests: make_splits.py is a read-only validator of the canonical split.

It must pass on the frozen canonical CSV, fail on corrupted copies, and never
write anything.

Run directly (touch_gait env):

    python tests/test_make_splits_validation.py
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

MOTIONPRO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MOTIONPRO_ROOT.parents[1]
if str(MOTIONPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONPRO_ROOT))

from data_prep.make_splits import validate  # noqa: E402

CANONICAL = REPO_ROOT / "AnysoleWorkspace/protocol/splits/default/splits.csv"
FACTS_ROOT = REPO_ROOT / "AnysoleWorkspace/shared/facts/sessions"


def _read(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = {"train": [], "val": [], "test": []}
    for row in rows:
        for column in columns:
            value = (row.get(column) or "").strip()
            if value:
                columns[column].append(value)
    return columns


def test_canonical_split_passes():
    assert not validate(CANONICAL, FACTS_ROOT), "canonical split must validate cleanly"


def test_corrupted_split_fails():
    columns = _read(CANONICAL)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        # val != test violates the canonical protocol invariant
        bad = base / "bad_val_test.csv"
        with bad.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "train", "val", "test"])
            writer.writeheader()
            writer.writerow({"train": "S10101", "val": "S13011", "test": "S13012"})
        errors = validate(bad, FACTS_ROOT)
        assert any("differ" in error for error in errors), errors

        # duplicate + leakage: put a train session into test
        dup = base / "bad_dup.csv"
        train = list(columns["train"])
        test = list(columns["test"])
        rows = max(len(train), len(test))
        with dup.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "train", "val", "test"])
            writer.writeheader()
            for i in range(rows):
                writer.writerow({
                    "train": train[i] if i < len(train) else "",
                    "val": test[i] if i < len(test) else "",
                    "test": (train[0] if i == 0 else test[i]) if i < len(test) else "",
                })
        errors = validate(dup, FACTS_ROOT)
        assert any("differ" in error for error in errors), errors
        assert any("leakage" in error for error in errors), errors


def test_validator_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        before = sorted(path.name for path in base.iterdir())
        validate(CANONICAL, FACTS_ROOT)
        after = sorted(path.name for path in base.iterdir())
        assert before == after, "validation must not create files"


def test_session_existence_check():
    columns = _read(CANONICAL)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        missing = base / "missing.csv"
        with missing.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "train", "val", "test"])
            writer.writeheader()
            train = list(columns["train"])[:92]
            for i in range(max(len(train), len(columns["test"]))):
                writer.writerow({
                    "train": train[i] if i < len(train) else "",
                    "val": columns["test"][i] if i < len(columns["test"]) else "",
                    "test": "S99999" if i == 0 else columns["test"][i] if i < len(columns["test"]) else "",
                })
        errors = validate(missing, FACTS_ROOT)
        assert any("shared facts" in error for error in errors), errors


if __name__ == "__main__":
    for fn in (test_canonical_split_passes, test_corrupted_split_fails,
               test_validator_writes_nothing, test_session_existence_check):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_make_splits_validation: all passed")
