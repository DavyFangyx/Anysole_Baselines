#!/usr/bin/env python3
"""Validate the canonical split for MotionPRO consumption (read-only).

MotionPRO reads the canonical split and must never write it (M4).  This
command only checks the structural invariants MotionPRO relies on:

  - val equals test;
  - no duplicate session ids within or across columns;
  - subjects are disjoint between train and val/test (no leakage);
  - every listed session exists in shared facts.

It reports the actual column counts without judging them: the frozen 92/12/36
declaration lives in Agent A's ``shared_schema.CANONICAL_SPLIT_COUNTS`` and
the canonical CSV is Agent A's artifact, so count drift is reported to the
delivery report rather than enforced here.

It writes nothing and exits 1 on any failure.  Do not use this script to
create or modify splits; the canonical file lives at
``AnysoleWorkspace/protocol/splits/default/splits.csv``.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

MOTIONPRO_ROOT = Path(__file__).resolve().parents[1]
if str(MOTIONPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONPRO_ROOT))

from lib.util.workspace import WORKSPACE_ROOT, resolve_path

CANONICAL_SPLIT = WORKSPACE_ROOT / "protocol/splits/default/splits.csv"
FACTS_ROOT = WORKSPACE_ROOT / "shared/facts/sessions"
SESSION_RE = re.compile(r"^(S\d+?)(\d{2})(\d)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the canonical split (read-only).")
    parser.add_argument("--split-csv", type=str, default=str(CANONICAL_SPLIT),
                        help="Split CSV to validate (default: canonical protocol path).")
    parser.add_argument("--facts-root", type=str, default=str(FACTS_ROOT),
                        help="Shared facts root for session existence checks.")
    return parser.parse_args()


def session_subject(session_id: str) -> str:
    match = SESSION_RE.match(session_id)
    if match is None:
        raise ValueError(f"Unrecognized session id: {session_id}")
    return match.group(1)


def read_columns(path: Path) -> dict[str, list[str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = defaultdict(list)
    for row in rows:
        for column in ("train", "val", "test"):
            value = (row.get(column) or "").strip()
            if value:
                columns[column].append(value)
    return dict(columns)


def validate(path: Path, facts_root: Path) -> list[str]:
    errors: list[str] = []
    if not Path(path).is_file():
        return [f"split CSV not found: {path}"]
    columns = read_columns(Path(path))

    val_set, test_set = set(columns.get("val", [])), set(columns.get("test", []))
    if val_set != test_set:
        errors.append("val and test columns differ (canonical protocol sets val=test)")

    all_ids = [sid for ids in columns.values() for sid in ids]
    # val==test by protocol, so duplicates are only an error *within* one
    # column; a train session leaked into val/test is caught by the subject
    # disjointness check below.
    for column in ("train", "val", "test"):
        ids = columns.get(column, [])
        duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
        if duplicates:
            errors.append(f"duplicate session ids in {column}: {duplicates}")

    train_subjects = {session_subject(sid) for sid in columns.get("train", [])}
    eval_subjects = {session_subject(sid) for sid in columns.get("val", []) + columns.get("test", [])}
    leakage = sorted(train_subjects & eval_subjects)
    if leakage:
        errors.append(f"subject leakage between train and val/test: {leakage}")

    facts_sessions = {
        path.parent.name
        for path in Path(facts_root).glob("cam3/*/*/*/session.json")
    }
    missing = sorted({sid for sid in all_ids if sid not in facts_sessions})
    if missing:
        errors.append(f"sessions missing from shared facts: {missing}")
    return errors


def main() -> int:
    args = parse_args()
    split_path = Path(resolve_path(args.split_csv, MOTIONPRO_ROOT))
    facts_root = Path(resolve_path(args.facts_root, MOTIONPRO_ROOT))
    errors = validate(split_path, facts_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"split validation FAILED: {split_path}", file=sys.stderr)
        return 1
    counts = {column: len(ids) for column, ids in read_columns(split_path).items()}
    print(f"split ok (read-only validation): {split_path} counts={counts}")
    try:
        from AnysoleWorkspace.tool.shared_schema import CANONICAL_SPLIT_COUNTS
        drift = {column: (declared, counts.get(column))
                 for column, declared in CANONICAL_SPLIT_COUNTS.items()
                 if counts.get(column) != declared}
        if drift:
            print(f"WARNING: counts differ from the frozen declaration "
                  f"{CANONICAL_SPLIT_COUNTS}: {drift} (report to Agent A)")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
