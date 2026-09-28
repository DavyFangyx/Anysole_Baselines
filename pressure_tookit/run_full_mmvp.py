#!/usr/bin/env python3
"""Run native pressure_toolkit fitting for canonical shared sessions.

This launcher only schedules ``main_singleview.py``; it never touches the
fitting math.  Two entry points:

* **queue mode** (``--sessions``, used by ``configs/run.sh``): exactly one
  session per invocation, which is what the shared task queue hands out.  The
  conf supplies ``--run-root`` / ``--output-dir``; the launcher adds what a bare
  ``main_singleview.py`` call cannot do on its own -- depth pre-flight,
  per-subject ``init_shape`` (computed once and reused), first-missing-frame
  resume, skip-existing, the run artifact and the standard prediction export.
* **batch mode** (default): every discovered session of the requested
  subjects/dates, with ``--gpu-list`` / ``--per-gpu`` process slots.

Fitting frames are the shared frame ids listed in each session's
``frame_ids.npy`` (valid and non-fake).  Process artifacts land under
``work://pressure_toolkit/<run_id>/`` and formal predictions are exported by
``AnysoleWorkspace/tool/export_baseline_motion.py`` into
``results/baselines/pressure_toolkit/predictions/eval_motion/``.

Run-root invariant
------------------
``OUTPUT_DIR`` is always ``<run_root>/fitting`` and ``INIT_DATA_DIR`` is always
``<run_root>``.  ``lib/dataextra/data_loader.py`` reads the per-subject shape
file from ``<init_root>/fitting/results/<date>/<subject>/init_shape_<sub>.npz``
(line 265), the CLIFF initialization from ``<init_root>/initialization/<date>/
<subject>/<seq>/<seq>_cliff_hr48.npz`` (line 272) and the previous frame of a
tracking session from ``<init_root>/fitting/results/<date>/<subject>/<seq>/
smpl_<prev>.npz`` (line 398), while ``main_singleview.py`` writes its results
under ``<output_dir>/results/...``.  A conf that points the two at unrelated
roots silently breaks tracking or resumes from a stale tree, so both arguments
are validated against the invariant at startup.  ``<run_root>/initialization``
is a read-only symlink to the adapter tree.

Pre-flight gate
---------------
A session is fittable only while every frame from the start of its declared
list is materialized (depth, depth_mask, insole, keypoints) *and* the
session/subject level inputs exist (``calibration.npy``, ``floor_<subject>.npy``,
the CLIFF npz).  ``main_singleview.py`` fits one ``[start, end)`` interval and
both the P7 previous-contact rule and the tracking init read the *preceding
fitting frame* of the session, so the fittable range is the longest prefix of
the declared frame list whose inputs are all present; anything after the first
hole is left for a later run.  Sessions that fail the gate are recorded in the
run artifact with a reason and skipped -- never fitted, never crashed.

Formal 口径 (M1/M2/M3 rulings, 2026-09-28)
-----------------------------------------
* **ICP correspondence device = cpu.**  The GPU kNN helpers
  (``lib/fitSMPL/depthTerm.py:34-69``) return *squared* distances
  (``|q|^2 - 2 q.r + |r|^2``) while ``findCorrsGPU`` gates them with the metric
  ``icp_dist_thresh=0.05`` (``depthTerm.py:370``) -- an effective 0.224 m gate
  instead of 5 cm, so the GPU path keeps far more correspondences than the
  numpy/cKDTree path (``findCorrs``, whose ``cKDTree.query`` returns true metric
  distances).  The defect is known and deliberately NOT patched here; GPU is
  therefore not the formal 口径 and is reachable only through the explicit
  ``--icp-device cuda`` acceleration switch, whose numbers must never be mixed
  into formal tables.
* **canvas = 640x576** (``--depth_size``) with the native intrinsics rescaled to
  that canvas (``lib/core/fit_single_frame.py:150-154``); the depth frames
  themselves are native 1624x1240.
* **maxiters = 101** (the config default).

None of the three is a free knob: the run artifact's ``parameters`` always carry
the values actually used, and the artifact writer refuses to merge a run root
whose parameters drift from the ones already recorded there.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
import csv
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLKIT_ROOT = Path(__file__).resolve().parent
MAIN_ENTRY = TOOLKIT_ROOT / "main_singleview.py"
CONFIG = TOOLKIT_ROOT / "configs" / "fit_smpl_rgbd.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from AnysoleWorkspace.tool.workspace import (  # noqa: E402
    WorkspacePathError,
    canonical_uri,
    resolve_uri,
)

# Canonical URIs: every workspace-owned path is resolved through the frozen
# resolver, so a repository escape is rejected instead of silently creating a
# second tree.  Only the toolkit's own entry script/config stay __file__-relative.
ADAPTER_URI = "model-input://pressure_toolkit/v1"
WORK_ROOT_URI = "work://pressure_toolkit"
MANIFEST_URI = "protocol://manifests/session_manifest.jsonl"
SPLITS_URI = "protocol://splits/default/splits.csv"
DEPTH_FRONTEND_URI = "shared://frontends/depthpro/v1"
BASDIR = ADAPTER_URI
ESSENTIAL_ROOT = "asset://third_party/pressure_toolkit/essential"
# AnysoleWorkspace is not itself a URI root (raw/protocol/shared/model-input/
# work/asset/results are); derive it from a canonical root instead of hardcoding:
# work://pressure_toolkit -> <workspace>/work/pressure_toolkit -> parents[1].
WORKSPACE_ROOT = resolve_uri(WORK_ROOT_URI).parents[1]
EXPORTER = WORKSPACE_ROOT / "tool" / "export_baseline_motion.py"

# Formal 口径 defaults (see the module docstring).
FORMAL_ICP_DEVICE = "cpu"
FORMAL_CANVAS = (640, 576)
FORMAL_MAXITERS = 101
NATIVE_FRAME_SIZE = (1624, 1240)
CONFIG_MAXITERS_FALLBACK = 101
ARTIFACT_SCHEMA = "work.pressure_toolkit.run.v1"

# Concurrent prints from the fitting-worker threads corrupt buffered stdout
# under this interpreter, so terminal prints from threads go through this lock.
_PRINT_LOCK = threading.Lock()


def split_subjects(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def manifest_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with resolve_uri(MANIFEST_URI).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["session_id"]] = row
    return rows


def split_sessions(split: str) -> set[str]:
    columns = ("train", "val", "test") if split == "all" else (split,)
    result = set()
    with resolve_uri(SPLITS_URI).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for column in columns:
                value = (row.get(column) or "").strip()
                if value:
                    result.add(value)
    return result


def fitting_frame_ids(adapter: Path, date: str, subject: str,
                      session: str) -> list[int]:
    """Shared frame ids the adapter declared fittable (valid, non-fake)."""
    path = adapter / "images" / date / subject / session / "frame_ids.npy"
    if not path.is_file():
        return []
    return [int(x) for x in np.load(path)]


def discover_sessions(subjects: set[str], dates: str,
                      canonical: set[str]) -> list[tuple[str, str, str]]:
    """(date, subject, session) tuples from the adapter tree + frozen split.

    A session listed here is only a *candidate*: the pre-flight gate decides
    whether it can actually be fitted.
    """
    rows = manifest_rows()
    requested_dates = {d for d in dates.split(",") if d} if dates else set()
    found = []
    images = resolve_uri(ADAPTER_URI) / "images"
    if not images.is_dir():
        return found
    for date_dir in sorted(images.iterdir()):
        if requested_dates and date_dir.name not in requested_dates:
            continue
        for subject_dir in sorted(date_dir.iterdir()):
            if subject_dir.name not in subjects:
                continue
            for session_dir in sorted(subject_dir.iterdir()):
                if session_dir.name not in canonical:
                    continue
                if session_dir.name not in rows:
                    continue
                found.append((date_dir.name, subject_dir.name, session_dir.name))
    return found


# ---------------------------------------------------------------------------
# pre-flight gate: session/subject inputs + per-frame depth coverage
# ---------------------------------------------------------------------------
def _stem_ids(directory: Path, suffixes: tuple[str, ...]) -> set[int]:
    if not directory.is_dir():
        return set()
    found = set()
    for path in directory.iterdir():
        if path.suffix.lower() not in suffixes:
            continue
        try:
            found.add(int(path.stem))
        except ValueError:
            continue
    return found


INPUT_DIRS = {
    "depth": ("depth", (".png",)),
    "depth_mask": ("depth_mask", (".png",)),
    "insole": ("insole", (".npy",)),
}
INPUT_SUBJECT_DIRS = {
    "keypoints": ("keypoints", (".npy",)),
}


def session_inputs(adapter: Path, date: str, subject: str,
                   session: str) -> dict[str, set[int]]:
    """Per-frame input coverage, one entry per required input directory."""
    session_dir = adapter / "images" / date / subject / session
    coverage = {
        name: _stem_ids(session_dir / sub, suffixes)
        for name, (sub, suffixes) in INPUT_DIRS.items()
    }
    coverage.update({
        name: _stem_ids(adapter / "input" / subject / session / sub, suffixes)
        for name, (sub, suffixes) in INPUT_SUBJECT_DIRS.items()
    })
    return coverage


def session_level_missing(adapter: Path, date: str, subject: str,
                          session: str) -> str:
    """Session/subject inputs the child needs before it can fit any frame."""
    session_dir = adapter / "images" / date / subject / session
    if not (session_dir / "calibration.npy").is_file():
        return "no-calibration"
    if not (adapter / "annotations" / date / "floor_info" /
            f"floor_{subject}.npy").is_file():
        return "no-floor-info"
    if not (adapter / "initialization" / date / subject / session /
            f"{session}_cliff_hr48.npz").is_file():
        return "no-cliff-initialization"
    return ""


def fittable_frames(frame_ids: list[int], materialized: set[int],
                    start_idx: int, end_idx: int) -> list[int]:
    """Longest fittable prefix of the declared frame list, trimmed to the range.

    The prefix rule is the P7/tracking constraint: frame *n* of a session reads
    the result (pose, contact) of the preceding fitting frame, so fitting may
    only continue up to the first frame whose inputs are not materialized.
    """
    prefix: list[int] = []
    for frame_id in frame_ids:
        if frame_id not in materialized:
            break
        prefix.append(frame_id)
    upper = end_idx if end_idx is not None and end_idx >= 0 else None
    return [f for f in prefix
            if f >= start_idx and (upper is None or f < upper)]


def plan_session(adapter: Path, date: str, subject: str, session: str,
                 frame_ids: list[int], start_idx: int, end_idx: int) -> dict:
    """Pre-flight one session: what can be fitted, and why not.

    The recorded ``reason`` names every missing prerequisite (``+``-joined), so
    an operator sees whether a session waits on depth, a mask, the floor info or
    the CLIFF initialization without opening the tree.
    """
    record = {
        "date": date, "subject": subject, "session": session,
        "declared": len(frame_ids), "materialized": 0, "fitted": 0,
        "frame_start": None, "frame_stop": None, "first_hole": None,
        "partial": False, "status": "skipped", "reason": "", "frames": [],
    }
    if not frame_ids:
        record["reason"] = "no-frame-ids"
        return record
    declared = set(frame_ids)
    coverage = session_inputs(adapter, date, subject, session)
    usable = set(declared)
    missing: list[str] = []
    for name in ("depth", "depth_mask", "insole", "keypoints"):
        if not (coverage[name] & declared):
            missing.append(f"no-{name.replace('_', '-')}")
        usable &= coverage[name]
    session_reason = session_level_missing(adapter, date, subject, session)
    if session_reason:
        missing.append(session_reason)

    materialized = [f for f in frame_ids if f in usable]
    frames = fittable_frames(frame_ids, usable, start_idx, end_idx)
    record["materialized"] = len(materialized)
    record["partial"] = len(materialized) < len(frame_ids)
    record["first_hole"] = (frame_ids[len(materialized)]
                            if record["partial"] else None)
    if not frames:
        if missing:
            record["reason"] = "+".join(missing)
        elif not materialized:
            record["reason"] = "no-materialized-frames"
        else:
            record["reason"] = "no-frames-in-range"
        return record
    record.update({
        "fitted": len(frames),
        "frame_start": frames[0],
        "frame_stop": frames[-1] + 1,
        "status": "ready",
        "frames": frames,
    })
    parts = list(missing)
    if record["partial"]:
        parts.append(f"partial-inputs-from-frame-{record['first_hole']}")
    record["reason"] = "+".join(parts)
    return record


def depth_frontend_sessions() -> dict[str, int]:
    """DepthPro frontend sessions -> number of depth frames produced.

    Diagnostic only: it shows how many sessions are waiting for
    ``build_inputs.py`` to link the frontend depth tree into ``model_inputs``.
    """
    root = resolve_uri(DEPTH_FRONTEND_URI)
    found: dict[str, int] = {}
    if not root.is_dir():
        return found
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue
        for subject_dir in sorted(date_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            for session_dir in sorted(subject_dir.iterdir()):
                depth = session_dir / "depth"
                if depth.is_dir():
                    found[session_dir.name] = len(_stem_ids(depth, (".png",)))
    return found


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def session_result_dir(output_dir: Path, date: str, subject: str,
                       session: str) -> Path:
    return output_dir / "results" / date / subject / session


def shape_path(run_root: Path, date: str, subject: str) -> Path:
    return run_root / "fitting" / "results" / date / subject / f"init_shape_{subject}.npz"


def session_result_count(output_dir: Path, date: str, subject: str,
                         session: str, since: float) -> int:
    """Count smpl_*.npz results written after ``since`` (unix epoch).

    Read-only helper for the progress display: frames produced by this run are
    told apart from pre-existing results by file mtime.
    """
    root = session_result_dir(output_dir, date, subject, session)
    if not root.is_dir():
        return 0
    count = 0
    for path in root.iterdir():
        if path.suffix.lower() != ".npz" or not path.name.startswith("smpl_"):
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= since:
            count += 1
    return count


def first_missing_frame(output_dir: Path, date: str, subject: str,
                        session: str, frames: list[int]) -> int | None:
    root = session_result_dir(output_dir, date, subject, session)
    for frame_id in frames:
        if not (root / f"smpl_{frame_id:06d}.npz").is_file():
            return frame_id
    return None


def command(args: argparse.Namespace, run_root: Path, output_dir: Path,
            date: str, subject: str, session: str, gender: str, stage: str,
            start_idx: int, end_idx: int) -> list[str]:
    result = [
        sys.executable, str(MAIN_ENTRY), "-c", str(CONFIG),
        "--dataset", date, "--sub_ids", subject, "--seq_name", session,
        "--fitting_stage", stage, "--start_idx", str(start_idx),
        "--end_idx", str(end_idx),
        "--color_size", str(NATIVE_FRAME_SIZE[0]), str(NATIVE_FRAME_SIZE[1]),
        "--depth_size", str(args.canvas[0]), str(args.canvas[1]),
        # Canonical URIs on purpose: the child resolves them through the same
        # frozen resolver, so a mistyped root cannot land outside the workspace.
        "--output_dir", canonical_uri(output_dir),
        "--init_data_dir", canonical_uri(run_root),
        "--basdir", BASDIR,
        "--essential_root", ESSENTIAL_ROOT,
        "--model_gender", gender,
    ]
    if args.maxiters is not None:
        result += ["--maxiters", str(args.maxiters)]
    if args.depth_approx_stride > 1:
        result += ["--depth-approx-stride", str(args.depth_approx_stride)]
    if args.skip_mesh_export:
        result.append("--skip-mesh-export")
    if args.skip_gt_depth_export:
        result.append("--skip-gt-depth-export")
    if args.no_export_obj:
        result.append("--no-export-obj")
    if args.reuse_session_resources:
        result.append("--reuse-session-resources")
    return result


def export_session(session: str, run_root: Path, dry_run: bool = False) -> None:
    """Standard prediction export into results/baselines/pressure_toolkit/.

    Only called for sessions that are complete: the exporter zero-fills frames
    that have no result yet (``export_pressure`` allocates ``n_frames`` from the
    manifest and marks missing ones as unavailable), so exporting a partial
    session would publish a mostly-empty motion under a full-session name.
    """
    cmd = [
        sys.executable, str(EXPORTER), "--model", "pressure_toolkit",
        "--session", session,
        "--pressure-root", canonical_uri(run_root / "fitting"),
    ]
    if dry_run:
        with _PRINT_LOCK:
            print("$ " + " ".join(cmd))
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


# ---------------------------------------------------------------------------
# run artifact: single source of truth for the run's 口径
# ---------------------------------------------------------------------------
@contextmanager
def file_lock(path: Path, blocking: bool = True, message: str = ""):
    """flock-backed mutex; the lock dies with the process, so it never goes stale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(fd, flags)
        except OSError:
            holder = ""
            try:
                with open(path, encoding="utf-8") as handle:
                    holder = handle.read().strip()
            except OSError:
                pass
            raise SystemExit(
                f"{message or 'lock is held'} ({path}, holder PID {holder or 'unknown'})")
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        yield fd
    finally:
        os.close(fd)


def lock_path(run_root: Path, scope: str) -> Path:
    return run_root / ".locks" / f"{scope}.lock"


def _fingerprint(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _artifact_sessions(sessions: dict) -> dict:
    """Session records without the in-memory frame list."""
    return {key: {k: v for k, v in record.items() if k != "frames"}
            for key, record in sessions.items()}


def write_run_artifact(run_root: Path, parameters: dict, sessions: dict,
                       run_meta: dict, requests: list) -> None:
    """Merge this invocation's record into ``<run_root>/artifact.json``.

    Several queue confs share one run root, so the write is mutex-protected and
    idempotent: sessions accumulate, requests accumulate, run metadata is
    first-wins, and any drift of the 口径 parameters is an error instead of a
    silent overwrite.
    """
    path = run_root / "artifact.json"
    with file_lock(lock_path(run_root, "artifact"), blocking=True):
        existing: dict = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
        previous = existing.get("parameters") or {}
        if previous and _fingerprint(previous) != _fingerprint(parameters):
            drifted = sorted(
                key for key in set(previous) | set(parameters)
                if previous.get(key) != parameters.get(key))
            raise SystemExit(
                f"run artifact 口径 drift at {run_root}: {drifted}; refusing to "
                "merge two different 口径 into one run root")
        merged_sessions = dict(existing.get("sessions") or {})
        merged_sessions.update(_artifact_sessions(sessions))
        # frame_start/frame_stop are the half-open fitting range of each session;
        # skipped records carry None and contribute nothing.
        spans = [(int(record["frame_start"]), int(record["frame_stop"]) - 1)
                 for record in merged_sessions.values()
                 if record.get("frame_start") is not None
                 and record.get("frame_stop")]
        declared = {value for span in spans for value in span}
        merged_requests = list(existing.get("requests") or [])
        seen = {_fingerprint(item) for item in merged_requests}
        for request in requests:
            if _fingerprint(request) not in seen:
                merged_requests.append(request)
                seen.add(_fingerprint(request))
        artifact = {
            "schema_version": ARTIFACT_SCHEMA,
            "producer": "run_full_mmvp.py",
            "parameters": parameters,
            "run_meta": {**(existing.get("run_meta") or {}), **run_meta},
            "requests": merged_requests,
            "sessions": merged_sessions,
            "session_count": len(merged_sessions),
            "frame_count": sum(int(record.get("fitted") or 0)
                               for record in merged_sessions.values()),
            "source_artifacts": [ADAPTER_URI],
            "source_hashes": {},
            # Frame-id span of the sessions in this artifact, from the adapter's
            # declared lists (the merged sessions carry their own frames).
            "frame_id_min": min(declared, default=None),
            "frame_id_max": max(declared, default=None),
            "consumers": ["export_baseline_motion.py"],
        }
        path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# process execution
# ---------------------------------------------------------------------------
def thread_cap_for(slots: int) -> int:
    """Per-process parallel-thread cap for ``slots`` concurrent fitting processes.

    Each fitting process would otherwise spawn os.cpu_count() threads for its
    kd-tree queries (plus BLAS threads), so with N slots the whole box gets
    N*cores threads and oversubscribes.  Give each process cores/N.
    """
    return max(2, (os.cpu_count() or 4) // max(1, slots))


def resolve_kdtree_workers(args: argparse.Namespace, parallelism: int) -> int:
    """Effective PRESSURE_KDTREE_WORKERS for this invocation.

    ``configs/run.sh`` exports it from the conf (concurrency-linked, so a
    96-core box is not oversubscribed by N processes); without the conf the same
    relation is derived from the parallel capacity (slots x per-gpu).
    """
    env = os.environ.get("PRESSURE_KDTREE_WORKERS", "").strip()
    if env:
        try:
            return max(2, int(env))
        except ValueError:
            raise SystemExit(f"PRESSURE_KDTREE_WORKERS must be an integer: {env!r}")
    return thread_cap_for(parallelism)


def effective_icp_device(flag: str) -> str:
    """The env is authoritative: run.sh exports the conf's fixed 口径."""
    env = os.environ.get("PRESSURE_ICP_DEVICE", "").strip().lower()
    if env and env != flag:
        print(f"WARNING: PRESSURE_ICP_DEVICE={env} overrides --icp-device {flag}",
              file=sys.stderr)
        return env
    return flag


def run(cmd: list[str], gpu: str, kdtree_workers: int, icp_device: str,
        log_path: Path | None = None) -> None:
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    # Bound implicit multithreading: scipy cKDTree workers plus BLAS backends.
    env["PRESSURE_KDTREE_WORKERS"] = str(kdtree_workers)
    env.setdefault("OMP_NUM_THREADS", str(kdtree_workers))
    env.setdefault("OPENBLAS_NUM_THREADS", str(kdtree_workers))
    env.setdefault("MKL_NUM_THREADS", str(kdtree_workers))
    env.setdefault("NUMEXPR_NUM_THREADS", str(kdtree_workers))
    # Set (not setdefault): the value the run artifact records must be the value
    # the fitting process actually used.
    env["PRESSURE_ICP_DEVICE"] = icp_device
    # Capture each session's output to its own log file: a concurrent batch
    # otherwise loses a child's traceback in the terminal and the failure can
    # not be diagnosed.
    if log_path is None:
        subprocess.run(cmd, cwd=TOOLKIT_ROOT, env=env, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        subprocess.run(cmd, cwd=TOOLKIT_ROOT, env=env, check=True,
                       stdout=handle, stderr=subprocess.STDOUT)


def ensure_initialization_link(run_root: Path) -> Path:
    """CLIFF initialization stays in the adapter tree; the run root only links it."""
    source = resolve_uri(ADAPTER_URI) / "initialization"
    target = run_root / "initialization"
    if target.is_symlink() or target.exists():
        return target
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source, target_is_directory=True)
    except FileExistsError:  # a concurrent conf won the race
        return target
    return target


class ProgressMonitor(threading.Thread):
    """Terminal-only progress display: one bar per subject plus a total line.

    Runs as a daemon thread while sessions fit, counting ``smpl_*.npz``
    results on disk (frames written by this run are told apart from
    pre-existing ones by file mtime).  It reads the filesystem and writes
    stdout exclusively -- scheduling and the fitting processes are never
    touched.  The main thread prints nothing while the monitor is alive, so
    no locking is needed.
    """

    BAR_WIDTH = 20
    BAR_FILL = "█"
    BAR_EMPTY = "░"

    def __init__(self, output_dir: Path, subjects: dict, since: float,
                 started: float, interval: float = 1.0):
        super().__init__(daemon=True, name="progress-monitor")
        self._output_dir = output_dir
        self._subjects = subjects
        self._since = since
        self._run_start = started
        self._interval = interval
        self._done_initial = sum(info["done_initial"] for info in subjects.values())
        self._stop_event = threading.Event()
        self._lines = 0
        self._reported = set()
        self._tty = sys.stdout.isatty()

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None:
            return "--:--"
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @classmethod
    def _bar(cls, ratio: float) -> str:
        filled = int(round(max(0.0, min(1.0, ratio)) * cls.BAR_WIDTH))
        return cls.BAR_FILL * filled + cls.BAR_EMPTY * (cls.BAR_WIDTH - filled)

    def _scan(self) -> bool:
        """Refresh done counts from disk; return True if anything changed."""
        changed = False
        now = time.monotonic()
        for subject, info in self._subjects.items():
            if info["end"] is not None:
                continue
            done = 0
            for (date, session), meta in info["sessions"].items():
                if not meta["live"]:
                    done += meta["base"]
                else:
                    done += meta["base"] + session_result_count(
                        self._output_dir, date, subject, session, since=self._since)
            done = min(done, info["total"])
            if done != info["done"]:
                info["done"] = done
                changed = True
            if done >= info["total"]:
                info["end"] = now
                changed = True
        return changed

    @staticmethod
    def _visible(info: dict) -> bool:
        return info["done_initial"] < info["total"]

    def _line(self, subject: str, info: dict, now: float) -> str:
        done, total = info["done"], info["total"]
        ratio = done / total if total else 0.0
        if info["end"] is not None:
            return (f"✓ {subject:<6}[{self._bar(1.0)}] {done:>6}/{total:<6} 100%    "
                    f"in {self._format_duration(info['end'] - info['start'])}")
        elapsed = now - info["start"]
        new_frames = done - info["done_initial"]
        rate = new_frames / elapsed if elapsed > 0 and new_frames > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else None
        return (f"  {subject:<6}[{self._bar(ratio)}] {done:>6}/{total:<6} "
                f"{ratio * 100:5.1f}%  {self._format_duration(elapsed)} "
                f"< {self._format_duration(eta)}")

    def _global_line(self, now: float) -> str:
        done = sum(info["done"] for info in self._subjects.values())
        total = sum(info["total"] for info in self._subjects.values())
        ratio = done / total if total else 0.0
        elapsed = now - self._run_start
        new_frames = done - self._done_initial
        rate = new_frames / elapsed if elapsed > 0 and new_frames > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else None
        return (f"total [{self._bar(ratio)}] {done:>6}/{total:<6} "
                f"{ratio * 100:5.1f}%  {self._format_duration(elapsed)} "
                f"< {self._format_duration(eta)}")

    def _render(self, final: bool = False) -> None:
        now = time.monotonic()
        lines = []
        for subject, info in self._subjects.items():
            if not self._visible(info):
                continue
            if not self._tty and final and subject in self._reported:
                continue
            lines.append(self._line(subject, info, now))
        lines.append(self._global_line(now))
        out = sys.stdout
        if self._tty:
            if self._lines:
                out.write(f"\033[{self._lines}A")
            for line in lines:
                out.write("\033[2K" + line + "\n")
            self._lines = len(lines)
        else:
            for line in lines:
                out.write(line + "\n")
        out.flush()

    def _report_completed(self) -> None:
        now = time.monotonic()
        out = sys.stdout
        for subject, info in self._subjects.items():
            if (info["end"] is not None and self._visible(info)
                    and subject not in self._reported):
                self._reported.add(subject)
                out.write(self._line(subject, info, now) + "\n")
        out.flush()

    def run(self) -> None:
        while not self._stop_event.wait(self._interval):
            changed = self._scan()
            if self._tty:
                self._render()
            elif changed:
                self._report_completed()

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5.0)
        self._scan()
        self._render(final=True)

    def print_summary(self) -> None:
        now = time.monotonic()
        out = sys.stdout
        out.write("=== pressure_tookit run summary ===\n")
        total_done = total_frames = new_frames = 0
        for subject, info in self._subjects.items():
            if not self._visible(info):
                continue
            total_done += info["done"]
            total_frames += info["total"]
            delta = info["done"] - info["done_initial"]
            new_frames += delta
            end = info["end"] if info["end"] is not None else now
            duration = end - info["start"]
            rate = f"{duration / delta:.1f} s/frame" if delta > 0 else "-"
            status = "complete" if info["end"] is not None else "incomplete"
            out.write(f"{subject:<6} {info['done']:>6}/{info['total']:<6} frames "
                      f"in {self._format_duration(duration)} ({rate}) [{status}]\n")
        duration = now - self._run_start
        rate = f"{duration / new_frames:.1f} s/frame" if new_frames > 0 else "-"
        out.write(f"total   {total_done:>6}/{total_frames:<6} frames "
                  f"in {self._format_duration(duration)} ({rate})\n")
        out.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_canvas(value: str) -> tuple[int, int]:
    parts = [item for item in str(value).replace("x", ",").split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"canvas must be W,H: {value}")
    return int(parts[0]), int(parts[1])


def config_maxiters() -> int:
    """The config's own maxiters: what the child uses when --maxiters is absent."""
    try:
        import yaml
        data = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - the artifact must never break the run
        return CONFIG_MAXITERS_FALLBACK
    try:
        return int(data.get("maxiters", CONFIG_MAXITERS_FALLBACK))
    except (TypeError, ValueError):
        return CONFIG_MAXITERS_FALLBACK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--male", default="", help="Comma-separated male subjects, e.g. S5,S6,S7")
    parser.add_argument(
        "--female", "--famale", dest="female", default="",
        help="Comma-separated female subjects, e.g. S14; --famale is an alias",
    )
    parser.add_argument("--subject", default="",
                        help="Single-subject mode (queue conf): comma-separated subject ids, "
                             "gender taken from --gender")
    parser.add_argument("--gender", choices=("male", "female"), default="male",
                        help="Gender for --subject (one queue conf = one subject)")
    parser.add_argument("--dates", default="", help="Comma-separated dates; default discovers all dates")
    parser.add_argument("--dataset", default="", help="Alias of --dates for a single date (queue conf)")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all",
                        help="Discovery filter from AnysoleWorkspace/protocol/splits/default/"
                             "splits.csv.  The adapter tree holds fitting sessions, not "
                             "evaluation splits, so batch mode defaults to all; a queue conf "
                             "passes the value its generator used.")
    parser.add_argument("--run-id", default="fitting",
                        help="Run id under work://pressure_toolkit/<run_id>/ (batch mode)")
    parser.add_argument("--run-root", default="",
                        help="Explicit run root URI (queue conf: INIT_DATA_DIR). Must live under "
                             "work://pressure_toolkit/")
    parser.add_argument("--output-dir", default="",
                        help="Fitting output dir URI; must equal <run_root>/fitting (queue conf: OUTPUT_DIR)")
    parser.add_argument("--sessions", "--session", dest="sessions", default="",
                        help="Comma-separated session IDs; queue mode (one session per conf)")
    parser.add_argument("--stage", choices=("fit", "init_shape"), default="fit",
                        help="fit: init_shape + init_pose/tracking (default); init_shape: only "
                             "produce the per-subject shape file and exit")
    parser.add_argument("--start-idx", default="auto",
                        help="auto (default): resume from the first missing frame; an integer "
                             "forces that start index")
    parser.add_argument("--end-idx", type=int, default=-1,
                        help="Exclusive end frame; -1 (default) runs through the last declared frame")
    parser.add_argument("--gpu", default="", help="Physical GPU id exposed through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--gpu-list", default="",
                        help="Comma-separated physical GPUs; sessions run in parallel, one process per GPU")
    parser.add_argument("--per-gpu", type=int, default=4,
                        help="Fitting processes per GPU (default 4: the M4 concurrency ruling). "
                             "The depth term is CPU-bound (~2.5GB GPU / ~10 cores per process), "
                             "so several processes share one card fine.")
    parser.add_argument("--maxiters", type=int, default=None,
                        help="Override native fitting iterations; omit for the config default (101)")
    parser.add_argument("--canvas", type=parse_canvas, default=FORMAL_CANVAS,
                        help="Formal depth render canvas W,H (default 640,576; the native frames "
                             "are 1624x1240 and the intrinsics are rescaled to this canvas)")
    parser.add_argument("--depth-approx-stride", type=int, default=1,
                        help="Approximation mode only: >1 subsamples the observed "
                             "depth cloud; the formal baseline always uses 1 (full "
                             "upstream sampling)")
    parser.add_argument("--skip-mesh-export", action="store_true",
                        help="Skip per-frame SMPL OBJ export")
    parser.add_argument("--skip-gt-depth-export", action="store_true",
                        help="Skip per-frame observed-depth OBJ export")
    parser.add_argument("--no-export-obj", action="store_true",
                        help="Skip both visualization/debug OBJ exports")
    parser.add_argument("--reuse-session-resources", action="store_true",
                        help="Reuse renderer and fixed loss resources across frames")
    parser.add_argument("--icp-device", choices=("cpu", "cuda"), default=FORMAL_ICP_DEVICE,
                        help="Correspondence pipeline device. cpu (default, formal 口径) is the "
                             "original numpy/cKDTree path; cuda is an explicit acceleration switch "
                             "whose kNN returns squared distances gated as metric distances "
                             "(lib/fitSMPL/depthTerm.py:34-69 vs depthTerm.py:370), so its numbers "
                             "are NOT equivalent and never belong in a formal table.")
    parser.add_argument("--export", dest="export", action="store_true", default=False,
                        help="After fitting, export completed sessions through "
                             "export_baseline_motion.py into results/baselines/")
    parser.add_argument("--no-export", dest="export", action="store_false",
                        help="Explicitly disable the export step")
    parser.add_argument("--force", action="store_true", help="Rerun existing outputs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and the commands without running them")
    parser.add_argument("--keep-going", action="store_true",
                        help="Record a failed session and continue with later sessions")
    return parser


def _normalized(path: Path) -> Path:
    """Collapse ``..`` lexically so containment checks see the real target."""
    return Path(os.path.normpath(str(path)))


def resolve_run_root(args: argparse.Namespace) -> tuple[Path, Path, str]:
    """Validate the run root and the run-root invariant (OUTPUT_DIR=fitting).

    Paths are normalized lexically before the checks: ``Path("a/../b")`` keeps
    ``a`` in ``parents``, so an unnormalized ``work://pressure_toolkit/../escape``
    would slip past the ``work://pressure_toolkit`` containment test and only be
    collapsed by the kernel at open time.
    """
    work_root = resolve_uri(WORK_ROOT_URI)
    if args.run_root:
        run_root = _normalized(resolve_uri(args.run_root))
    else:
        run_id = str(args.run_id or "").strip().strip("/")
        if not run_id or run_id in {".", ".."} or "/" in run_id:
            raise ValueError(f"invalid run id: {args.run_id!r}")
        run_root = work_root / run_id
    if run_root == work_root or work_root not in run_root.parents:
        raise ValueError(f"run root must be {WORK_ROOT_URI}/<run_id>: {run_root}")
    if run_root.exists() and not run_root.is_dir():
        raise ValueError(f"run root is not a directory: {run_root}")
    expected_output = run_root / "fitting"
    output_dir = (_normalized(resolve_uri(args.output_dir))
                  if args.output_dir else expected_output)
    if output_dir != expected_output:
        raise ValueError(
            "run-root invariant broken: OUTPUT_DIR must be <run_root>/fitting "
            f"(got {output_dir}, expected {expected_output}); tracking reads the "
            "previous frame from <run_root>/fitting/results/...")
    return run_root, output_dir, canonical_uri(run_root)


def group_subjects(args: argparse.Namespace) -> dict[str, str]:
    male = split_subjects(args.male)
    female = split_subjects(args.female)
    overlap = male & female
    if overlap:
        raise ValueError(f"subjects listed in both --male and --female: {sorted(overlap)}")
    groups = {subject: "male" for subject in male}
    groups.update({subject: "female" for subject in female})
    queue_subjects = split_subjects(args.subject)
    if queue_subjects:
        if groups:
            raise ValueError("use either --male/--female or --subject/--gender, not both")
        groups = {subject: args.gender for subject in queue_subjects}
    if not groups:
        raise ValueError("provide at least one subject in --male/--female or --subject")
    return groups


def gpu_slots(args: argparse.Namespace) -> list[str]:
    """One slot per concurrent fitting process: physical GPUs x --per-gpu.

    Batch mode runs one process per slot (``<per-gpu>`` processes share one
    card: the fitting is CPU-bound, see ``--per-gpu``).  Queue mode leaves the
    GPU empty -- the worker's ``CUDA_VISIBLE_DEVICES`` is inherited and the
    queue itself provides the concurrency -- so the slot list is only used to
    derive the per-process thread cap there.
    """
    gpus = ([item.strip() for item in args.gpu_list.split(',') if item.strip()]
            if args.gpu_list else [args.gpu])
    return [gpu for gpu in gpus for _ in range(max(1, args.per_gpu))]


def native_depth_size(adapter: Path, ready: list[dict]) -> tuple[tuple[int, int], str]:
    """Native depth frame size from the adapter's own depth meta (actual value)."""
    for record in ready:
        meta = (adapter / "images" / record["date"] / record["subject"] /
                record["session"] / "depth" / "meta.json")
        if not meta.is_file():
            continue
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for key in sorted(payload, key=str):
            entry = payload[key]
            if isinstance(entry, dict) and entry.get("w") and entry.get("h"):
                return ((int(entry["w"]), int(entry["h"])),
                        f"depth/meta.json {record['date']}/{record['subject']}"
                        f"/{record['session']} {key}")
    return NATIVE_FRAME_SIZE, "declared default (no depth meta.json in this selection)"


def print_plan(ready: list[dict], skipped: list[dict], parameters: dict,
               run_meta: dict, run_root: Path, output_dir: Path,
               groups: dict[str, str], requested: set[str],
               frontend: dict[str, int]) -> None:
    out = sys.stdout
    out.write("=== pressure_toolkit plan ===\n")
    out.write(f"run root   : {parameters['run_root']}  -> {run_root}\n")
    out.write(f"output dir : {parameters['output_dir']}  (== run root/fitting)\n")
    out.write("口径       : icp_device={icp_device}  canvas={canvas}  "
              "maxiters={maxiters}  kdtree_workers={kdtree_workers}  "
              "depth_approx_stride={depth_approx_stride}  formal={formal}\n"
              .format(**parameters))
    out.write(f"native     : {run_meta['canvas_native']}  "
              f"({run_meta['canvas_native_source']})\n")
    out.write(f"subjects   : {sorted(groups)}  sessions="
              f"{','.join(sorted(requested)) if requested else 'all discovered'}\n")
    selected = {record["session"] for record in ready + skipped}
    linked = selected & set(frontend)
    out.write(f"depth frontend: {len(frontend)} sessions have DepthPro depth; "
              f"{len(linked)} of them are adapter-linked here "
              f"({len(frontend) - len(linked)} still waiting for build_inputs.py)\n")
    for record in ready + skipped:
        produced = frontend.get(record["session"], 0)
        if produced > record["materialized"]:
            out.write(f"  frontend  {record['session']}: frontend has {produced} "
                      f"depth frames, the adapter links "
                      f"{record['materialized']}; re-run build_inputs.py to unlock "
                      f"the rest of the session\n")
    for record in ready:
        out.write(f"  ready   {record['date']}/{record['subject']}/"
                  f"{record['session']}  frames "
                  f"[{record['frame_start']},{record['frame_stop']}) of "
                  f"{record['declared']} declared  ({record['reason']})\n")
    for record in skipped:
        out.write(f"  SKIP    {record['date']}/{record['subject']}/"
                  f"{record['session']}  {record['reason']} "
                  f"(declared {record['declared']})\n")
    out.flush()


def main() -> None:
    args = build_parser().parse_args()
    groups = group_subjects(args)
    dates = args.dates or args.dataset
    canvas = tuple(args.canvas)
    icp_device = effective_icp_device(args.icp_device)
    run_root, output_dir, run_uri = resolve_run_root(args)

    adapter = resolve_uri(ADAPTER_URI)
    requested_sessions = split_subjects(args.sessions)
    canonical_sessions = split_sessions(args.split)
    forced_start = (None if str(args.start_idx).strip().lower() == "auto"
                    else int(args.start_idx))

    candidates = discover_sessions(set(groups), dates, canonical_sessions)
    if requested_sessions:
        candidates = [item for item in candidates if item[2] in requested_sessions]
        missing = sorted(requested_sessions - {item[2] for item in candidates})
        if missing:
            raise SystemExit(
                "requested session(s) not discoverable in the adapter tree / "
                f"split={args.split}: {missing}")

    ready: list[dict] = []
    skipped: list[dict] = []
    for date, subject, session in candidates:
        frame_ids = fitting_frame_ids(adapter, date, subject, session)
        record = plan_session(adapter, date, subject, session, frame_ids,
                              forced_start if forced_start is not None else 0,
                              args.end_idx)
        record["gender"] = groups[subject]
        (ready if record["status"] == "ready" else skipped).append(record)

    slots = gpu_slots(args)
    # --per-gpu is already folded into the slot list; the process count is the
    # slot count.
    parallelism = len(slots)
    kdtree_workers = resolve_kdtree_workers(args, parallelism)
    maxiters = args.maxiters if args.maxiters is not None else config_maxiters()
    native_size, native_source = native_depth_size(adapter, ready)
    formal = (icp_device == FORMAL_ICP_DEVICE
              and canvas == FORMAL_CANVAS
              and args.depth_approx_stride == 1
              and maxiters == FORMAL_MAXITERS)
    parameters = {
        # M1/M2/M3 口径 (2026-09-28): the values actually used by this run.
        "icp_device": icp_device,
        "icp_device_note": (
            "formal = cpu. GPU kNN returns squared distances gated as metric "
            "(lib/fitSMPL/depthTerm.py:34-69 vs depthTerm.py:370 "
            "icp_dist_thresh=0.05): defect recorded, not patched, GPU is never "
            "formal"),
        "canvas": [canvas[0], canvas[1]],
        "color_size": [NATIVE_FRAME_SIZE[0], NATIVE_FRAME_SIZE[1]],
        "dintr_scale_rule": ("fx,cx *= canvas_w/native_w; fy,cy *= canvas_h/"
                             "native_h (lib/core/fit_single_frame.py:150-154)"),
        "maxiters": maxiters,
        "kdtree_workers": kdtree_workers,
        "depth_approx_stride": args.depth_approx_stride,
        "run_root": run_uri,
        "output_dir": canonical_uri(output_dir),
        "basdir": BASDIR,
        "essential_root": ESSENTIAL_ROOT,
        "formal": formal,
    }
    scale = {"fx": canvas[0] / native_size[0], "fy": canvas[1] / native_size[1],
             "cx": canvas[0] / native_size[0], "cy": canvas[1] / native_size[1]}
    run_meta = {
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "canvas_native": [native_size[0], native_size[1]],
        "canvas_native_source": native_source,
        "dintr_scale": scale,
        "maxiters_source": ("cli --maxiters" if args.maxiters is not None
                            else f"config {CONFIG.name}"),
        "python": sys.executable,
        "toolkit_root": str(TOOLKIT_ROOT),
    }
    request = {
        "mode": "queue" if requested_sessions else "batch",
        "subjects": groups,
        "dates": dates or "all",
        "split": args.split,
        "sessions": sorted(requested_sessions) if requested_sessions else [],
        "stage": args.stage,
        "start_idx": str(args.start_idx),
        "end_idx": args.end_idx,
        "per_gpu": args.per_gpu,
        "export": bool(args.export),
    }
    print_plan(ready, skipped, parameters, run_meta, run_root, output_dir,
               groups, requested_sessions, depth_frontend_sessions())
    if not formal:
        print("WARNING: this run is NOT the formal 口径 (M1/M2/M3); its numbers "
              "must not enter a formal table", file=sys.stderr)

    if args.dry_run:
        print("--- planned commands (dry run, nothing written) ---")
        for record in ready:
            existing = (record["frames"][0] if args.force else first_missing_frame(
                output_dir, record["date"], record["subject"], record["session"],
                record["frames"]))
            if existing is None:
                print(f"# {record['session']}: complete "
                      f"({len(record['frames'])} frames already fitted)")
                continue
            if args.stage == "init_shape":
                print("$ " + " ".join(command(
                    args, run_root, output_dir, record["date"], record["subject"],
                    record["session"], record["gender"], "init_shape",
                    record["frames"][0], record["frames"][0] + 1)))
                continue
            stage = "init_pose" if existing == record["frames"][0] else "tracking"
            if not shape_path(run_root, record["date"], record["subject"]).is_file():
                print(f"# {record['session']}: init_shape first "
                      f"(frame {record['frames'][0]})")
                print("$ " + " ".join(command(
                    args, run_root, output_dir, record["date"], record["subject"],
                    record["session"], record["gender"], "init_shape",
                    record["frames"][0], record["frames"][0] + 1)))
            print(f"# {record['session']}: {stage} from frame {existing}")
            print("$ " + " ".join(command(
                args, run_root, output_dir, record["date"], record["subject"],
                record["session"], record["gender"], stage, existing,
                record["frames"][-1] + 1)))
        if args.export:
            for record in ready:
                if not record["partial"]:
                    export_session(record["session"], run_root, dry_run=True)
        return

    run_root.mkdir(parents=True, exist_ok=True)
    ensure_initialization_link(run_root)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    records = {record["session"]: record for record in ready + skipped}
    write_run_artifact(run_root, parameters, records, run_meta, [request])
    if not ready:
        print("nothing to fit (all selected sessions were skipped or absent)")

    try:
        if args.stage == "init_shape":
            run_init_shape_only(args, run_root, output_dir, ready, slots,
                                kdtree_workers, icp_device, records, parameters,
                                run_meta, request)
            return
        with ExitStack() as stack:
            # Batch: one run-root lock, non-blocking, so two full runs cannot
            # share a tree.  Queue: one lock per session, blocking, so a
            # duplicate conf waits and then finds the work already done.
            if requested_sessions:
                for record in ready:
                    stack.enter_context(file_lock(
                        lock_path(run_root,
                                  f"session_{record['date']}_"
                                  f"{record['subject']}_{record['session']}")))
            else:
                stack.enter_context(file_lock(
                    lock_path(run_root, "run"), blocking=False,
                    message="another run_full_mmvp.py instance is already "
                            f"writing to {run_root}"))
            run_sessions(args, run_root, output_dir, ready, groups, slots,
                         kdtree_workers, icp_device, forced_start, parameters,
                         run_meta, request, records)
    finally:
        # Queue confs share one run root: merge the final per-session status
        # (done/failed, stage, resume frame) written by this invocation.
        write_run_artifact(run_root, parameters, records, run_meta, [request])


def run_sessions(args: argparse.Namespace, run_root: Path, output_dir: Path,
                 ready: list[dict], groups: dict[str, str], gpu_slots: list[str],
                 kdtree_workers: int, icp_device: str, forced_start: int | None,
                 parameters: dict, run_meta: dict, request: dict,
                 records: dict) -> None:
    run_start = time.monotonic()
    since_ts = time.time()
    monitor_subjects: dict = {}
    failures: list[tuple[str, str, str, str]] = []
    pending: list[dict] = []

    for record in ready:
        date, subject, session = record["date"], record["subject"], record["session"]
        frames = record["frames"]
        info = monitor_subjects.setdefault(subject, {
            "total": 0, "done": 0, "done_initial": 0,
            "start": time.monotonic(), "end": None, "sessions": {},
        })
        info["total"] += len(frames)
        existing = None if args.force else first_missing_frame(
            output_dir, date, subject, session, frames)
        if existing is None:
            print(f"[{date}/{subject}/{session}] complete: {len(frames)} frames "
                  "already fitted")
            record["status"] = "done"
            record["stage"] = "already-complete"
            record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            info["done_initial"] += len(frames)
            continue

        if not ensure_init_shape(args, run_root, output_dir, record, gpu_slots,
                                 kdtree_workers, icp_device, failures):
            record["status"] = "failed"
            record["reason"] = "init_shape-failed"
            if not args.keep_going:
                return finalize(args, run_root, output_dir, failures,
                                parameters, run_meta, request, records)
            continue

        temp_marker = (output_dir / "temp" / date / subject / session /
                       f"init_pose_{subject}.npz")
        if args.force or existing == frames[0]:
            # Nothing fitted yet for this session: init_pose on its first
            # materialized frame, then automatic tracking.
            stage, start_idx = "init_pose", frames[0]
        elif temp_marker.is_file():
            # init_pose for this session already ran (its first frame wrote the
            # temp marker) -> resume the tracking chain at the gap.
            stage, start_idx = "tracking", existing
        else:
            # A gap without a usable previous pose: restart from the first frame
            # instead of tracking from a pose this run never produced.
            stage, start_idx = "init_pose", frames[0]
        print(f"[{date}/{subject}/{session}] "
              + (f"{stage} from frame {start_idx} to {frames[-1]} "
                 f"({groups[subject]})" if stage == "tracking" else
                 f"init_pose (then tracking) from frame {start_idx} to "
                 f"{frames[-1]} ({groups[subject]})"))
        skip = frames.index(start_idx)
        info["done_initial"] += skip
        info["sessions"][(date, session)] = {"base": skip, "live": True}
        record["stage"] = stage
        record["resume_from"] = start_idx
        pending.append({
            "record": record, "date": date, "subject": subject, "session": session,
            "stage": stage,
            "command": command(args, run_root, output_dir, date, subject, session,
                               groups[subject], stage, start_idx, frames[-1] + 1),
        })

    def run_pending(item, gpu):
        try:
            run(item["command"], gpu, kdtree_workers, icp_device,
                run_root / "logs" / item["date"] / item["subject"] /
                f"{item['session']}_{item['stage']}.log")
            item["record"]["status"] = "done"
            item["record"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            return None
        except subprocess.CalledProcessError:
            item["record"]["status"] = "failed"
            item["record"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            return (item["date"], item["subject"], item["session"], item["stage"])

    monitor = None
    if pending:
        monitor = ProgressMonitor(output_dir, monitor_subjects, since=since_ts,
                                  started=run_start)
        monitor.start()
    try:
        if len(gpu_slots) <= 1:
            for item in pending:
                failure = run_pending(item, gpu_slots[0] if gpu_slots else "")
                if failure is not None:
                    failures.append(failure)
                    if not args.keep_going:
                        return finalize(args, run_root, output_dir, failures,
                                        parameters, run_meta, request, records)
        else:
            with ThreadPoolExecutor(max_workers=len(gpu_slots)) as executor:
                futures = [executor.submit(run_pending, item,
                                           gpu_slots[index % len(gpu_slots)])
                           for index, item in enumerate(pending)]
                for future in as_completed(futures):
                    failure = future.result()
                    if failure is not None:
                        failures.append(failure)
                        if not args.keep_going:
                            for other in futures:
                                other.cancel()
                            return finalize(args, run_root, output_dir, failures,
                                            parameters, run_meta, request,
                                            records)
    finally:
        if monitor is not None:
            monitor.stop()
    if monitor is not None:
        monitor.print_summary()
    finalize(args, run_root, output_dir, failures, parameters, run_meta,
             request, records)


def finalize(args: argparse.Namespace, run_root: Path, output_dir: Path,
             failures: list[tuple[str, str, str, str]], parameters: dict,
             run_meta: dict, request: dict, records: dict) -> None:
    """Export the sessions that are complete; report the failures.

    Every session of this invocation's plan is considered, not only the ones
    that fitted frames this time: a session that was already complete (or was
    fitted by an earlier conf with ``PRESSURE_EXPORT=0``) still belongs in
    ``results/baselines/``.  The exporter skips an existing archive unless
    ``--force`` is given, so a re-export is a no-op.
    """
    if args.export:
        for record in records.values():
            if not record["frames"]:
                continue
            if record["partial"]:
                print(f"[export] skip {record['session']}: partial session "
                      f"({record['materialized']} of {record['declared']} frames "
                      "materialized); the exporter zero-fills missing frames",
                      flush=True)
                continue
            if first_missing_frame(output_dir, record["date"], record["subject"],
                                   record["session"], record["frames"]) is not None:
                print(f"[export] skip {record['session']}: frames still missing",
                      flush=True)
                continue
            try:
                export_session(record["session"], run_root)
            except subprocess.CalledProcessError:
                failures.append(("", "", record["session"], "export"))
    write_run_artifact(run_root, parameters, records, run_meta, [request])
    if failures:
        failure_path = run_root / "batch_failures.json"
        failure_path.write_text(json.dumps(
            [{"date": d, "subject": s, "session": sid, "stage": stage}
             for d, s, sid, stage in failures],
            ensure_ascii=False, indent=2,
        ) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(failures)} pressure_toolkit sessions failed; "
                         f"see {failure_path}")


def ensure_init_shape(args: argparse.Namespace, run_root: Path, output_dir: Path,
                      record: dict, gpu_slots: list[str], kdtree_workers: int,
                      icp_device: str, failures: list) -> bool:
    """Guarantee the per-subject init_shape file before an init_pose/tracking run.

    init_shape is per (date, subject) while sessions are many: the first session
    of a subject computes it under a per-subject flock and every later session
    sees the published file and skips both the computation and the lock.  That
    removes the old three-conf topology (init_shape -> init_pose -> tracking)
    without serializing the whole run.
    """
    date, subject, session = record["date"], record["subject"], record["session"]
    shape = shape_path(run_root, date, subject)
    if shape.is_file() and not args.force:
        return True
    print(f"[{date}/{subject}] init_shape on frame {record['frames'][0]}")
    with file_lock(lock_path(run_root, f"shape_{date}_{subject}"), blocking=True):
        if shape.is_file() and not args.force:
            return True
        cmd = command(args, run_root, output_dir, date, subject, session,
                      record["gender"], "init_shape", record["frames"][0],
                      record["frames"][0] + 1)
        try:
            run(cmd, gpu_slots[0] if gpu_slots else "", kdtree_workers, icp_device,
                run_root / "logs" / date / subject / f"{subject}_init_shape.log")
        except subprocess.CalledProcessError:
            failures.append((date, subject, session, "init_shape"))
            return False
    return True


def run_init_shape_only(args: argparse.Namespace, run_root: Path, output_dir: Path,
                        ready: list[dict], gpu_slots: list[str],
                        kdtree_workers: int, icp_device: str, records: dict,
                        parameters: dict, run_meta: dict, request: dict) -> None:
    """--stage init_shape: produce the per-subject shape files and stop.

    Available for a scheduler pre-run; the generated queue confs do not need it
    because the fit stage ensures init_shape on its own.
    """
    seen: set[tuple[str, str]] = set()
    failures: list = []
    for record in ready:
        key = (record["date"], record["subject"])
        if key in seen:
            continue
        seen.add(key)
        if ensure_init_shape(args, run_root, output_dir, record, gpu_slots,
                             kdtree_workers, icp_device, failures):
            record["stage"] = "init_shape"
            record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        else:
            record["status"] = "failed"
            record["reason"] = "init_shape-failed"
    write_run_artifact(run_root, parameters, records, run_meta, [request])
    if failures:
        raise SystemExit(f"{len(failures)} init_shape runs failed")


if __name__ == "__main__":
    try:
        main()
    except (WorkspacePathError, ValueError) as error:
        raise SystemExit(str(error))
