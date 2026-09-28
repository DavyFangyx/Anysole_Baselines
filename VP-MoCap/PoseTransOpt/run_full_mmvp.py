#!/usr/bin/env python3
"""Run the native PoseTransOpt entrypoint session-by-session.

Inputs are the adapter-joined PoseTransOpt packages:

    model-input://PoseTransOpt/adapter_v1/<date>/<subject>/<session>/
    ├── color/                  final joined frames only (for visualization)
    ├── keypoints/              RTMPose HALPE-26 sidecars with frame ids
    ├── CLIFF_results.npz       single-person mask-bbox CLIFF, one row per frame
    ├── pred_contact_smpl/      FPP-Net sidecars with frame ids
    ├── template_scene_rgbd.npy
    └── join_manifest.json      frame-id join report (authority for alignment)

Outputs land under work://VP-MoCap/v1/pose_optimization/<date>/<subject>/<session>.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "AnysoleWorkspace"
ENTRY_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = WORKSPACE / "model_inputs" / "PoseTransOpt" / "adapter_v1"
OUTPUT_ROOT = WORKSPACE / "work" / "VP-MoCap" / "v1" / "pose_optimization"


def selected_sessions(split: str) -> list[str]:
    columns = ("train", "val", "test") if split == "all" else (split,)
    values = []
    with (WORKSPACE / "protocol/splits/default/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for column in columns:
                value = (row.get(column) or "").strip()
                if value:
                    values.append(value)
    return sorted(set(values))


def manifest() -> dict[str, dict]:
    result = {}
    with (WORKSPACE / "protocol/manifests/session_manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[row["session_id"]] = row
    return result


def session_parts(row: dict) -> tuple[str, str]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from AnysoleWorkspace.tool.workspace import resolve_uri
    parts = resolve_uri(row["video_path"], must_exist=True).parts
    return parts[-3], parts[-2]


def command(args: argparse.Namespace, date: str, subject: str, session: str) -> list[str]:
    base = INPUT_ROOT / date / subject / session
    cmd = [
        sys.executable, "-m", "app.optimize",
        f"task.input_path_base={base}",
        f"task.scene_rgbd={base / 'template_scene_rgbd.npy'}",
        f"task.output_path={OUTPUT_ROOT / date / subject / session}",
        f"gpu={args.gpu}",
    ]
    if args.max_iter is not None:
        cmd.append(f"method.max_iter={args.max_iter}")
    if args.max_frames is not None:
        cmd.append(f"task.max_frames={args.max_frames}")
    if args.no_visualization:
        cmd.append("task.write_visualization=false")
    if args.hydra_root:
        cmd.append(f"hydra.run.dir={args.hydra_root}/{session}")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--sessions", default="", help="comma-separated session IDs")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Native CUDA device index used by PoseTransOpt; no CUDA_VISIBLE_DEVICES remap")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Smoke override; omit for the native config default")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Smoke override; must be >=21 for the native Savitzky-Golay filter")
    parser.add_argument("--hydra-root", default="",
                        help="Optional Hydra run root for smoke jobs")
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_rows = manifest()
    sessions = [item.strip() for item in args.sessions.split(",") if item.strip()] if args.sessions else selected_sessions(args.split)
    for session in sessions:
        if session not in manifest_rows:
            raise KeyError(f"session not in manifest: {session}")
        date, subject = session_parts(manifest_rows[session])
        base = INPUT_ROOT / date / subject / session
        join_path = base / "join_manifest.json"
        if not join_path.is_file():
            print(f"skip {session}: no join manifest (run the PoseTransOpt adapter first)")
            continue
        join = json.loads(join_path.read_text(encoding="utf-8"))
        if not join.get("final_frames"):
            print(f"skip {session}: empty join (see join_manifest.json reasons)")
            continue
        required = [base / "color", base / "keypoints",
                    base / "pred_contact_smpl", base / "CLIFF_results.npz",
                    base / "template_scene_rgbd.npy"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{session}: missing PoseTransOpt inputs: {missing}")
        native_output = OUTPUT_ROOT / date / subject / session / "opt_result.pth"
        if native_output.is_file() and not args.force:
            print(f"skip existing {native_output}")
            continue
        cmd = command(args, date, subject, session)
        print("$ " + " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=ENTRY_ROOT, env=os.environ.copy(), check=True)


if __name__ == "__main__":
    main()
