"""BVH pose conversion (canonical home: ``Baselines/utils``).

This mirrors MocapVideoAligner_0811/utils/bvh_parser.py and bvh_pose.py.
``results_display/script/utils/bvh_aligner_pose.py`` is a re-export shim.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _rotation_matrix(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def parse_bvh_aligner(path: str | Path, trim_leading_seconds: float = 0.40) -> dict:
    """Parse and convert a BVH using MocapVideoAligner_0811's pose logic."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    motion_pos = text.find("MOTION")
    if motion_pos < 0:
        raise ValueError(f"BVH 文件缺少 MOTION 段：{path}")
    hierarchy = text[:motion_pos]

    names = []
    parents = []
    offsets = []
    channels = []
    channel_indices = []
    stack = []
    pending_name = None
    cursor = 0
    end_site_count = 0
    for line in hierarchy.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("ROOT ") or s.startswith("JOINT "):
            pending_name = s.split(None, 1)[1]
        elif s.startswith("End Site"):
            parent_name = names[stack[-1]] if stack else "End"
            pending_name = f"{parent_name}_EndSite{end_site_count}"
            end_site_count += 1
        elif s == "{":
            if pending_name is None:
                continue
            names.append(pending_name)
            parents.append(stack[-1] if stack else -1)
            offsets.append(np.zeros(3, dtype=np.float64))
            channels.append([])
            channel_indices.append([])
            stack.append(len(names) - 1)
            pending_name = None
        elif s == "}":
            if stack:
                stack.pop()
        elif s.startswith("OFFSET") and stack:
            offsets[stack[-1]] = np.asarray(s.split()[1:4], dtype=np.float64)
        elif s.startswith("CHANNELS") and stack:
            parts = s.split()
            count = int(parts[1])
            channels[stack[-1]] = parts[2 : 2 + count]
            channel_indices[stack[-1]] = list(range(cursor, cursor + count))
            cursor += count

    ft_match = re.search(r"Frame Time:\s*([\d.eE+\-]+)", text)
    if ft_match is None:
        raise ValueError(f"BVH 文件缺少 Frame Time：{path}")
    frame_time = float(ft_match.group(1))
    motion_start = text.find("Frame Time:")
    raw_lines = [line.strip() for line in text[motion_start:].splitlines()[1:] if line.strip()]
    raw_frames = np.asarray([[float(v) for v in line.split()] for line in raw_lines], dtype=np.float64)
    if raw_frames.ndim != 2 or raw_frames.shape[1] != cursor:
        raise ValueError(f"BVH 通道数不匹配：expected {cursor}, got {raw_frames.shape}")

    trim_frames = int(round(trim_leading_seconds / frame_time)) if frame_time > 0 else 0
    trim_frames = max(0, min(trim_frames, max(len(raw_frames) - 2, 0)))
    raw_frames = raw_frames[trim_frames:]

    all_positions = np.zeros((len(raw_frames), len(names), 3), dtype=np.float64)
    all_rotations = np.zeros((len(raw_frames), len(names), 3, 3), dtype=np.float64)
    all_local_rotations = np.zeros_like(all_rotations)
    display_rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    for frame_index, frame in enumerate(raw_frames):
        positions = np.zeros((len(names), 3), dtype=np.float64)
        global_rotations = []
        for joint_index in range(len(names)):
            translation = offsets[joint_index].copy()
            local_rotation = np.eye(3, dtype=np.float64)
            for channel_name, value in zip(channels[joint_index], frame[channel_indices[joint_index]]):
                axis = channel_name.lower()[0]
                if channel_name.lower().endswith("position"):
                    translation["xyz".index(axis)] += value
                elif channel_name.lower().endswith("rotation"):
                    local_rotation = local_rotation @ _rotation_matrix(axis, np.deg2rad(value))
            if parents[joint_index] < 0:
                global_rot = local_rotation
                global_pos = translation
            else:
                parent = parents[joint_index]
                global_rot = global_rotations[parent] @ local_rotation
                global_pos = positions[parent] + global_rotations[parent] @ translation
            positions[joint_index] = global_pos
            global_rotations.append(global_rot)
            all_rotations[frame_index, joint_index] = display_rotation @ global_rot @ display_rotation.T
            all_local_rotations[frame_index, joint_index] = display_rotation @ local_rotation @ display_rotation.T
        # Exact MocapVideoAligner default z-up display conversion.
        all_positions[frame_index] = np.column_stack(
            (positions[:, 0], -positions[:, 2], positions[:, 1])
        )

    return {
        "joints": all_positions.astype(np.float32),
        "rotations": all_rotations.astype(np.float32),
        "local_rotations": all_local_rotations.astype(np.float32),
        "parents": np.asarray(parents, dtype=np.int64),
        "names": names,
        "fps": 1.0 / frame_time if frame_time > 0 else 40.0,
        "frame_time": frame_time,
        "trim_frames": trim_frames,
    }
