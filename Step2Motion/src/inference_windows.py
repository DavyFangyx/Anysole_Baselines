"""Inference windows never move backwards at the tail; padding is context only."""
import numpy as np
import torch


def inference_starts(n_frames, window):
    if n_frames <= 0 or window < 2 or window % 2:
        raise ValueError('Need a nonempty clip and an even window >= 2')
    return [0] + list(range(window // 2, n_frames - window // 2, window // 2))


def inference_window(dataset, start, window):
    if len(dataset.poses) != 1:
        raise ValueError('Isolate one clip before inference')
    pose, condition = dataset.poses[0], dataset.insole[0]
    n = len(pose)
    end = min(start + window, n)
    if not 0 <= start < end or len(condition) != n:
        raise ValueError('Invalid inference window or condition length')
    frame_ids = np.asarray(getattr(dataset, 'clip_frame_ids', [np.arange(n)])[0], dtype=np.int64)
    valid = np.asarray(getattr(dataset, 'clip_valid', [np.ones(n, bool)])[0], dtype=bool)
    if len(frame_ids) != n or len(valid) != n or np.any(np.diff(frame_ids) != 1):
        raise ValueError('Clip frame ids must be contiguous; split gaps into separate clips')
    def padded(values):
        chunk = values[start:end]
        if len(chunk) < window:
            chunk = torch.cat([chunk, chunk[-1:].expand(window-len(chunk), *chunk.shape[1:])], dim=0)
        return chunk
    ids = np.full(window, -1, dtype=np.int64)
    mask = np.zeros(window, dtype=bool)
    ids[:end-start] = frame_ids[start:end]
    mask[:end-start] = valid[start:end]
    return padded(pose), padded(condition), dataset.distances[0], ids, mask, end-start
