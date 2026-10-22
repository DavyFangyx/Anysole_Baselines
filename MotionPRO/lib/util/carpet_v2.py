"""Fixed world-coordinate carpets, fitted on train only; no target recentering."""
from __future__ import annotations
import hashlib
import json
import numpy as np


def geometry_hash(geometry):
    payload = {k: v for k, v in geometry.items() if k != 'sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def footprint_bounds(tracks):
    lows, highs = [], []
    for side in ('left', 'right'):
        center = np.asarray(tracks[side]['center'], dtype=np.float64)
        direction = np.asarray(tracks[side]['direction'], dtype=np.float64)
        if not np.isfinite(center).all() or not np.isfinite(direction).all():
            raise ValueError('Nonfinite footprint geometry')
        perp = np.stack((-direction[:, 1], direction[:, 0]), axis=-1)
        reach = .12 * np.abs(direction) + .06 * np.abs(perp)
        lows.append((center - reach).min(axis=0))
        highs.append((center + reach).max(axis=0))
    return np.min(lows, axis=0), np.max(highs, axis=0)


def fit_geometry(train_bounds, train_ids, cam_id=3):
    """One global rectangle; evaluation records never enter this fit."""
    if len(train_bounds) != len(train_ids) or not train_ids:
        raise ValueError('Train bounds and identities must match')
    cell = .0125
    quantum = 5 * cell
    lower = np.floor((np.min([b[0] for b in train_bounds], axis=0) - .125) / quantum) * quantum
    upper = np.ceil((np.max([b[1] for b in train_bounds], axis=0) + .125) / quantum) * quantum
    width, height = np.rint((upper - lower) / cell).astype(int)
    geometry = dict(schema='motionpro.fixed_world_carpet.v2', cam_id=int(cam_id),
        fit_split='train', fit_sessions=sorted(train_ids), safety_margin_m=.125,
        cell_m=cell, origin_xy_m=[float(lower[0]), float(upper[1])],
        bounds_xy_m=[lower.tolist(), upper.tolist()], shape_hw=[int(height), int(width)],
        entry_hw=[int(height * 3 // 5), int(width * 4 // 5)], resize_mode='area',
        pixel_center_world='x=origin_x+(col+0.5)*cell; y=origin_y-(row+0.5)*cell',
        targets='unchanged original world coordinates', per_session_recentering=False)
    geometry['sha256'] = geometry_hash(geometry)
    return geometry


def world_to_pixel(xy, geometry):
    xy = np.asarray(xy, dtype=np.float64)
    origin = np.asarray(geometry['origin_xy_m'])
    return (xy - origin) * np.array([1., -1.]) / geometry['cell_m'] - .5


def pixel_to_world(col_row, geometry):
    return np.asarray(geometry['origin_xy_m']) + (np.asarray(col_row) + .5) * geometry['cell_m'] * np.array([1., -1.])


def render_carpet(tracks, left48, right48, geometry, painter):
    if geometry_hash(geometry) != geometry['sha256']:
        raise ValueError('Invalid carpet geometry hash')
    image = np.zeros((len(left48), *geometry['shape_hw']), dtype=np.float32)
    origin = np.asarray(geometry['origin_xy_m'])
    for side, cells in [('left', left48), ('right', right48)]:
        values = np.clip(np.asarray(cells, dtype=np.float64), 0., 1023.) / 1023. * 255.
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite source pressure')
        for i in range(len(image)):
            # The existing painter uses x=0 at the left and y=0 at the top.
            # Only render coordinates change; labels and world targets do not.
            painter(image[i], tracks[side]['center'][i] - origin,
                    tracks[side]['direction'][i], values[i])
    return image
