"""Build an opt-in, train-fitted fixed-world MotionPRO carpet version.

Run from gait: python Baselines/MotionPRO/data_prep/build_carpet_v2.py
Old adapter_v1 inputs, labels, features and checkpoints are never overwritten.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / 'Baselines/MotionPRO'
for path in (ROOT, MODEL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from AnysoleWorkspace.tool.adapters.MotionPRO import adapter as A
from lib.util.carpet_v2 import fit_geometry, footprint_bounds, render_carpet, world_to_pixel, pixel_to_world

VERSION = 'adapter_v2_carpet'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_tracks(sid, cam):
    session = A.load_shared_session(sid, cam)
    tracks = A.foot_tracks(A._context(session))
    return session, tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cam', type=int, default=3)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    split_path = A.WORKSPACE_ROOT / 'protocol/splits/default/splits.csv'
    split_digest = sha(split_path)
    records = list(csv.DictReader(split_path.open(encoding='utf-8-sig')))
    groups = {key: sorted({r[key].strip() for r in records if r.get(key, '').strip()}) for key in ('train', 'val', 'test')}
    sessions = sorted(set.union(*(set(v) for v in groups.values())))
    output = args.output_root or A.WORKSPACE_ROOT / 'model_inputs/MotionPRO' / VERSION
    output.mkdir(parents=True, exist_ok=True)
    geometry_path = output / f'geometry_cam{args.cam}.json'
    config_path = output / f'imagepressure2smpl_carpet_v2_cam{args.cam}.yaml'
    if geometry_path.exists() or (output / f'cam{args.cam}').exists():
        raise FileExistsError('Use a fresh version directory; existing inputs are never overwritten')
    originals = {}
    cache = {}
    bounds = []
    # This is the only geometry fit: validation/test identities are excluded.
    for i, sid in enumerate(groups['train']):
        session, tracks = load_tracks(sid, args.cam)
        bounds.append(footprint_bounds(tracks))
        cache[sid] = (session, tracks)
        if (i + 1) % 20 == 0:
            print('fit train footprints', i + 1, len(groups['train']), flush=True)
    geometry = fit_geometry(bounds, groups['train'], args.cam)
    geometry_path.write_text(json.dumps(geometry, indent=2) + '\n')
    origin = np.asarray(geometry['origin_xy_m'])
    lower, upper = np.asarray(geometry['bounds_xy_m'])
    totals = dict(sessions=0, frames=0, old_blank=0, old_blank_after_resize=0,
                  new_blank=0, new_blank_after_resize=0, source_nonzero_valid_lost=0,
                  full_empty_sessions_before=0, full_empty_sessions_after=0,
                  footprint_frames_outside=0)
    details = []
    # Use the actual production consumer's resizing implementation.
    from lib.dataset.image_pressure import ImagePressureDataset
    reader = ImagePressureDataset.__new__(ImagePressureDataset)
    reader.pressure_hw = tuple(geometry['entry_hw'])
    reader.pressure_resize_mode = 'area'
    for i, sid in enumerate(sessions):
        session, tracks = cache.pop(sid) if sid in cache else load_tracks(sid, args.cam)
        old = A.model_input_dir(sid, args.cam)
        target = output / f'cam{args.cam}' / old.parent.parent.name / old.parent.name / sid
        target.mkdir(parents=True)
        old_pressure_path = old / 'pressure.npz'
        originals[str(old_pressure_path)] = sha(old_pressure_path)
        linked = {}
        for name in ('feature_hrnet.pth', 'keypoints.npy', 'smpl.npy', 'contact.npy', 'frame_id.npy'):
            source = old / name
            if not source.is_file():
                raise FileNotFoundError(source)
            linked[name] = dict(path=str(source), sha256=sha(source))
            target.joinpath(name).symlink_to(source)
        if (old / 'color').exists():
            target.joinpath('color').symlink_to(old / 'color', target_is_directory=True)
        left, right = session['pressure']['left48'], session['pressure']['right48']
        carpet = render_carpet(tracks, left, right, geometry, A._paint_footprint)
        small = reader._resize_pressure(carpet / 255.).numpy()
        with np.load(old_pressure_path) as z:
            before = z['pressure']
        old_blank = ~np.any(before != 0., axis=(1, 2))
        old_small = torch.nn.functional.interpolate(torch.from_numpy(before)[:, None], size=(96, 96), mode='bilinear', align_corners=False)[:, 0].numpy()
        blank = ~np.any(carpet != 0., axis=(1, 2))
        small_blank = ~np.any(small != 0., axis=(1, 2))
        before_small_blank = ~np.any(old_small != 0., axis=(1, 2))
        valid = np.asarray(session['frames']['valid'], bool) & ~np.asarray(session['frames']['fake'], bool)
        source_nonzero = np.any(np.asarray(left) > 0., axis=1) | np.any(np.asarray(right) > 0., axis=1)
        lost = source_nonzero & small_blank
        if np.any(source_nonzero & small_blank):
            raise AssertionError(f'{sid}: source pressure lost at {np.flatnonzero(source_nonzero & small_blank).tolist()}')
        np.testing.assert_array_equal(blank, small_blank)
        outside = np.zeros(len(carpet), bool)
        for side in ('left', 'right'):
            d = tracks[side]['direction']; p = np.stack((-d[:, 1], d[:, 0]), axis=-1)
            reach = .12 * abs(d) + .06 * abs(p)
            outside |= np.any(tracks[side]['center'] - reach < lower - 1e-8, axis=1)
            outside |= np.any(tracks[side]['center'] + reach > upper + 1e-8, axis=1)
            np.testing.assert_allclose(pixel_to_world(world_to_pixel(tracks[side]['center'], geometry), geometry), tracks[side]['center'], atol=1e-12)
        if np.any(outside):
            raise AssertionError(f'{sid}: frozen world grid does not cover all footprint frames')
        if outside.any():
            raise AssertionError(f'{sid}: footprints outside fixed geometry')
        np.savez_compressed(target / 'pressure.npz', pressure=carpet, geometry_sha256=np.asarray(geometry['sha256']), world_origin_xy_m=origin, cell_m=np.asarray(geometry['cell_m']))
        artifact = dict(schema_version='model_input.motionpro.fixed_world_carpet.v2',
            producer=str(Path(__file__).relative_to(ROOT)), adapter_version=VERSION,
            session_id=sid, cam_id=args.cam, geometry_sha256=geometry['sha256'],
            geometry_file=str(geometry_path), source_pressure=str(session['dir'] / 'pressure_48.npz'),
            source_pressure_sha256=sha(session['dir'] / 'pressure_48.npz'),
            source_frames_sha256=sha(session['dir'] / 'frames.npz'),
            source_bvh_sha256=sha(A.resolve_uri(session['meta']['source_files']['bvh'], must_exist=True)),
            old_pressure_sha256=originals[str(old_pressure_path)], preserved_inputs=linked,
            gt_and_world_targets='unchanged; read-only links to adapter_v1',
            stats=dict(frames=len(carpet), empty_frames=int(blank.sum()), source_nonzero_valid_lost=int(lost.sum())))
        (target / 'artifact.json').write_text(json.dumps(artifact, indent=2) + '\n')
        assert sha(old_pressure_path) == originals[str(old_pressure_path)]
        for entry in linked.values():
            assert sha(entry['path']) == entry['sha256']
        row = dict(session=sid, split='train' if sid in groups['train'] else 'val' if sid in groups['val'] else 'test',
            frames=len(carpet), old_blank=int(old_blank.sum()), old_blank_after_resize=int(before_small_blank.sum()),
            new_blank=int(blank.sum()), new_blank_after_resize=int(small_blank.sum()),
            source_nonzero_valid_lost=int(lost.sum()), footprint_frames_outside=int(outside.sum()),
            full_empty_sessions_before=int(old_blank.all()), full_empty_sessions_after=int(blank.all()))
        details.append(row)
        for key in totals:
            totals[key] += 1 if key == 'sessions' else row[key]
        print('verified', i + 1, len(sessions), sid, 'old/new empty', row['old_blank'], row['new_blank'], flush=True)
    if sha(split_path) != split_digest:
        raise RuntimeError('Split changed during generation; revalidate outputs before use')
    cfg = OmegaConf.load(MODEL / 'config/task/imagepressure2smpl.yaml')
    cfg.name = 'imagepressure2smpl_carpet_v2'
    cfg.cam_id = args.cam
    root_uri = str(output / f'cam{args.cam}') if args.output_root else f'model-input://MotionPRO/{VERSION}/cam{args.cam}'
    for key in ('seq_root', 'train_data_dir', 'eval_data_dir'):
        cfg[key] = root_uri
    cfg.pressure_shape = geometry['shape_hw']
    cfg.pressure_hw = geometry['entry_hw']
    cfg.pressure_resize_mode = 'area'
    cfg.pressure_geometry_sha256 = geometry['sha256']
    cfg.unified_prediction_dir = 'results://baselines/MotionPRO/predictions/carpet_v2/eval_motion'
    cfg.checkpoint_path = None
    cfg.train_continue = False
    OmegaConf.save(cfg, config_path)
    # Camera-specific opt-in config for the normal Hydra training entry.
    installed_config = MODEL / f'config/task/imagepressure2smpl_carpet_v2_cam{args.cam}.yaml'
    if installed_config.exists():
        raise FileExistsError(installed_config)
    OmegaConf.save(cfg, installed_config)
    summary = dict(status='PASS', geometry=geometry, totals=totals, details=details,
        old_inputs_unchanged=True, unchanged_targets_and_features=True, model_retrained=False,
        split_file_sha256=sha(split_path), training_config=str(installed_config),
        new_inputs=str(output / f'cam{args.cam}'), source_original_hashes=originals)
    (output / f'validation_cam{args.cam}.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', geometry_shape=geometry['shape_hw'], entry_hw=geometry['entry_hw'], totals=totals, training_config=str(installed_config)), indent=2), flush=True)


if __name__ == '__main__':
    main()
