"""Regression checks for fixed world origins and nonnegative signal retention."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'Baselines/MotionPRO'
for path in (ROOT, MODEL):
    sys.path.insert(0, str(path))
from lib.util.carpet_v2 import fit_geometry, footprint_bounds, render_carpet, world_to_pixel, pixel_to_world
from lib.dataset.image_pressure import ImagePressureDataset, apply_cam_seq_root
from AnysoleWorkspace.tool.adapters.MotionPRO import adapter as A


def test_world_coordinates_survive_new_origin():
    g = fit_geometry([(np.array([-.5, -1.]), np.array([.5, 1.]))], ['train-only'])
    xy = np.array([[-.4, .8], [.2, -.9], [0., 0.]])
    np.testing.assert_allclose(pixel_to_world(world_to_pixel(xy, g), g), xy, atol=1e-12)
    assert g['fit_sessions'] == ['train-only']
    assert not g['per_session_recentering']
    np.testing.assert_allclose(pixel_to_world([0., 0.], g), np.asarray(g['origin_xy_m']) + [.00625, -.00625])


def test_negative_x_and_positive_y_footprints_are_preserved():
    tracks = {side: dict(center=np.array([center], float), direction=np.array([[0., -1.]]))
              for side, center in [('left', [-.2, .15]), ('right', [.2, .15])]}
    g = fit_geometry([footprint_bounds(tracks)], ['train-only'])
    values = np.full((1, 48), 100., np.float32)
    old = np.zeros((320, 120), np.float32)
    for side in tracks:
        A._paint_footprint(old, tracks[side]['center'][0], tracks[side]['direction'][0], values[0])
    assert not old.any(), 'fixture must expose the legacy out-of-bounds problem'
    new = render_carpet(tracks, values, values, g, A._paint_footprint)
    assert new.any()
    assert new.max() <= 255.


def test_area_resize_keeps_edge_impulse_and_constant_values():
    reader = ImagePressureDataset.__new__(ImagePressureDataset)
    reader.pressure_hw = (96, 96)
    impulse = np.zeros((1, 320, 120), np.float32)
    impulse[0, 0, 0] = 1.
    reader.pressure_resize_mode = 'bilinear'
    assert not reader._resize_pressure(impulse).any(), 'legacy sampler must miss this impulse'
    reader.pressure_resize_mode = 'area'
    assert reader._resize_pressure(impulse).any()
    np.testing.assert_allclose(reader._resize_pressure(np.full_like(impulse, .25)).numpy(), .25, atol=1e-7)


def test_legacy_resize_is_unchanged():
    reader = ImagePressureDataset.__new__(ImagePressureDataset)
    reader.pressure_hw = (96, 96)
    reader.pressure_resize_mode = 'bilinear'
    pressure = np.random.default_rng(4).random((2, 320, 120)).astype(np.float32)
    expected = torch.nn.functional.interpolate(torch.from_numpy(pressure)[:, None], size=(96, 96), mode='bilinear', align_corners=False)[:, 0]
    torch.testing.assert_close(reader._resize_pressure(pressure), expected, rtol=0, atol=0)


def test_camera_switch_retains_explicit_adapter_version():
    cfg = dict(cam_id=4, seq_root='model-input://MotionPRO/adapter_v2_carpet/cam3',
               train_data_dir='model-input://MotionPRO/adapter_v2_carpet/cam3',
               eval_data_dir='model-input://MotionPRO/adapter_v2_carpet/cam3')
    apply_cam_seq_root(cfg)
    for key in ['seq_root', 'train_data_dir', 'eval_data_dir']:
        assert cfg[key].endswith('/MotionPRO/adapter_v2_carpet/cam4')


def test_new_prediction_directory_does_not_overwrite_legacy():
    import tempfile
    from unittest.mock import patch
    from app import test_frappe as exporter
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        args=('SYNTH', {0:np.zeros((24,3))}, {0:np.zeros((8,3))},
              {0:np.zeros(72)}, {0:True}, 1, 'synthetic-checkpoint')
        with patch.object(exporter,'RESULTS_ROOT',root):
            exporter.write_unified_motion(*args)
            legacy=root/'baselines/MotionPRO/predictions/eval_motion/SYNTH.npz'
            before=legacy.read_bytes()
            exporter.write_unified_motion(*args,output_dir=root/'new/eval_motion')
            assert legacy.read_bytes()==before
            assert (root/'new/eval_motion/SYNTH.npz').is_file()
