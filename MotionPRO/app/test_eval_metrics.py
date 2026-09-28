import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.eval.metrics import (
    DEFAULT_FPS,
    apply_similarity,
    compute_session_metrics,
    summarize_metrics,
    umeyama,
    write_metrics_files,
)


def _assert_close(value, target, atol=1e-5, name='value'):
    if target is None or (isinstance(target, float) and math.isnan(target)):
        assert math.isnan(value), f'{name} expected nan, got {value}'
        return
    assert abs(value - target) <= atol, f'{name}: {value} != {target} +/- {atol}'


def _zeroish(metrics, names, atol=1e-5):
    for name in names:
        _assert_close(metrics[name], 0.0, atol=atol, name=name)


def _gt_sequence(n=8, n_joints=23, n_verts=12):
    t = np.arange(n, dtype=np.float64)[:, None, None]
    joints = np.zeros((n, n_joints, 3), dtype=np.float64)
    joints[..., 0] = np.linspace(-0.2, 0.2, n_joints)[None, :]
    joints[..., 1] = np.linspace(0.0, 0.8, n_joints)[None, :]
    joints[..., 2] = 0.05 * np.sin(np.linspace(0, np.pi, n_joints))[None, :]
    joints = joints + np.array([0.02, 0.0, 0.01]) * t
    verts = np.zeros((n, n_verts, 3), dtype=np.float64)
    verts[..., 0] = np.linspace(-0.15, 0.15, n_verts)[None, :]
    verts[..., 1] = np.linspace(0.0, 0.6, n_verts)[None, :]
    verts[..., 2] = 0.03 * np.cos(np.linspace(0, np.pi, n_verts))[None, :]
    verts = verts + np.array([0.02, 0.0, 0.01]) * t
    verts[:, :3, 1] = 0.0
    return joints, verts


def test_identical_pred_is_zero():
    joints, verts = _gt_sequence()
    metrics = compute_session_metrics(joints, joints, verts, verts, fps=DEFAULT_FPS, session_id='same')
    _zeroish(metrics, ['mpjpe_mm', 'pa_mpjpe_mm', 'pve_mm', 'accel_error_m_s2',
                       'w_mpjpe100_mm', 'wa_mpjpe100_mm', 'root_ate_mm', 'WBCE'])
    _zeroish(metrics, ['jitter_pred_m_s3', 'jitter_gt_m_s3'])


def test_global_translation_is_removed_by_pelvis_and_first_two_frames():
    joints, verts = _gt_sequence()
    pred_j = joints + np.array([0.10, -0.03, 0.04])
    pred_v = verts + np.array([0.10, -0.03, 0.04])
    metrics = compute_session_metrics(pred_j, joints, pred_v, verts, fps=DEFAULT_FPS)
    _zeroish(metrics, ['mpjpe_mm', 'pa_mpjpe_mm', 'pve_mm', 'accel_error_m_s2',
                       'w_mpjpe100_mm', 'wa_mpjpe100_mm'], atol=1e-4)
    assert metrics['root_ate_mm'] > 1.0
    _assert_close(metrics['WBCE'], 30.0, atol=1e-4, name='WBCE_world')


def test_per_frame_similarity_zeroes_pmpjpe_only():
    joints, verts = _gt_sequence()
    pred_j = np.empty_like(joints)
    for t in range(joints.shape[0]):
        angle = 0.2 + 0.05 * t
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
        pred_j[t] = 1.1 * joints[t] @ R.T + np.array([0.03 * t, 0.01, -0.02])
    metrics = compute_session_metrics(pred_j, joints, fps=DEFAULT_FPS)
    _assert_close(metrics['pa_mpjpe_mm'], 0.0, atol=1e-4, name='pa_mpjpe_mm')
    assert metrics['mpjpe_mm'] > 1.0


def test_constant_velocity_has_zero_accel():
    n = 10
    gt = np.zeros((n, 23, 3))
    pred = np.zeros((n, 23, 3))
    gt[:, :, 0] = np.arange(n)[:, None] * 0.04
    pred[:, :, 0] = np.arange(n)[:, None] * 0.07
    metrics = compute_session_metrics(pred, gt, fps=DEFAULT_FPS)
    _assert_close(metrics['accel_error_m_s2'], 0.0, atol=1e-6, name='accel_error_m_s2')
    pred[4, 5] += np.array([0.0, 0.2, 0.0])
    metrics_bad = compute_session_metrics(pred, gt, fps=DEFAULT_FPS)
    assert metrics_bad['accel_error_m_s2'] > 1.0


def test_wbce_uses_ground_contact_vertices():
    n, v = 5, 8
    gt_j = np.zeros((n, 23, 3))
    pred_j = np.zeros((n, 23, 3))
    gt_v = np.zeros((n, v, 3))
    pred_v = np.zeros((n, v, 3))
    gt_v[:, :, 1] = 0.20
    gt_v[:, :2, 1] = 0.0
    pred_v[:, :, 1] = 0.20
    pred_v[:, :2, 1] = 0.03
    metrics = compute_session_metrics(pred_j, gt_j, pred_v, gt_v, fps=DEFAULT_FPS)
    _assert_close(metrics['WBCE'], 30.0, atol=1e-6, name='WBCE')
    pred_v[:, :2, 1] = 0.0
    metrics_zero = compute_session_metrics(pred_j, gt_j, pred_v, gt_v, fps=DEFAULT_FPS)
    _assert_close(metrics_zero['WBCE'], 0.0, atol=1e-6, name='WBCE_zero')


def test_umeyama_recovers_rigid_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    angle = 0.4
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    t = np.array([0.2, -0.1, 0.3])
    dst = apply_similarity(src, R, t, 1.0)
    R_hat, t_hat, scale = umeyama(src, dst, with_scale=False)
    aligned = apply_similarity(src, R_hat, t_hat, scale)
    assert np.allclose(aligned, dst, atol=1e-8)
    assert abs(scale - 1.0) < 1e-8


def test_summarize_micro_averages(tmp_path=None):
    row_a = compute_session_metrics(*([np.zeros((2, 23, 3))] * 2), session_id='A')
    joints, verts = _gt_sequence(n=4)
    pred_j = joints.copy()
    pred_j[-1, 0] += np.array([0.05, 0.0, 0.0])
    row_b = compute_session_metrics(pred_j, joints, verts, verts, session_id='B')
    row_a['n_valid_frames'] = 2
    row_b['n_valid_frames'] = 4
    summary = summarize_metrics([row_a, row_b])
    expected_mpjpe = (2 * row_a['mpjpe_mm'] + 4 * row_b['mpjpe_mm']) / 6
    _assert_close(summary['mpjpe_mm'], expected_mpjpe, atol=1e-8, name='mpjpe_micro')
    with tempfile.TemporaryDirectory() as tmp:
        csv_path, log_path = write_metrics_files([row_a, row_b], tmp, fps=DEFAULT_FPS)
        assert csv_path.is_file()
        assert 'OVERALL' in log_path.read_text()


if __name__ == '__main__':
    test_identical_pred_is_zero()
    test_global_translation_is_removed_by_pelvis_and_first_two_frames()
    test_per_frame_similarity_zeroes_pmpjpe_only()
    test_constant_velocity_has_zero_accel()
    test_wbce_uses_ground_contact_vertices()
    test_umeyama_recovers_rigid_transform()
    test_summarize_micro_averages()
    print('ok')
