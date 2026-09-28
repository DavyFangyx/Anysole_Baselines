import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'app'))

from train_frappe import parse_cli_args, expand_session_ids, flatten_batch, predict_contact, FOOT_HEIGHT_THRESH
import torch


def test_parse_cli_strips_wandb_flags():
    args, remaining = parse_cli_args([
        '--wandb_mode', 'offline',
        '--wandb_project', 'MotionPRO',
        '--wandb_entity', 'davyfangyuxuan-nanjing-university-of-aeronautics-and-ast',
        'task.epochs=1',
    ])
    assert args.wandb_mode == 'offline'
    assert args.wandb_project == 'MotionPRO'
    assert remaining == ['task.epochs=1']


def test_parse_cli_keeps_hydra_overrides():
    args, remaining = parse_cli_args([
        'wandb_mode=offline',
        'wandb_project=MotionPRO',
        'task.epochs=1',
    ])
    assert args.wandb_mode is None
    assert remaining == ['wandb_mode=offline', 'wandb_project=MotionPRO', 'task.epochs=1']


def test_expand_session_ids_repeats_windows():
    ids = expand_session_ids(['S13011', 'S13021'], n_frames=6)
    assert ids == ['S13011', 'S13011', 'S13011', 'S13021', 'S13021', 'S13021']


def test_predict_contact_uses_relative_height():
    gt = torch.zeros(4, 23, 3)
    pred = torch.zeros(4, 23, 3)
    gt[:, [10, 11], 1] = -0.20
    pred[:, [10, 11], 1] = -0.20 + FOOT_HEIGHT_THRESH + 0.02
    pred[0, 10, 1] = -0.20
    pred[1, 10, 1] = -0.20 + FOOT_HEIGHT_THRESH + 0.01
    pred[2, 11, 1] = -0.20 + 0.01
    pred[3, 10, 1] = -0.20 + FOOT_HEIGHT_THRESH + 0.005
    valid = torch.ones(4, dtype=torch.bool)
    contact = predict_contact(pred, gt, valid, n_windows=1)
    assert contact.tolist() == [True, False, True, False]


def test_flatten_batch_keeps_frame_index():
    item = {
        'joint': torch.zeros(2, 20, 23, 3),
        'valid': torch.ones(2, 20),
        'frame_index': torch.arange(40).reshape(2, 20),
        'theta': torch.zeros(2, 20, 72),
        'pressure': torch.zeros(2, 20, 8, 8),
        'feature': torch.zeros(2, 20, 16),
        'session_id': ['S12021', 'S12022'],
    }
    out = flatten_batch(item, torch.device('cpu'))
    assert tuple(out['joint'].shape) == (40, 23, 3)
    assert tuple(out['valid'].shape) == (40,)
    assert tuple(out['frame_index'].shape) == (40,)
    assert tuple(out['pressure'].shape) == (2, 20, 8, 8)
    assert out['session_id'] == ['S12021', 'S12022']
    assert out['frame_index'][:5].tolist() == [0, 1, 2, 3, 4]


def test_load_split_ids_reads_test_column(tmp_path=None):
    import tempfile
    from pathlib import Path as P
    from lib.dataset.image_pressure import load_split_ids
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = P(tmp) / 'splits.csv'
        csv_path.write_text('train,val,test\nS12011,S12013,S12013\nS12012,,S13013\n')
        assert load_split_ids(csv_path, 'test') == ['S12013', 'S13013']
        assert load_split_ids(csv_path, 'train') == ['S12011', 'S12012']


if __name__ == '__main__':
    test_parse_cli_strips_wandb_flags()
    test_parse_cli_keeps_hydra_overrides()
    test_expand_session_ids_repeats_windows()
    test_predict_contact_uses_relative_height()
    test_flatten_batch_keeps_frame_index()
    test_load_split_ids_reads_test_column()
    print('ok')
