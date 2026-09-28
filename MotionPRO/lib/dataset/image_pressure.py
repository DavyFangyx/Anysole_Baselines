import torch
import numpy as np
from pathlib import Path
import csv

from torch.utils.data import Dataset

from lib.util.io import load_smpl_npy
from lib.util.workspace import WORKSPACE_ROOT, sequence_root


def load_split_ids(split_csv, column):
    ids = []
    with Path(split_csv).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{split_csv} missing column {column}")
        for row in reader:
            value = (row.get(column) or "").strip()
            if value:
                ids.append(value)
    return ids


def default_seq_root(cam_id):
    return sequence_root(cam_id)


def is_default_cam_seq_root(seq_root):
    """Recognize the model-input default shape ``<root>/cam<id>`` with an
    optional adapter-version segment, e.g. ``model-input://MotionPRO/
    adapter_v1/cam3``."""
    value = str(seq_root or '')
    if not value or value in ('None', 'none', 'null'):
        return True
    normalized = Path(value).as_posix().rstrip("/")
    if not normalized.startswith('model-input://MotionPRO/'):
        return False
    name = normalized[len('model-input://MotionPRO/'):]
    if not name:
        return False
    if name.startswith('cam') and name[3:].isdigit() and '/' not in name:
        return True
    parts = name.split('/')
    return len(parts) == 2 and parts[1].startswith('cam') and parts[1][3:].isdigit()


def apply_cam_seq_root(task):
    cam_id = int(task.get('cam_id', 3))
    task['cam_id'] = cam_id
    seq_root = str(task.get('seq_root', '') or '')
    if is_default_cam_seq_root(seq_root):
        task['seq_root'] = str(default_seq_root(cam_id))
    return cam_id, task['seq_root']


def resolve_seq_root(cfg):
    task = cfg['task']
    apply_cam_seq_root(task)
    return Path(task['seq_root'])


def shared_facts_session_dir(session_dir: Path) -> Path:
    """Shared facts twin of one model input dir.

    model_inputs/MotionPRO/<version>/cam<id>/<date>/<subject>/<session>
    -> shared/facts/sessions/cam<id>/<date>/<subject>/<session>
    """
    subject = session_dir.parent.name
    date = session_dir.parent.parent.name
    cam_dir = session_dir.parent.parent.parent.name
    session_id = session_dir.name
    return WORKSPACE_ROOT / 'shared' / 'facts' / 'sessions' / cam_dir / date / subject / session_id


def session_dir_for(seq_root, session_id, cam_id=None):
    matches = sorted(seq_root.glob(f"*/*/{session_id}"))
    if not matches:
        cam_text = f' cam_id={cam_id}' if cam_id is not None else ''
        raise FileNotFoundError(
            f"No model input dir for {session_id} under {seq_root}.{cam_text} "
            f"Run the MotionPRO adapter first (C2)."
        )
    return matches[0]


def load_frames_facts(session_dir: Path) -> dict:
    facts_dir = shared_facts_session_dir(session_dir)
    frames_path = facts_dir / 'frames.npz'
    if not frames_path.is_file():
        raise FileNotFoundError(f"Missing shared facts frames: {frames_path}")
    frames = dict(np.load(frames_path, allow_pickle=True))
    frame_id = np.asarray(frames['frame_id'])
    valid = np.asarray(frames['valid'], dtype=np.uint8)
    fake = np.asarray(frames['fake'], dtype=np.uint8)
    # A frame is unusable for training when shared facts mark it invalid/fake.
    return {'frame_id': frame_id, 'invalid': (valid == 0).astype(np.uint8) | (fake != 0).astype(np.uint8)}


def window_ranges(n_frames, window_length, invalid, mode):
    """Frame windows for one session.

    M5: train/eval index complete, fully valid windows only — the tail that
    cannot fill a window never enters training and no loss mask is invented.
    test (inference export) keeps full session coverage: the final partial
    window is returned and zero-padded in ``__getitem__`` with valid=0.
    """
    if mode == 'test':
        n_windows = (n_frames + window_length - 1) // window_length
    else:
        n_windows = n_frames // window_length
    ranges = []
    for window_idx in range(n_windows):
        left = window_idx * window_length
        right = left + window_length
        if right > n_frames:
            right = n_frames
        if invalid[left:right].any():
            continue
        ranges.append((left, right))
    return ranges


class ImagePressureDataset(Dataset):

    def __init__(self, cfg, mode):
        self.mode = mode
        self.window_length = cfg['task']['window_length']
        self.dtype = torch.float32
        self.pressure_hw = (96, 96)

        seq_root = resolve_seq_root(cfg)
        cam_id = cfg['task'].get('cam_id')
        split_csv = cfg['task'].get('split_csv')
        if split_csv:
            if self.mode == 'train':
                column = 'train'
            elif self.mode == 'eval':
                column = 'val'
            elif self.mode == 'test':
                column = 'test'
            else:
                raise ValueError(f'Unknown dataset mode: {mode}')
            session_ids = load_split_ids(split_csv, column)
            self.session_dirs = [
                session_dir_for(seq_root, session_id, cam_id)
                for session_id in session_ids
            ]
        elif self.mode == 'train':
            self.session_dirs = sorted(
                path.parent for path in seq_root.glob('*/*/*/feature_hrnet.pth'))
        elif self.mode in ('eval', 'test'):
            self.session_dirs = sorted(
                path.parent for path in seq_root.glob('*/*/*/feature_hrnet.pth'))
        else:
            raise ValueError(f'Unknown dataset mode: {mode}')
        print(f"{self.mode} seq_root={seq_root} n_sessions={len(self.session_dirs)}")

        self.gt_kps = []
        self.gt_contact = []
        self.gt_smpl = []
        self.feature_all = []
        self.pressure_all = []
        self.valid_windows = []

        for file_index, session_dir in enumerate(self.session_dirs):
            feature = torch.load(session_dir / 'feature_hrnet.pth')
            self.feature_all.append(feature)

            pressure_file = session_dir / 'pressure.npz'
            gt_kps_file = session_dir / 'keypoints.npy'
            gt_smpl_file = session_dir / 'smpl.npy'
            contact_file = session_dir / 'contact.npy'
            frame_id_file = session_dir / 'frame_id.npy'

            pressure = np.load(pressure_file)['pressure']
            if tuple(pressure.shape[-2:]) != (320, 120):
                raise ValueError(
                    f'{session_dir.name}: pressure must be the virtual carpet '
                    f'(T,320,120), got {tuple(pressure.shape)}')
            if not contact_file.is_file():
                raise FileNotFoundError(
                    f"Missing MotionPRO private soft-f6 contact {contact_file}. Run "
                    f"`python -m AnysoleWorkspace.tool.adapters.MotionPRO.adapter "
                    f"--session {session_dir.name}`."
                )
            contact_gt = np.load(contact_file)
            if contact_gt.ndim != 2 or contact_gt.shape[1] != 10:
                raise ValueError(f'{contact_file}: expected (T,10) soft-f6 contact')
            facts = load_frames_facts(session_dir)
            if frame_id_file.is_file():
                saved_frame_id = np.load(frame_id_file)
                if len(saved_frame_id) != pressure.shape[0] or \
                        not np.array_equal(saved_frame_id, facts['frame_id']):
                    raise ValueError(f'{session_dir.name}: frame_id.npy differs from shared facts')
            if len(facts['frame_id']) != pressure.shape[0]:
                raise ValueError(f'{session_dir.name}: pressure T != shared facts frame count')
            kps_gt = np.load(gt_kps_file)
            smpl_gt = load_smpl_npy(gt_smpl_file)

            n_frames = pressure.shape[0]
            windows = window_ranges(n_frames, self.window_length, facts['invalid'], self.mode)
            for (left, right) in windows:
                self.valid_windows.append((file_index, left, right))

            self.pressure_all.append(pressure)
            self.gt_kps.append(kps_gt)
            self.gt_smpl.append(smpl_gt)
            self.gt_contact.append(contact_gt)

        print('Valid windows: ', len(self.valid_windows))

    def __len__(self):
        return len(self.valid_windows)

    def _resize_pressure(self, pressure):
        height, width = self.pressure_hw
        pressure = torch.from_numpy(np.ascontiguousarray(pressure)).float()
        if pressure.ndim != 3:
            raise ValueError(f'pressure must be (T, H, W), got {tuple(pressure.shape)}')
        if tuple(pressure.shape[-2:]) == (height, width):
            return pressure
        return torch.nn.functional.interpolate(
            pressure.unsqueeze(1),
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)

    def __getitem__(self, idx):
        item = {}
        file_index, window_left, window_right = self.valid_windows[idx]
        session_id = self.session_dirs[file_index].name

        pressure = self.pressure_all[file_index][window_left:window_right] / 255
        pressure = self._resize_pressure(pressure)
        feature = self.feature_all[file_index][window_left:window_right, :]
        keypoint = self.gt_kps[file_index][window_left:window_right, :23]
        contact = self.gt_contact[file_index][window_left:window_right, :10]
        smpl_gt = self.gt_smpl[file_index]

        n_real = window_right - window_left
        beta = torch.from_numpy(smpl_gt['betas'][:10]).repeat(n_real, 1).float()
        body_pose = torch.from_numpy(smpl_gt['body_pose'][window_left:window_right]).float()
        global_orient = torch.from_numpy(smpl_gt['global_orient'][window_left:window_right]).float()
        transl = torch.from_numpy(smpl_gt['transl'][window_left:window_right]).float()
        theta = torch.cat([global_orient, body_pose], dim=1)

        keypoint = torch.from_numpy(keypoint).float()
        contact = torch.from_numpy(contact).float()
        valid = torch.ones(n_real, dtype=self.dtype)
        frame_index = torch.arange(window_left, window_right, dtype=torch.long)

        if feature.shape[0] < self.window_length:
            # Only the test (full-session export) mode can reach here: the
            # final partial window is zero-padded and marked valid=0 so the
            # inference export can drop the padding. Train/eval index complete
            # windows only (M5).
            pad = self.window_length - feature.shape[0]
            pressure = torch.cat([pressure, torch.zeros(pad, pressure.shape[1], pressure.shape[2], dtype=self.dtype)], dim=0)
            feature = torch.cat([feature, torch.zeros(pad, feature.shape[1], dtype=self.dtype)], dim=0)
            keypoint = torch.cat([keypoint, torch.zeros(pad, keypoint.shape[1], keypoint.shape[2], dtype=self.dtype)], dim=0)
            contact = torch.cat([contact, torch.zeros(pad, contact.shape[1], dtype=self.dtype)], dim=0)
            beta = torch.cat([beta, torch.zeros(pad, beta.shape[1], dtype=self.dtype)], dim=0)
            theta = torch.cat([theta, torch.zeros(pad, theta.shape[1], dtype=self.dtype)], dim=0)
            transl = torch.cat([transl, torch.zeros(pad, transl.shape[1], dtype=self.dtype)], dim=0)
            valid = torch.cat([valid, torch.zeros(pad, dtype=self.dtype)], dim=0)
            frame_index = torch.cat([frame_index, torch.full((pad,), -1, dtype=torch.long)], dim=0)

        item['pressure'] = pressure
        item['feature'] = feature
        item['beta'] = beta
        item['theta'] = theta
        item['trans'] = transl
        item['joint'] = keypoint
        item['contact'] = contact
        item['valid'] = valid
        item['frame_index'] = frame_index
        item['session_id'] = session_id
        return item
