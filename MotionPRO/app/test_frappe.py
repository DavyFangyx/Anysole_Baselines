import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import smplx
import torch
from loguru import logger as log
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from app.train_frappe import expand_session_ids, flatten_batch, parse_cli_args
from lib.dataset.image_pressure import ImagePressureDataset, apply_cam_seq_root
from lib.eval.metrics import DEFAULT_FPS, compute_session_metrics, write_metrics_files
from lib.model.FRAPPE import FRAPPE
from lib.util.workspace import RESULTS_ROOT, resolve_path as resolve_workspace_path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from anysole.types import JOINT_NAMES  # noqa: E402


def smpl_yup_to_display(points):
    points = np.asarray(points, dtype=np.float32)
    return np.stack((points[..., 0], -points[..., 2], points[..., 1]), axis=-1)


def write_unified_motion(session_id, pred_by_frame, vertices_by_frame, poses_by_frame,
                         valid_by_frame, n_frames, checkpoint_path):
    """Write one MotionPRO session in the shared SMPL-24 eval_motion contract."""
    joints = np.zeros((n_frames, 24, 3), dtype=np.float32)
    sample_vertices = next(iter(vertices_by_frame.values()))
    vertices = np.zeros((n_frames, sample_vertices.shape[0], 3), dtype=np.float32)
    poses = np.zeros((n_frames, 72), dtype=np.float32)
    valid = np.zeros((n_frames,), dtype=bool)
    for frame, value in pred_by_frame.items():
        frame = int(frame)
        if 0 <= frame < n_frames:
            joints[frame] = smpl_yup_to_display(value)
            vertices[frame] = smpl_yup_to_display(vertices_by_frame[frame])
            poses[frame] = np.asarray(poses_by_frame[frame], dtype=np.float32).reshape(72)
            valid[frame] = bool(valid_by_frame.get(frame, False))
    output = RESULTS_ROOT / "baselines" / "MotionPRO" / "predictions" / "eval_motion" / f"{session_id}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        joint_xyz_world=joints,
        vertices_world=vertices,
        poses=poses,
        joint_names=np.asarray(JOINT_NAMES),
        valid_mask=valid,
        frame_indices=np.arange(n_frames, dtype=np.int64),
        motion_protocol=np.asarray("smpl24"),
        model_units=np.asarray("m"),
        coordinate_system=np.asarray("world_z_up"),
        joint_coordinate_system=np.asarray("world_z_up"),
        session_id=np.asarray(session_id),
        source_native_output=np.asarray(str(checkpoint_path)),
        target_fps=np.asarray(40.0, dtype=np.float32),
        # 2026-10-01 U7 拍 A / E6：三基线 provenance 统一——MotionPRO 的
        # pose/shape 全部来自模型输出（与注册表 sources 一致），PVE 可评估。
        provenance_source_type=np.asarray("model_prediction"),
        surface_source=np.asarray("model_prediction"),
        shape_source=np.asarray("model_prediction"),
        public_surface_metrics=np.asarray("true"),
    )
    log.info(f"Wrote unified MotionPRO prediction: {output}")


class SessionAccumulator:
    def __init__(self):
        self.n_frames = 0
        self.pred_joints = {}
        self.gt_joints = {}
        self.pred_verts = {}
        self.gt_verts = {}
        self.pred_rotations = {}
        self.gt_rotations = {}
        self.valid = {}

    def add(self, frame_idx, pred_joint, gt_joint, pred_vert, gt_vert,
            pred_rotation, gt_rotation, keep):
        frame_idx = int(frame_idx)
        if frame_idx < 0:
            return
        self.n_frames = max(self.n_frames, frame_idx + 1)
        self.pred_joints[frame_idx] = np.asarray(pred_joint, dtype=np.float32)
        self.gt_joints[frame_idx] = np.asarray(gt_joint, dtype=np.float32)
        self.pred_verts[frame_idx] = np.asarray(pred_vert, dtype=np.float32)
        self.gt_verts[frame_idx] = np.asarray(gt_vert, dtype=np.float32)
        self.pred_rotations[frame_idx] = np.asarray(pred_rotation, dtype=np.float32).reshape(24, 3)
        self.gt_rotations[frame_idx] = np.asarray(gt_rotation, dtype=np.float32).reshape(24, 3)
        self.valid[frame_idx] = bool(keep)

    def packed(self):
        n = self.n_frames
        if n == 0:
            empty_j = np.zeros((0, 23, 3), dtype=np.float32)
            empty_v = np.zeros((0, 0, 3), dtype=np.float32)
            empty_r = np.zeros((0, 24, 3), dtype=np.float32)
            return empty_j, empty_j, empty_v, empty_v, empty_r, empty_r, np.zeros((0,), dtype=bool)
        sample_v = next(iter(self.pred_verts.values()))
        n_verts = sample_v.shape[0]
        pred_j = np.zeros((n, 23, 3), dtype=np.float32)
        gt_j = np.zeros((n, 23, 3), dtype=np.float32)
        pred_v = np.zeros((n, n_verts, 3), dtype=np.float32)
        gt_v = np.zeros((n, n_verts, 3), dtype=np.float32)
        pred_r = np.zeros((n, 24, 3), dtype=np.float32)
        gt_r = np.zeros((n, 24, 3), dtype=np.float32)
        keep = np.zeros((n,), dtype=bool)
        for t, is_valid in self.valid.items():
            if not is_valid:
                continue
            pred_j[t] = self.pred_joints[t]
            gt_j[t] = self.gt_joints[t]
            pred_v[t] = self.pred_verts[t]
            gt_v[t] = self.gt_verts[t]
            pred_r[t] = self.pred_rotations[t]
            gt_r[t] = self.gt_rotations[t]
            keep[t] = True
        return pred_j, gt_j, pred_v, gt_v, pred_r, gt_r, keep


def resolve_path(path, orig_cwd):
    return resolve_workspace_path(path, orig_cwd)


def resolve_checkpoint(cfg, orig_cwd, task_info):
    explicit = str(cfg['task'].get('checkpoint_path', 'None'))
    if explicit not in ('', 'None', 'none', 'null'):
        path = resolve_path(explicit, orig_cwd)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'checkpoint_path does not exist: {path}')
        return path
    checkpoint_dir = resolve_path(cfg['task']['checkpoint_dir'], orig_cwd)
    path = os.path.join(checkpoint_dir, task_info, 'imagepressure2smpl_best.pth')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing best checkpoint: {path}')
    return path


def fmt_metric(value):
    if value is None or not math.isfinite(value):
        return 'nan'
    return f'{value:.3f}'


def smpl_output(smpl, beta, theta, trans):
    pred_global_orient, pred_body_pose = torch.split(theta, [3, 69], dim=1)
    return smpl(
        betas=beta,
        body_pose=pred_body_pose,
        global_orient=pred_global_orient,
        transl=trans,
    )


def evaluate_checkpoint(cfg, checkpoint_path, result_dir, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fps = float(cfg['task'].get('eval_fps', DEFAULT_FPS))
    os.makedirs(result_dir, exist_ok=True)

    model = torch.nn.DataParallel(FRAPPE())
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    smpl = smplx.create(cfg['task']['smpl_model']).to(device)
    dataset = ImagePressureDataset(cfg, 'test')
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['task']['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    sessions = defaultdict(SessionAccumulator)
    canonical_pred = defaultdict(dict)
    canonical_pred_vertices = defaultdict(dict)
    canonical_pred_poses = defaultdict(dict)
    canonical_valid = defaultdict(dict)

    with torch.no_grad():
        for item in dataloader:
            item = flatten_batch(item, device)
            smpl_params = model(item['feature'], item['pressure'])
            _, pred_theta, pred_trans = torch.split(smpl_params, [10, 72, 3], dim=1)
            pred_smpl = smpl_output(smpl, item['beta'], pred_theta, pred_trans)
            gt_smpl = smpl_output(smpl, item['beta'], item['theta'], item['trans'])
            pred_joint = pred_smpl.joints[:, :23]
            gt_joint = gt_smpl.joints[:, :23]
            pred_verts = pred_smpl.vertices
            gt_verts = gt_smpl.vertices
            n_frames = pred_joint.shape[0]
            valid = item['valid'].reshape(n_frames) > 0.5
            frame_index = item['frame_index'].reshape(n_frames).detach().cpu().numpy()
            session_ids = expand_session_ids(item['session_id'], n_frames)
            pred_j_np = pred_joint.detach().cpu().numpy()
            pred_j24_np = pred_smpl.joints[:, :24].detach().cpu().numpy()
            gt_j_np = gt_joint.detach().cpu().numpy()
            pred_v_np = pred_verts.detach().cpu().numpy()
            gt_v_np = gt_verts.detach().cpu().numpy()
            pred_r_np = pred_theta.detach().cpu().numpy().reshape(-1, 24, 3)
            gt_r_np = item['theta'].detach().cpu().numpy().reshape(-1, 24, 3)
            valid_np = valid.detach().cpu().numpy()
            for sid, t, pj, pj24, gj, pv, gv, pr, gr, keep in zip(
                session_ids, frame_index, pred_j_np, pred_j24_np, gt_j_np,
                pred_v_np, gt_v_np, pred_r_np, gt_r_np, valid_np
            ):
                sessions[sid].add(t, pj, gj, pv, gv, pr, gr, keep)
                if int(t) >= 0:
                    canonical_pred[sid][int(t)] = pj24
                    canonical_pred_vertices[sid][int(t)] = pv
                    canonical_pred_poses[sid][int(t)] = pr.reshape(72)
                    canonical_valid[sid][int(t)] = bool(keep)

    rows = []
    for sid in sorted(sessions):
        pred_j, gt_j, pred_v, gt_v, pred_r, gt_r, keep = sessions[sid].packed()
        rows.append(compute_session_metrics(
            pred_j, gt_j, pred_v, gt_v, keep, fps=fps, session_id=sid,
            pred_rotations=pred_r, gt_rotations=gt_r,
        ))
        row = rows[-1]
        log.info(
            f"{sid}: n={int(row['n_valid_frames'])} "
            f"MPJPE={fmt_metric(row['mpjpe_mm'])} PA={fmt_metric(row['pa_mpjpe_mm'])} "
            f"PVE={fmt_metric(row['pve_mm'])} Accel={fmt_metric(row['accel_error_m_s2'])} "
            f"WMPJPE={fmt_metric(row['w_mpjpe100_mm'])} "
            f"RTE={fmt_metric(row['root_rte_percent'])}% "
            f"Jerk={fmt_metric(row['jitter_pred_m_s3'])} WBCE={fmt_metric(row['WBCE'])}"
        )
        write_unified_motion(
            sid,
            canonical_pred[sid],
            canonical_pred_vertices[sid],
            canonical_pred_poses[sid],
            canonical_valid[sid],
            sessions[sid].n_frames,
            checkpoint_path,
        )

    csv_path, log_path = write_metrics_files(rows, result_dir, fps=fps)
    log.info(f'Wrote {csv_path}')
    log.info(f'Wrote {log_path}')
    return csv_path, log_path


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['task']['gpu'])
    orig_cwd = hydra.utils.get_original_cwd()
    os.chdir(orig_cwd)
    apply_cam_seq_root(cfg.task)
    for key in ('split_csv', 'seq_root', 'smpl_model', 'smpl_mean_params', 'checkpoint_dir', 'result_dir'):
        if cfg.task.get(key):
            cfg.task[key] = resolve_path(cfg.task[key], orig_cwd)

    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    hydra_name = hydra.core.hydra_config.HydraConfig.get().job.name
    log.add(os.path.join(hydra_path, hydra_name + '.log'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_info = os.path.join(str(cfg['task']['name']), str(cfg['task']['loss_name']), str(cfg['task']['learning_rate']))
    checkpoint_path = resolve_checkpoint(cfg, orig_cwd, task_info)
    result_dir = os.path.join(cfg['task']['result_dir'], task_info)

    log.info(f'Device: {device}')
    log.info(f'Checkpoint: {checkpoint_path}')
    log.info(f'Sequence root: {cfg.task.seq_root}')
    log.info(f'Result dir: {result_dir}')
    log.info(f'Configuration: {OmegaConf.to_yaml(cfg)}')
    evaluate_checkpoint(cfg, checkpoint_path, result_dir, device=device)


if __name__ == '__main__':
    cli_args, remaining = parse_cli_args()
    import sys
    sys.argv = [sys.argv[0]] + remaining
    if cli_args.wandb_mode is not None:
        os.environ['WANDB_MODE'] = cli_args.wandb_mode
    if cli_args.wandb_project is not None:
        os.environ['WANDB_PROJECT'] = cli_args.wandb_project
    if cli_args.wandb_entity is not None:
        os.environ['WANDB_ENTITY'] = cli_args.wandb_entity
    main()
