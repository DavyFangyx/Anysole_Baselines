import argparse
import os
import os.path as osp
from collections import defaultdict

import copy
import hydra
import numpy as np
import smplx
import torch
import torch.nn as nn
from loguru import logger as log
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.dataset.image_pressure import ImagePressureDataset, apply_cam_seq_root
from lib.model.FRAPPE import FRAPPE
from lib.util.workspace import RESULTS_ROOT, resolve_path


class Loss(nn.Module):
    def __init__(self, weight):
        super(Loss, self).__init__()
        self.weight = weight
        self.mse = torch.nn.MSELoss()
        self.mae_smooth = torch.nn.SmoothL1Loss()

        # Official ten-column contact alignment (upstream contract): the ten
        # columns align to SMPL joints 1,2,4,5,7,8,10,11,20,21.  In the
        # insole adaptation only the left/right foot columns carry non-zero
        # weights, so footContactIds addresses *contact columns* 6/7 while
        # the predicted/GT joints use SMPL FOOT_JOINT_IDS [10,11] (M1 fix:
        # the columns must never be reused as joint indices).
        self.contactIds = [1,2,4,5,7,8,10,11,20,21]
        self.footContactIds = [6, 7]

    def cal_footloss(self, pred, target, contact):

        loss_contact = torch.sum((pred - target) ** 2, dim=2)
        contact_num = torch.sum(contact)
        if contact_num.item() == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        loss = torch.sum(loss_contact * contact) / contact_num

        return loss

    def forward(self, pred, target, contact):
        loss_theta = self.mse(pred['theta'], target['theta'])
        loss_trans = self.mae_smooth(pred['trans'], target['trans'])
        loss_2d_joint = self.mae_smooth(pred['joint'][:,:,[0,2]]-pred['joint'][:,0:1,[0,2]], target['joint'][:,:,[0,2]]-target['joint'][:,0:1,[0,2]])
        loss_joint = self.mae_smooth(pred['joint']-pred['joint'][:,[0]], target['joint']-target['joint'][:,[0]])
        foot_contact = contact[:, self.footContactIds]
        loss_foot = self.cal_footloss(
            pred['joint'][:, FOOT_JOINT_IDS],
            target['joint'][:, FOOT_JOINT_IDS],
            foot_contact,
        )
        
        loss = loss_theta*self.weight['theta']+ loss_trans*self.weight['trans']+ loss_2d_joint*self.weight['2d_joint']+ loss_joint*self.weight['joint'] + loss_foot*self.weight['foot']

        loss_item = {
            'loss': loss,
            'loss_theta': loss_theta,
            'loss_trans': loss_trans,
            'loss_2d_joint': loss_2d_joint,
            'loss_joint': loss_joint,
            'loss_foot': loss_foot,
        }

        return loss_item


FOOT_JOINT_IDS = [10, 11]
FOOT_HEIGHT_AXIS = 1
FOOT_HEIGHT_THRESH = 0.05
PRESSURE_CONTACT_IDS = [6, 7]


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--wandb_mode', choices=['disabled', 'offline', 'online'])
    parser.add_argument('--wandb_project')
    parser.add_argument('--wandb_entity')
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def flatten_batch(item, device):
    for key in item:
        if key == 'session_id':
            if isinstance(item[key], (list, tuple)):
                item[key] = list(item[key])
            continue
        if key != 'pressure' and key != 'feature':
            if key == 'joint':
                item[key] = item[key].reshape(-1, 23, 3)
            elif key in ('valid', 'frame_index'):
                item[key] = item[key].reshape(-1)
            else:
                item[key] = item[key].reshape(-1, item[key].shape[-1])
        item[key] = item[key].to(device=device, non_blocking=True)
    return item


def expand_session_ids(session_ids, n_frames, device_n=None):
    target_n = n_frames if device_n is None else device_n
    if len(session_ids) == 0:
        return ['unknown'] * target_n
    if len(session_ids) == target_n:
        return list(session_ids)
    if target_n % len(session_ids) == 0:
        repeats = target_n // len(session_ids)
        return [sid for sid in session_ids for _ in range(repeats)]
    repeats = max(1, target_n // len(session_ids))
    expanded = []
    for sid in session_ids:
        expanded.extend([sid] * repeats)
    if len(expanded) < target_n:
        expanded.extend([session_ids[-1]] * (target_n - len(expanded)))
    return expanded[:target_n]


def window_ground_height(gt_joint, valid, n_windows):
    n_frames = gt_joint.shape[0]
    if n_windows <= 0 or n_frames % n_windows != 0:
        feet = gt_joint[:, FOOT_JOINT_IDS, FOOT_HEIGHT_AXIS]
        kept = feet[valid] if valid.any() else feet
        return kept.min().expand(n_frames)
    window_length = n_frames // n_windows
    feet = gt_joint[:, FOOT_JOINT_IDS, FOOT_HEIGHT_AXIS].reshape(n_windows, window_length, -1)
    valid_w = valid.reshape(n_windows, window_length)
    ground = []
    for window_idx in range(n_windows):
        kept = feet[window_idx][valid_w[window_idx]]
        if kept.numel() == 0:
            kept = feet[window_idx]
        ground.append(kept.min().expand(window_length))
    return torch.cat(ground, dim=0)


def predict_contact(pred_joint, gt_joint, valid, n_windows):
    ground = window_ground_height(gt_joint, valid, n_windows)
    foot_height = pred_joint[:, FOOT_JOINT_IDS, FOOT_HEIGHT_AXIS] - ground.unsqueeze(1)
    return (foot_height < FOOT_HEIGHT_THRESH).any(dim=1)


def predict_smpl(model, smpl, item):
    smpl_params = model(item['feature'], item['pressure'])
    pred_beta, pred_theta, pred_trans = torch.split(smpl_params, [10, 72, 3], dim=1)
    pred_global_orient, pred_body_pose = torch.split(pred_theta, [3, 69], dim=1)
    smpl_result = smpl(
        betas=item['beta'],
        body_pose=pred_body_pose,
        global_orient=pred_global_orient,
        transl=pred_trans,
    )
    pred_joint = smpl_result.joints[:, :23]
    pred = {
        'beta': pred_beta,
        'theta': pred_theta,
        'trans': pred_trans,
        'joint': pred_joint,
    }
    target = {
        'beta': item['beta'],
        'theta': item['theta'],
        'trans': item['trans'],
        'joint': item['joint'],
    }
    return smpl_params, pred, target


def batch_valid_mask(item, n_frames):
    if 'valid' in item:
        return item['valid'].reshape(n_frames) > 0.5
    return torch.ones(n_frames, dtype=torch.bool, device=item['joint'].device)


def total_grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total += param_norm.item() ** 2
    return total ** 0.5


class DisabledWandbRun:
    def log(self, *args, **kwargs):
        return None

    def finish(self):
        return None


def init_wandb(cfg, hydra_path):
    mode = os.environ.get('WANDB_MODE', str(cfg.get('wandb_mode', 'disabled')))
    os.environ['WANDB_MODE'] = mode
    try:
        import wandb
    except ImportError:
        if mode != 'disabled':
            raise
        log.warning('wandb is not installed; logging is disabled')
        return DisabledWandbRun()
    wandb_kwargs = {
        'project': os.environ.get('WANDB_PROJECT', str(cfg.get('wandb_project', 'MotionPRO'))),
        'entity': os.environ.get('WANDB_ENTITY', str(cfg.get('wandb_entity', 'davyfangyuxuan-nanjing-university-of-aeronautics-and-ast'))),
        'mode': mode,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'dir': hydra_path,
        'job_type': 'train',
    }
    if mode == 'disabled':
        wandb_kwargs['anonymous'] = 'allow'
    run = wandb.init(**wandb_kwargs)
    log.info(f"wandb mode={mode} project={wandb_kwargs['project']} entity={wandb_kwargs['entity']}")
    return run


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['task']['gpu'])
    if os.environ.get('WANDB_MODE'):
        cfg.wandb_mode = os.environ['WANDB_MODE']
    if os.environ.get('WANDB_PROJECT'):
        cfg.wandb_project = os.environ['WANDB_PROJECT']
    if os.environ.get('WANDB_ENTITY'):
        cfg.wandb_entity = os.environ['WANDB_ENTITY']

    orig_cwd = hydra.utils.get_original_cwd()
    os.chdir(orig_cwd)
    apply_cam_seq_root(cfg.task)
    for key in ('split_csv', 'seq_root', 'smpl_model', 'smpl_mean_params', 'checkpoint_dir', 'result_dir', 'checkpoint_path'):
        value = str(cfg.task.get(key, '') or '')
        if value and value not in ('None', 'none', 'null'):
            cfg.task[key] = resolve_path(value, orig_cwd)

    # manual log
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    hydra_name = hydra.core.hydra_config.HydraConfig.get().job.name
    log.add(os.path.join(hydra_path, hydra_name+'.log'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Working directory: {os.getcwd()}")
    log.info(f"Sequence root: {cfg.task.seq_root}")
    log.info(f"Configuration: {cfg}")
    log.info(f"PyTorch version: {torch.__version__}")

    wandb_run = init_wandb(cfg, hydra_path)

    task_info = os.path.join(str(cfg['task']['name']), str(cfg['task']['loss_name']), str(cfg['task']['learning_rate']))

    tensorboard_path = os.path.join(RESULTS_ROOT, 'MotionPRO', 'tensorboard', task_info)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)

    checkpoint_dir = os.path.join(cfg['task']['checkpoint_dir'], task_info)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    result_dir = os.path.join(cfg['task']['result_dir'], task_info)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    np.random.seed(0)
    torch.manual_seed(0)
    model = torch.nn.DataParallel(FRAPPE()) # model
    smpl = smplx.create(cfg['task']['smpl_model']).to(device)

    train_dataset = ImagePressureDataset(cfg, 'train')
    eval_dataset = ImagePressureDataset(cfg, 'eval')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['task']['batch_size'], shuffle=False, pin_memory=True, num_workers=4)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg['task']['batch_size'], shuffle=False, pin_memory=True, num_workers=4)

    model.to(device=device, non_blocking=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['task']['learning_rate'], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=cfg['task']['scheduler_patience'], verbose=True)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info (f'pytorch_total_params: {pytorch_total_params}')

    start_epoch = 0
    best_eval_loss = np.inf
    global_step = 0

    weight = {
        'theta': cfg['task']['lamda_theta'],
        'trans': cfg['task']['lamda_trans'],
        '2d_joint': cfg['task']['lamda_2d_joint'],
        'joint': cfg['task']['lamda_joint'],
        'foot': cfg['task']['lamda_foot'],
    }
    
    loss = Loss(weight)

    if cfg['task']['train_continue']:

        checkpoint = torch.load(cfg['task']['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        log.info("Now continue training")
        log.info(f"Start epoch: {start_epoch}")

    for epoch in tqdm(range(start_epoch, cfg['task']['epochs'])):

        smpl_params_all = []
        joint_all = []
        
        loss_train = {
            'loss': 0.0,
            'loss_theta': 0.0,
            'loss_trans': 0.0,
            'loss_2d_joint': 0.0,
            'loss_joint': 0.0,
            'loss_foot': 0.0
        }

        loss_eval = copy.deepcopy(loss_train)
        
        for i_batch, item in enumerate(train_dataloader, 0):
            model.train(True)
            optimizer.zero_grad()

            item = flatten_batch(item, device)
            
            with torch.set_grad_enabled(True):
                _, pred_train, target_train = predict_smpl(model, smpl, item)
            
            loss_item_train = loss(pred_train, target_train, item['contact'])

            loss_item_train['loss'].backward()
            grad_norm = total_grad_norm(model.parameters())
            optimizer.step()

            for key in loss_item_train:
                loss_train[key] += loss_item_train[key].item()

            wandb_run.log({
                'loss/total': loss_item_train['loss'].item(),
                'loss/foot': loss_item_train['loss_foot'].item(),
                'loss/joint': loss_item_train['loss_joint'].item(),
                'grad_norm/total': grad_norm,
                'epoch': epoch,
            }, step=global_step)
            global_step += 1
            
        log.info("Now running on val set")

        mpjpe_sum = 0.0
        mpjpe_count = 0.0
        # --- diagnostic_only native contact IoU --------------------------------
        # contact_inter/union and every `contact_iou` log below are training
        # logs only: they do not feed the scheduler (scheduler.step uses
        # loss_eval['loss']) and do not select the best checkpoint (best uses
        # loss_eval['loss']).  They are not formal public metrics and must not
        # be read by Baselines/utils/evaluate.py.
        contact_inter = 0.0
        contact_union = 0.0
        sess_mpjpe_sum = defaultdict(float)
        sess_mpjpe_count = defaultdict(float)
        sess_contact_inter = defaultdict(float)
        sess_contact_union = defaultdict(float)

        for i_batch, item in enumerate(eval_dataloader, 0):
            model.eval()
            session_ids = item.get('session_id', [])
            item = flatten_batch(item, device)

            with torch.no_grad():
                smpl_params, pred_eval, target_eval = predict_smpl(model, smpl, item)
                smpl_params_all.append(smpl_params.cpu().detach().numpy())
                pred_joint = pred_eval['joint']
                joint_all.append(pred_joint.cpu().detach().numpy())

            loss_item_eval = loss(pred_eval, target_eval, item['contact'])

            for key in loss_item_eval:
                loss_eval[key] += loss_item_eval[key].item()

            n_frames = pred_joint.shape[0]
            valid = batch_valid_mask(item, n_frames)
            if valid.any():
                per_frame = torch.norm(pred_joint - item['joint'], dim=-1).mean(dim=-1)
                mpjpe_sum += (per_frame * valid.float()).sum().item()
                mpjpe_count += valid.float().sum().item()

                n_windows = len(session_ids) if len(session_ids) > 0 else 1
                pred_contact = predict_contact(pred_joint, item['joint'], valid, n_windows)
                # diagnostic_only: the private soft-f6 weights are binarized
                # (>0.5) only for this log; the soft values themselves feed
                # loss_foot unchanged.
                gt_contact = (item['contact'][:, PRESSURE_CONTACT_IDS] > 0.5).any(dim=1)
                pred_on = pred_contact & valid
                gt_on = gt_contact & valid
                contact_inter += (pred_on & gt_on).sum().item()
                contact_union += (pred_on | gt_on).sum().item()

                session_ids = expand_session_ids(session_ids, n_frames)
                per_frame_np = per_frame.detach().cpu().numpy()
                valid_np = valid.detach().cpu().numpy()
                pred_on_np = pred_on.detach().cpu().numpy()
                gt_on_np = gt_on.detach().cpu().numpy()
                for sid, err, keep, pred_hit, gt_hit in zip(session_ids, per_frame_np, valid_np, pred_on_np, gt_on_np):
                    if not keep:
                        continue
                    sess_mpjpe_sum[sid] += float(err)
                    sess_mpjpe_count[sid] += 1.0
                    sess_contact_inter[sid] += float(pred_hit and gt_hit)
                    sess_contact_union[sid] += float(pred_hit or gt_hit)

        for key in loss_train:
            writer.add_scalar(f'train/{key}', loss_train[key], epoch)
        for key in loss_eval:
            writer.add_scalar(f'eval/{key}', loss_eval[key], epoch)

        val_mpjpe = mpjpe_sum / mpjpe_count if mpjpe_count > 0 else float('nan')
        # diagnostic_only (see marker above the accumulator block)
        val_contact_iou = contact_inter / contact_union if contact_union > 0 else float('nan')
        val_payload = {
            'val/mpjpe': val_mpjpe,
            'val/contact_iou_diagnostic_only': val_contact_iou,
            'epoch': epoch,
        }
        for sid in sorted(sess_mpjpe_count):
            val_payload[f'val_by_session/{sid}/mpjpe'] = sess_mpjpe_sum[sid] / sess_mpjpe_count[sid]
            union = sess_contact_union[sid]
            val_payload[f'val_by_session/{sid}/contact_iou_diagnostic_only'] = (
                sess_contact_inter[sid] / union if union > 0 else float('nan')
            )
        wandb_run.log(val_payload, step=global_step)

        scheduler.step(loss_eval['loss'])

        last_checkpoint_path = osp.join(checkpoint_dir, f'imagepressure2smpl_{epoch-1}.pth')
        if osp.exists(last_checkpoint_path):
            os.remove(last_checkpoint_path)
        
        if loss_eval['loss'] < best_eval_loss:
            best_eval_loss = loss_eval['loss']
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eval_loss': loss_eval},
            osp.join(checkpoint_dir, f'imagepressure2smpl_best.pth'))

            eval_smpl_params = np.concatenate(np.array(smpl_params_all), axis=0)
            joint_all = np.concatenate(np.array(joint_all), axis=0)
            output = {'smpl_params': eval_smpl_params,
                      'joint': joint_all}
            output_file = os.path.join(result_dir, f'eval_result_best.npy')
            np.save(output_file, output)
        else:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eval_loss': loss_eval},
            osp.join(checkpoint_dir, f'imagepressure2smpl_{epoch}.pth'))
        
            eval_smpl_params = np.concatenate(np.array(smpl_params_all), axis=0)
            joint_all = np.concatenate(np.array(joint_all), axis=0)
            output = {'smpl_params': eval_smpl_params,
                        'joint': joint_all}
            output_file = os.path.join(result_dir, f'eval_result_latest.npy')
            np.save(output_file, output)

        log.info(f"Epoch: {epoch}")
        log.info("Train Loss: %.6f, Evaluate Loss: %.6f, MPJPE: %.6f, Contact IoU: %.6f" % (
            loss_train['loss'], loss_eval['loss'], val_mpjpe, val_contact_iou))

    wandb_run.finish()

    best_checkpoint = os.path.join(checkpoint_dir, 'imagepressure2smpl_best.pth')
    if os.path.isfile(best_checkpoint):
        from app.test_frappe import evaluate_checkpoint
        log.info(f'Training finished. Evaluating {best_checkpoint}')
        evaluate_checkpoint(cfg, best_checkpoint, result_dir, device=device)
    else:
        log.warning(f'Training finished without best checkpoint: {best_checkpoint}')

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
