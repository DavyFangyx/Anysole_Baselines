from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple
from backward_diffusion import ControlTransformer, PriorTransformer, TransformerTranslation
from config import Config, PredictionMode
from dataset import MotionDataset
from dataset_utils import get_poses_from_data
from normalizer import Normalizer
from forward_diffusion import ForwardDiffusion
from inference_windows import inference_starts, inference_window
from skeleton_layout import metric_joint_indices


@torch.no_grad()
def validate_pose(
    val_loader: DataLoader,
    normalizer: Normalizer,
    forward_diffusion: ForwardDiffusion,
    backward_diffusion: Optional[ControlTransformer],
    backward_diffusion_prior: PriorTransformer,
    loss_fn: torch.nn.Module,
    config: Config,
    device: torch.device,
) -> float:
    if backward_diffusion is not None:
        backward_diffusion.eval()
    backward_diffusion_prior.eval()
    T = config["diffusion_val_T"]
    val_loss = 0
    for i, (x_0, c, distances, _) in enumerate(val_loader):

        x_0 = normalizer.normalize_poses(x_0)
        c = normalizer.normalize_insole(c)
        # distances = normalizer.normalize_distances(distances)

        x_0 = x_0[..., 3:]  # Keep only pose
        batch_size = x_0.shape[0]

        ts = torch.empty((batch_size,), device=device).int()
        c_mask = torch.ones(1, device=device)
        x_t = torch.randn_like(x_0, device=device)

        for t in range(T, 0, -1):  # t in [T, 1]
            ts[:] = t

            if backward_diffusion is None:
                prediction, _ = backward_diffusion_prior(
                    x=x_t, distances=distances, t=ts, control=None, c=None, c_mask=None
                )
            else:
                prediction, _ = backward_diffusion_prior(
                    x=x_t, distances=distances, t=ts, control=backward_diffusion, c=c, c_mask=c_mask
                )

            if config["prediction_mode"] == PredictionMode.X0:
                x_t_minus_one = forward_diffusion.q_posterior_mean_from_x_0(x_t, prediction, ts)
            else:
                raise ValueError(f"Not supported prediction mode: {config['prediction_mode']}")

            x_t = x_t_minus_one

        loss = loss_fn(x_0, x_t, c)
        val_loss += loss.item()
    val_loss /= len(val_loader)
    if backward_diffusion is not None:
        backward_diffusion.train()
    backward_diffusion_prior.train()
    return val_loss


@torch.no_grad()
def validate_translation(
    val_loader: DataLoader,
    normalizer: Normalizer,
    model: TransformerTranslation,
    loss_fn: torch.nn.Module,
    config: Config,
    device: torch.device,
) -> float:
    if model is not None:
        model.eval()
    val_loss = 0
    for i, (x_0, c, distances, _) in enumerate(val_loader):
        x_0 = normalizer.normalize_poses(x_0)
        c = normalizer.normalize_insole(c)
        # distances = normalizer.normalize_distances(distances)

        pose = x_0[..., 3:]
        translation = x_0[..., :3]

        prediction = model(pose, c)

        loss = loss_fn(translation, prediction)
        val_loss += loss.item()
    val_loss /= len(val_loader)
    if model is not None:
        model.train()
    return val_loss


@torch.no_grad()
def test(
    test_dataset: MotionDataset,
    normalizer: Normalizer,
    forward_diffusion: ForwardDiffusion,
    backward_diffusion: Optional[ControlTransformer],
    backward_diffusion_prior: PriorTransformer,
    model_trans: Optional[TransformerTranslation],
    loss_fn: torch.nn.Module,
    loss_fn_trans: torch.nn.Module,
    config: Config,
    w: float,
    device: torch.device,
    return_attn_mat: bool,
    vertical_zero_threshold: bool,
) -> Tuple[
    float,
    float,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
]:
    backward_diffusion_prior.eval()
    if backward_diffusion is not None:
        backward_diffusion.eval()
    if model_trans is not None:
        model_trans.eval()
    T = forward_diffusion.T
    half_temporal_stride = config["input_T"] // 2
    # The diffusion model predicts pose channels only; translation occupies
    # the first three channels of the dataset's output dimension.  Keep this
    # derived from the config because gait uses 66 pose channels (69 total),
    # while the original Step2Motion data used a different skeleton width.
    pose_dim = int(config["output_dim"]) - 3
    indices = inference_starts(len(test_dataset.poses[0]), config["input_T"])
    emitted_ids, emitted_valid = [], []
    test_loss_pose = 0
    test_loss_trans = 0
    gt = []
    preds = []
    cs = []
    attn_mats_chunks_left = []
    attn_mats_chunks_right = []
    attn_mats_chunks_body = []
    prev_x_t = None
    for i, index in enumerate(indices):
        x_0, c, distances, frame_ids, valid, real_length = inference_window(
            test_dataset, index, config["input_T"])
        emit_start = 0 if i == 0 else half_temporal_stride
        emit_end = real_length
        emitted_ids.append(frame_ids[emit_start:emit_end])
        emitted_valid.append(valid[emit_start:emit_end])
        if x_0 is not None:
            x_0 = x_0.unsqueeze(0)
            x_0 = normalizer.normalize_poses(x_0)
        c = c.unsqueeze(0)
        c = normalizer.normalize_insole(c)

        device = c.device

        if x_0 is not None:
            pose = x_0[..., 3:]
            translation = x_0[..., :3]

        if i % 50 == 0:
            print(f"Testing {i + 1}/{len(indices)}")

        ts = torch.empty((1,), device=device).int()

        if prev_x_t is None:
            prev_x_t = torch.empty((T, 1, half_temporal_stride, pose_dim), device=device)

        # Predict Pose ---------------------------------
        x_t = torch.randn(c.shape[:-1] + (pose_dim,), device=device)
        cmask_ones = torch.ones(1, device=device)
        for t in range(T, 0, -1):  # t in [T, 1]
            ts[:] = t

            # Inpainting:
            # 1- First half comes from previous frames
            # 2- Second half is inpainted
            if i > 0:
                x_t[:, :half_temporal_stride] = prev_x_t[t - 1]

            prev_x_t[t - 1] = x_t[:, half_temporal_stride:]

            if backward_diffusion is None:
                prediction, _ = backward_diffusion_prior(
                    x=x_t, distances=distances, t=ts, control=None, c=None, c_mask=None
                )
            else:
                prediction, attn_mats = backward_diffusion_prior(
                    x=x_t,
                    distances=distances,
                    t=ts,
                    control=backward_diffusion,
                    c=c,
                    c_mask=cmask_ones,
                    test=return_attn_mat and t == 1,
                )
                if w < 1.0:
                    prediction_uncond, _ = backward_diffusion_prior(
                        x=x_t, distances=distances, t=ts, control=None, c=None, c_mask=None
                    )
                    prediction = prediction_uncond + w * (prediction - prediction_uncond)

            if config["prediction_mode"] == PredictionMode.NOISE:
                x_t_minus_one = forward_diffusion.q_posterior_mean_from_noise(x_t, prediction, ts)
            elif config["prediction_mode"] == PredictionMode.X0:
                x_t_minus_one = forward_diffusion.q_posterior_mean_from_x_0(x_t, prediction, ts)
            else:
                raise ValueError(f"Unknown prediction mode: {config['prediction_mode']}")

            x_t = x_t_minus_one

        # Predict Translation ---------------------------
        prediction_trans = torch.zeros(c.shape[:-1] + (3,), device=device)
        if model_trans is not None:
            prediction_trans = prediction_trans + model_trans(x_t, c)
        else:
            # --no-imu ablation: the translation model's sole condition is the
            # deleted IMU channels, so it is not trained. Fall back to the GT
            # displacement for the exported BVH root (pose metrics are
            # root-relative and unaffected; mrpe is reported as None).
            prediction_trans = prediction_trans + x_0[..., :3]

        # Score and export only real, newly emitted frames. Padding is context.
        score_mask = torch.as_tensor(valid[emit_start:emit_end], device=device)
        if isinstance(test_dataset, MotionDataset) and score_mask.any():
            loss_pose = loss_fn(x_t[:, emit_start:emit_end][:, score_mask],
                                pose[:, emit_start:emit_end][:, score_mask],
                                c[:, emit_start:emit_end][:, score_mask])
            loss_trans = loss_fn_trans(prediction_trans[:, emit_start:emit_end][:, score_mask],
                                       translation[:, emit_start:emit_end][:, score_mask])
            test_loss_pose += loss_pose.item()
            test_loss_trans += loss_trans.item()
        if vertical_zero_threshold:
            prediction_trans[..., 1][abs(prediction_trans[..., 1]) < 5] = 0
        prediction = torch.cat([prediction_trans, x_t], dim=-1)
        gt.append((x_0 if x_0 is not None else prediction)[:, emit_start:emit_end])
        preds.append(prediction[:, emit_start:emit_end])
        cs.append(c[:, emit_start:emit_end])
        if return_attn_mat:
            for group, chunks in enumerate([attn_mats_chunks_left, attn_mats_chunks_right, attn_mats_chunks_body]):
                q = group * config["input_T"]
                chunks.append(attn_mats[0][0, :, q+emit_start:q+emit_end, emit_start:emit_end])

    test_dataset.inference_frame_ids = np.concatenate(emitted_ids)
    test_dataset.inference_valid = np.concatenate(emitted_valid)
    expected_ids = np.asarray(getattr(test_dataset, 'clip_frame_ids',
                                     [np.arange(len(test_dataset.poses[0]))])[0])
    if not np.array_equal(test_dataset.inference_frame_ids, expected_ids):
        raise RuntimeError('Inference omitted, repeated, or reordered canonical frames')

    backward_diffusion_prior.train()
    if backward_diffusion is not None:
        backward_diffusion.train()
    if model_trans is not None:
        model_trans.train()

    test_loss_pose /= len(indices)
    test_loss_trans /= len(indices)
    return (
        test_loss_pose,
        test_loss_trans,
        torch.cat(gt, dim=1).flatten(0, 1),
        torch.cat(preds, dim=1).flatten(0, 1),
        torch.cat(cs, dim=1).flatten(0, 1),
        (attn_mats_chunks_left, attn_mats_chunks_right, attn_mats_chunks_body),
    )


@torch.no_grad()
def get_metrics(normalizer: Normalizer, y: torch.Tensor, pred: torch.Tensor, framerate: float, joint_names=None) -> tuple[
    tuple[float, float, np.ndarray],
    tuple[float, float, np.ndarray],
    tuple[float, float, np.ndarray],
    tuple[float, float, np.ndarray],
    tuple[float, float, np.ndarray],
    tuple[float, float, np.ndarray],
]:
    """
    y is the ground truth tensor of shape (frames, output_dim)
    pred is the predicted tensor of shape (frames, output_dim)
    """
    y = normalizer.denormalize_poses(y).cpu().numpy()  # type: ignore
    pred = normalizer.denormalize_poses(pred).cpu().numpy()  # type: ignore
    poses_y = y[:, 3:].reshape(len(y), -1, 3)
    poses_pred = pred[:, 3:].reshape(len(pred), -1, 3)
    global_poses_y = np.cumsum(y[:, :3], axis=0)[:, None, :]
    global_poses_pred = np.cumsum(pred[:, :3], axis=0)[:, None, :]
    groups = metric_joint_indices(poses_y.shape[1] + 1, joint_names)
    # mean per joint positional error
    mpjpe = np.linalg.norm(poses_y - poses_pred, axis=-1)
    mpjpe_mean = np.mean(mpjpe)
    mpjpe_std = float(np.std(mpjpe))
    # mean per end effector positional error
    end_effectors = groups["end_effectors"]
    mpeepe = np.linalg.norm(poses_y[:, end_effectors] - poses_pred[:, end_effectors], axis=-1)
    mpeepe_mean = np.mean(mpeepe)
    mpeepe_std = float(np.std(mpeepe))
    # mean per end effector positional error (toes)
    toes = groups["toes"]
    mpeepe_toes = np.linalg.norm(poses_y[:, toes] - poses_pred[:, toes], axis=-1)
    mpeepe_toes_mean = np.mean(mpeepe_toes)
    mpeepe_toes_std = float(np.std(mpeepe_toes))
    # mean per joint positional error (legs)
    legs = groups["legs"]
    mpjpe_legs = np.linalg.norm(poses_y[:, legs] - poses_pred[:, legs], axis=-1)
    mpjpe_legs_mean = np.mean(mpjpe_legs)
    mpjpe_legs_std = float(np.std(mpjpe_legs))
    # mean per joint velocity error (legs)
    velocities_y = np.zeros_like(poses_y)
    velocities_pred = np.zeros_like(poses_pred)
    velocities_y[1:] = (poses_y[1:] - poses_y[:-1]) * framerate
    velocities_pred[1:] = (poses_pred[1:] - poses_pred[:-1]) * framerate
    mpjve_legs = np.linalg.norm(velocities_y[:, legs] - velocities_pred[:, legs], axis=-1)
    mpjve_legs_mean = np.mean(mpjve_legs)
    mpjve_legs_std = float(np.std(mpjve_legs))
    # mean root positional error
    mrpe = np.linalg.norm(global_poses_y - global_poses_pred, axis=-1)
    mrpe_mean = np.mean(mrpe)
    mrpe_std = float(np.std(mrpe))
    return (
        (mpjpe_mean, mpjpe_std, mpjpe),
        (mpeepe_mean, mpeepe_std, mpeepe),
        (mrpe_mean, mrpe_std, mrpe),
        (mpjpe_legs_mean, mpjpe_legs_std, mpjpe_legs),
        (
            mpjve_legs_mean,
            mpjve_legs_std,
            mpjve_legs,
        ),
        (mpeepe_toes_mean, mpeepe_toes_std, mpeepe_toes),
    )
