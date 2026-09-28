from __future__ import annotations
import prefer_env_site  # noqa: F401  # must run before pymotion imports
import json
import os
import random
import torch
import numpy as np
import warnings
from typing import Optional
from argparse import ArgumentParser
from config import Config, load_config
from dataset import MotionDataset
from backward_diffusion import ControlTransformer, PriorTransformer, TransformerTranslation, model_from_config
from forward_diffusion import ForwardDiffusion
from losses import PoseLoss, TranslationLoss
from metrics import get_metrics, test
from normalizer import Normalizer
from utils import skeleton_pos_to_rot
from visualize_compare import DEFAULT_VIZ_DIR, visualize_from_exports
from bvh_export import joint_names_for_count, write_bvh
from workspace import DISPLAY_ROOT, RESULTS_ROOT, normalizer_path, resolve_path


def pred_to_bvh(
    data: torch.Tensor,
    clip: int,
    dataset: MotionDataset,
    normalizer: Normalizer,
    out_path: str,
):
    """
    data is the ground truth tensor of shape (frames, output_dim)
    """
    data_np: np.ndarray = normalizer.denormalize_poses(data).cpu().numpy()
    displacements = data_np[:, :3]
    global_pos = np.cumsum(displacements, axis=0)
    poses = data_np[:, 3:].reshape(data_np.shape[0], -1, 3)
    poses = np.concatenate([np.zeros((data_np.shape[0], 1, 3)), poses], axis=1)
    parents = dataset.parents.cpu().numpy()
    offsets = dataset.offsets[clip].cpu().numpy()
    rots = skeleton_pos_to_rot(poses, parents, offsets)
    joint_names = list(getattr(dataset, "joint_names", joint_names_for_count(len(parents))))
    write_bvh(out_path, joint_names, parents, offsets, rots, global_pos, dataset.delta_time)


def raw_bvh_info(dataset: MotionDataset, clip_name: str) -> tuple[Optional[str], Optional[float]]:
    paths = getattr(dataset, "raw_bvh_paths", None)
    starts = getattr(dataset, "raw_start_times", None)
    if paths:
        return str(paths[0]), float(starts[0]) if starts else 0.0
    # Existing gait datasets predate raw metadata. The visualizer can resolve
    # their align_meta.json from the session id.
    if clip_name.startswith("S"):
        return None, None
    raise ValueError(
        "Dataset has no raw BVH metadata. Re-run its preprocessing script before testing."
    )


def cs_to_json(cs_in: torch.Tensor, dataset: MotionDataset, normalizer: Normalizer, out_path: str) -> None:
    """
    cs is the condition tensor of shape (frames, input_dim)
    """
    cs: np.ndarray = normalizer.denormalize_insole(cs_in).cpu().numpy()
    l_pressures = cs[:, slice(*dataset.l_pressure_idx)]
    l_total_force = cs[:, slice(*dataset.l_total_force_idx)]
    l_center_of_pressure = cs[:, slice(*dataset.l_center_of_pressure_idx)]
    r_pressures = cs[:, slice(*dataset.r_pressure_idx)]
    r_total_force = cs[:, slice(*dataset.r_total_force_idx)]
    r_center_of_pressure = cs[:, slice(*dataset.r_center_of_pressure_idx)]
    data = {
        "l_pressures": l_pressures.reshape(-1).tolist(),
        "l_total_force": l_total_force.reshape(-1).tolist(),
        "l_center_of_pressure": l_center_of_pressure.reshape(-1).tolist(),
        "r_pressures": r_pressures.reshape(-1).tolist(),
        "r_total_force": r_total_force.reshape(-1).tolist(),
        "r_center_of_pressure": r_center_of_pressure.reshape(-1).tolist(),
    }
    # --no-imu datasets have no acc/gyro channels (idx fields are None).
    if dataset.l_acceleration_idx is not None:
        l_accelerations = cs[:, slice(*dataset.l_acceleration_idx)]
        l_angular_accelerations = cs[:, slice(*dataset.l_angular_velocity_idx)]
        r_accelerations = cs[:, slice(*dataset.r_acceleration_idx)]
        r_angular_accelerations = cs[:, slice(*dataset.r_angular_velocity_idx)]
        data["l_accelerations"] = l_accelerations.reshape(-1).tolist()
        data["l_angular_accelerations"] = l_angular_accelerations.reshape(-1).tolist()
        data["r_accelerations"] = r_accelerations.reshape(-1).tolist()
        data["r_angular_accelerations"] = r_angular_accelerations.reshape(-1).tolist()
    with open(out_path, "w") as f:
        json.dump(data, f)


def attn_mat_to_json(
    attn_mats: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]], out_path: str
) -> None:
    """
    attn_mats is a tuple of 3 list of attention matrices of shape (n_heads, T//2 (left leg or right leg or body), T//2 (insole))
    """
    assert len(attn_mats) > 0, "No attention matrices to save"
    num_heads = attn_mats[0][0].shape[0]
    heads_left_leg = []
    heads_right_leg = []
    heads_body = []
    for head in range(num_heads):
        left_leg = []
        right_leg = []
        body = []
        for attn_mat in attn_mats[0]:
            left_leg.append(attn_mat[head].cpu().numpy())
        for attn_mat in attn_mats[1]:
            right_leg.append(attn_mat[head].cpu().numpy())
        for attn_mat in attn_mats[2]:
            body.append(attn_mat[head].cpu().numpy())
        heads_left_leg.append(np.concatenate(left_leg, axis=0))
        heads_right_leg.append(np.concatenate(right_leg, axis=0))
        heads_body.append(np.concatenate(body, axis=0))

    data = {}
    for i in range(num_heads):
        data[f"head_{i}_left_leg"] = heads_left_leg[i].flatten().tolist()
        data[f"head_{i}_right_leg"] = heads_right_leg[i].flatten().tolist()
        data[f"head_{i}_body"] = heads_body[i].flatten().tolist()

    with open(out_path, "w") as f:
        json.dump(data, f)


def clip_output_name(dataset: MotionDataset, dataset_path: str, clip: int) -> str:
    name = dataset.clip_name(clip)
    if name.startswith("S"):
        return name
    return os.path.basename(dataset_path).split(".")[0] + "_" + name


@torch.no_grad()
def run_clip(
    config: Config,
    dataset_path: str,
    dataset: MotionDataset,
    clip: int,
    normalizer: Normalizer,
    forward_diffusion: ForwardDiffusion,
    bw_diff_pose_prior: PriorTransformer,
    bw_diff_pose,
    model_trans,
    w: float,
    seed: Optional[int],
    vertical_zero_threshold: bool,
    visualize: bool = True,
) -> dict:
    clip_dataset = dataset.isolate_clip(clip)
    clip_dataset.set_temporality(config["input_T"])
    clip_dataset.set_stride(1)
    name = clip_output_name(dataset, dataset_path, clip)
    print(f"Clip {clip}/{dataset.n_clips() - 1}: {name}")
    if bw_diff_pose is not None:
        assert isinstance(bw_diff_pose, ControlTransformer)
    assert isinstance(bw_diff_pose_prior, PriorTransformer)
    if model_trans is not None:
        assert isinstance(model_trans, TransformerTranslation)
    device = next(bw_diff_pose_prior.parameters()).device
    loss_fn = PoseLoss(device)
    loss_fn_trans = TranslationLoss()
    loss_pose, loss_trans, gt, pred, cs, _ = test(
        clip_dataset,
        normalizer,
        forward_diffusion,
        bw_diff_pose,
        bw_diff_pose_prior,
        model_trans,
        loss_fn,
        loss_fn_trans,
        config,
        w,
        device,
        False,
        vertical_zero_threshold,
    )
    print(f"Test Loss Pose: {loss_pose:.5f} - Test Loss Translation: {loss_trans:.5f}")
    mpjpe, mpeepe, mrpe, mpjpe_legs, mpjve_legs, mpeepe_legs = get_metrics(
        normalizer, gt, pred, clip_dataset.sample_rate
    )
    if model_trans is None:
        # --no-imu ablation: translation is the GT fallback, so the
        # translation loss and root-position error carry no information.
        loss_trans = None
        mrpe = (None, None, np.full(mrpe[2].shape, np.nan))

    def print_error(label: str, error: tuple[float, float, np.ndarray]) -> None:
        if error[0] is None:
            print(f"{label}: N/A (no translation model)")
        else:
            print(f"{label}: {error[0]:.5f} (std: {error[1]:.5f})")

    print_error("MPJPE", mpjpe)
    print_error("MPEEPE", mpeepe)
    print_error("MRPE", mrpe)
    print_error("MPJPE Legs", mpjpe_legs)
    print_error("MPJVE Legs", mpjve_legs)
    print_error("MPEEPE Legs", mpeepe_legs)

    print("Writing predictions -----------------")
    output_dir = config.get(
        "predictions_dir",
        str(RESULTS_ROOT / "baselines/Step2Motion/predictions" / os.path.basename(config["model_dir"])),
    )
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, name + "_gen.bvh")
    if seed is not None:
        pred_path = pred_path.replace(".bvh", f"_s{seed}.bvh")
    pred_to_bvh(pred, 0, clip_dataset, normalizer, pred_path)
    cs_path = os.path.join(output_dir, name + "_cs.json")
    cs_to_json(cs, clip_dataset, normalizer, cs_path)
    raw_path, raw_start = raw_bvh_info(clip_dataset, name)
    export_meta_path = os.path.join(output_dir, name + "_meta.json")
    with open(export_meta_path, "w") as handle:
        json.dump(
            {
                "raw_bvh_path": raw_path,
                "raw_start_time": raw_start,
                "fps": float(clip_dataset.sample_rate),
            },
            handle,
            indent=2,
        )
    np.savez(
        os.path.join(output_dir, name + "_stats.npz"),
        mpjpe=mpjpe[2],
        mpeepe=mpeepe[2],
        mrpe=mrpe[2],
        mpjpe_legs=mpjpe_legs[2],
        mpjve_legs=mpjve_legs[2],
        mpeepe_legs=mpeepe_legs[2],
    )
    print("Visualizing predictions -----------------")
    if visualize:
        visualize_from_exports(
            pred_path,
            raw_path,
            cs_path,
            config.get("visualizations_dir", str(DEFAULT_VIZ_DIR)),
            clip_id=name,
            fps=clip_dataset.sample_rate,
            raw_start_time=raw_start,
        )
    else:
        print("skipped (--no-visualize)")
    return {
        "clip": clip,
        "name": name,
        "n_frames": int(gt.shape[0]),
        "loss_pose": float(loss_pose),
        "loss_trans": None if loss_trans is None else float(loss_trans),
        "mpjpe": float(mpjpe[0]),
        "mpeepe": float(mpeepe[0]),
        "mrpe": None if mrpe[0] is None else float(mrpe[0]),
        "mpjpe_legs": float(mpjpe_legs[0]),
        "mpjve_legs": float(mpjve_legs[0]),
        "mpeepe_legs": float(mpeepe_legs[0]),
    }


@torch.no_grad()
def main(
    config: Config,
    dataset_path: str,
    verbose: bool,
    w: float,
    clip: Optional[int],
    seed: Optional[int],
    vertical_zero_threshold: bool,
    visualize: bool = True,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    normalizer: Normalizer = torch.load(
        normalizer_path(config),
        weights_only=False,
        map_location=device,
    )
    normalizer.to(device)

    print("Setting up models -----------------")
    forward_diffusion = ForwardDiffusion(T=config["diffusion_T"]).to(device)
    bw_diff_pose_prior, _ = model_from_config(config, is_prior=True)
    bw_diff_pose_prior.to(device)
    bw_diff_pose = None
    model_trans = None
    bw_diff_pose, model_trans = model_from_config(config, is_prior=False)
    bw_diff_pose.to(device)
    if model_trans is not None:
        model_trans.to(device)

    bw_prior_name = None
    bw_name = None
    trans_name = None
    bw_prior_name = "model_prior.pth"
    bw_name = "model.pth"
    trans_name = "model_trans.pth"

    if bw_prior_name is not None:
        bw_diff_pose_prior.load_state_dict(torch.load(os.path.join(config["model_dir"], bw_prior_name), map_location=device))
        assert bw_diff_pose_prior.normalizer_id.item() == normalizer.id
    if bw_name is not None and bw_diff_pose is not None:
        bw_diff_pose.load_state_dict(torch.load(os.path.join(config["model_dir"], bw_name), map_location=device))
        assert bw_diff_pose.normalizer_id.item() == normalizer.id
    if trans_name is not None and model_trans is not None:
        model_trans.load_state_dict(torch.load(os.path.join(config["model_dir"], trans_name), map_location=device))
        assert model_trans.normalizer_id.item() == normalizer.id

    print("Setting up datasets -----------------")
    dataset = MotionDataset.load(dataset_path, device)
    n_clips = dataset.n_clips()
    if clip is None:
        clip_ids = list(range(n_clips))
        print(f"Testing all {n_clips} clips")
    else:
        if clip < 0 or clip >= n_clips:
            raise ValueError(f"Clip {clip} is out of range for {n_clips} clips")
        clip_ids = [clip]
        print(f"Testing clip {clip}/{n_clips - 1}: {dataset.clip_name(clip)}")
    rows = []
    for clip_idx in clip_ids:
        rows.append(
            run_clip(
                config,
                dataset_path,
                dataset,
                clip_idx,
                normalizer,
                forward_diffusion,
                bw_diff_pose_prior,
                bw_diff_pose,
                model_trans,
                w,
                seed,
                vertical_zero_threshold,
                visualize,
            )
        )
    if rows:
        output_dir = config.get(
            "predictions_dir",
            str(RESULTS_ROOT / "Step2Motion/predictions" / os.path.basename(config["model_dir"])),
        )
        os.makedirs(output_dir, exist_ok=True)
        keys = ["mpjpe", "mpeepe", "mrpe", "mpjpe_legs", "mpjve_legs", "mpeepe_legs", "loss_pose", "loss_trans"]
        # --no-imu ablation rows carry None for translation-only metrics.
        mean = {
            key: (float(sum(row[key] for row in rows) / len(rows))
                  if all(row[key] is not None for row in rows) else None)
            for key in keys
        }
        summary = {"dataset": dataset_path, "n_clips": len(rows), "mean": mean, "clips": rows}
        summary_path = os.path.join(output_dir, "metrics_summary.json")
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote {summary_path}")
        mrpe_str = "N/A" if mean["mrpe"] is None else "%.5f" % mean["mrpe"]
        print("Mean over %d clips: MPJPE=%.5f MPEEPE=%.5f MRPE=%s" % (len(rows), mean["mpjpe"], mean["mpeepe"], mrpe_str))


if __name__ == "__main__":
    DEFAULT_SEED = 2222

    warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")
    warnings.filterwarnings(
        "ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False"
    )
    warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
    warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention")

    parser = ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Path to the model directory")
    parser.add_argument(
        "--dataset", type=str, default="from_config_file", help="Path of the dataset to use for prediction"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print additional information during the process"
    )
    parser.add_argument(
        "--w",
        type=float,
        default=1.0,
        help="Classifier-Free Guidance weight (default: 1.0), it increases accuracy at the cost of diversity.",
    )
    parser.add_argument(
        "--clip",
        type=int,
        default=None,
        help="Clip index to process. Default: all clips in the dataset.",
    )
    parser.add_argument(
        "--vertical_th",
        action="store_true",
        default=False,
        help="Improve vertical prediction in some cases by usign a threshold to remove noise around zero",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed to use for random number generation (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        default=False,
        help="Skip prediction visualization (smoke runs).",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    args.model_dir = resolve_path(args.model_dir)
    if not os.path.exists(args.model_dir):
        raise ValueError(f"Model directory {args.model_dir} does not exist")

    config_path = os.path.join(args.model_dir, "copy_config.json")
    config = load_config(config_path)

    config["model_dir"] = args.model_dir
    model_name = os.path.basename(os.path.normpath(args.model_dir))
    config["predictions_dir"] = str(RESULTS_ROOT / "baselines/Step2Motion/predictions" / model_name)
    config["visualizations_dir"] = str(DISPLAY_ROOT / "Step2Motion" / model_name)

    dataset_path = args.dataset
    if args.dataset == "from_config_file":
        dataset_path = config["test_data"]
    else:
        dataset_path = resolve_path(dataset_path)

    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset {dataset_path} does not exist")

    main(
        config,
        dataset_path,
        args.verbose,
        args.w,
        args.clip,
        None if args.seed == DEFAULT_SEED else args.seed,
        args.vertical_th,
        visualize=not args.no_visualize,
    )
