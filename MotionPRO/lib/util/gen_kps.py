import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import smplx
import torch
from lib.util.io import findAllFilesWithSpecifiedName, load_smpl_npy
from lib.util.workspace import WORKSPACE_ROOT, resolve_path, sequence_root

DEFAULT_CAM_ID = 3


def generate_kps(gt_files, skip_existing=False):
    device = torch.device('cpu')
    dtype = torch.float32
    smpl = smplx.create(WORKSPACE_ROOT / 'assets/third_party/smpl/SMPL_NEUTRAL.pkl').to(device)
    for gt_file in gt_files:
        output_file_kps = gt_file.replace('smpl.npy', 'keypoints.npy')
        if skip_existing and os.path.isfile(output_file_kps):
            continue
        smpl_gt = load_smpl_npy(gt_file)
        betas = torch.tensor(smpl_gt['betas'][:10], dtype=dtype, device=device).unsqueeze(0)
        body_pose = torch.tensor(smpl_gt['body_pose'], dtype=dtype, device=device).reshape(-1, 23, 3)
        global_orient = torch.tensor(smpl_gt['global_orient'], dtype=dtype,device=device).unsqueeze(1)
        transl = torch.tensor(smpl_gt['transl'], dtype=dtype, device=device)
        result = smpl(betas=betas,
                    body_pose=body_pose,
                    global_orient=global_orient,
                    transl=transl)
        keypoints = result.joints[:, :24].cpu().numpy()
        print(keypoints.shape)
        print(output_file_kps)
        np.save(output_file_kps, keypoints)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate keypoints.npy from smpl.npy.")
    parser.add_argument("--cam-id", type=int, default=DEFAULT_CAM_ID)
    parser.add_argument("--seq-root", type=str, default=None, help="Override the centralized sequence root.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-existing", action="store_true", help="Skip sessions with keypoints.npy.")
    mode.add_argument("--force", action="store_true", help="Recompute and overwrite keypoints.npy.")
    return parser.parse_args()


def resolve_seq_root(args):
    if args.seq_root:
        return resolve_path(args.seq_root, REPO_ROOT)
    return str(sequence_root(args.cam_id))


if __name__ == '__main__':
    args = parse_args()
    seq_root = resolve_seq_root(args)
    gt_files = findAllFilesWithSpecifiedName(seq_root, 'smpl.npy')
    if not gt_files:
        raise FileNotFoundError(f"No smpl.npy under {seq_root}")
    gt_files.sort()
    for gt_file in gt_files:
        print(gt_file)
        generate_kps([gt_file], args.skip_existing)
