import argparse
import glob
import os
import os.path as osp
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm

from lib.util.workspace import RESULTS_ROOT, WORKSPACE_ROOT, resolve_path, sequence_root

DEFAULT_CAM_ID = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Generate bbox.npy for MotionPRO sequences.")
    parser.add_argument("--cam-id", type=int, default=DEFAULT_CAM_ID)
    parser.add_argument("--seq-root", type=str, default=None, help="Override the centralized sequence root.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-existing", action="store_true", help="Skip sessions with bbox.npy.")
    mode.add_argument("--force", action="store_true", help="Recompute and overwrite bbox.npy.")
    return parser.parse_args()


def resolve_seq_root(args):
    if args.seq_root:
        return Path(resolve_path(args.seq_root, REPO_ROOT))
    return sequence_root(args.cam_id)


def list_subject_dirs(seq_root: Path):
    if not seq_root.is_dir():
        raise FileNotFoundError(f"No sequences under {seq_root}")
    subject_dirs = []
    for date_dir in sorted(path for path in seq_root.iterdir() if path.is_dir()):
        for subject_dir in sorted(path for path in date_dir.iterdir() if path.is_dir()):
            subject_dirs.append(subject_dir)
    if not subject_dirs:
        raise FileNotFoundError(f"No sequences under {seq_root}")
    return subject_dirs


def generate_bbox(seq_root: Path, skip_existing: bool = False):
    subject_dirs = list_subject_dirs(seq_root)
    config = str(WORKSPACE_ROOT / "assets/third_party/MotionPRO/mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py")
    checkpoint = str(WORKSPACE_ROOT / "assets/third_party/MotionPRO/mmdetection/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = init_detector(config, checkpoint, device)
    for working_name_dir in tqdm(subject_dirs):
        working_dirs = glob.glob(str(working_name_dir / "*"))
        working_dirs.sort()
        print(working_dirs)
        for working_dir in tqdm(working_dirs):
            if skip_existing and osp.isfile(osp.join(working_dir, "bbox.npy")):
                continue
            color_dir = os.path.join(working_dir, "color")
            detection_all = []
            img_path_list = glob.glob(osp.join(color_dir, "*.jpg"))
            img_path_list.extend(glob.glob(osp.join(color_dir, "*.png")))
            img_path_list.sort()
            print("Loading images ...")
            orig_img_bgr_all = [cv2.imread(img_path) for img_path in tqdm(img_path_list)]
            print("Image number:", len(img_path_list))
            imgs = orig_img_bgr_all
            class_id = 0
            last_x1 = 0
            last_y1 = 0
            last_x2 = 0
            last_y2 = 0
            last_score = 0
            for i, img in enumerate(tqdm(imgs)):
                try:
                    result = inference_detector(model, img)
                    x1, y1, x2, y2 = result.pred_instances.bboxes[class_id].cpu()
                    score = result.pred_instances.scores[class_id].cpu()
                    last_x1, last_y1, last_x2, last_y2, last_score = x1, y1, x2, y2, score
                except Exception:
                    x1, y1, x2, y2, score = last_x1, last_y1, last_x2, last_y2, last_score
                    log_path = RESULTS_ROOT / "baselines/MotionPRO/logs/gen_bbox.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a") as file:
                        file.write("%s use last bbox\n" % i)
                detection_all.append([i, x1, y1, x2, y2, score, 0.99, 0])
            detection_all = np.array(detection_all)
            print(detection_all.shape)
            np.save(os.path.join(working_dir, "bbox.npy"), detection_all)


def main():
    args = parse_args()
    generate_bbox(resolve_seq_root(args), args.skip_existing)


if __name__ == "__main__":
    main()
