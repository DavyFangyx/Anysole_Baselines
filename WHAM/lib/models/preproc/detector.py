from __future__ import annotations

import os
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import scipy.signal as signal
from progress.bar import Bar

from ultralytics import YOLO
from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    get_track_id,
    vis_pose_result,
)

ROOT_DIR = osp.abspath(f"{__file__}/../../../../")
VIT_DIR = osp.join(ROOT_DIR, "third-party/ViTPose")

VIS_THRESH = 0.3
BBOX_CONF = 0.5
TRACKING_THR = 0.1
MINIMUM_FRMAES = 30
MINIMUM_JOINTS = 6

class DetectionModel(object):
    def __init__(self, device, target_bboxes=None):
        
        # ViTPose
        pose_model_cfg = osp.join(VIT_DIR, 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py')
        pose_model_ckpt = osp.join(ROOT_DIR, 'checkpoints', 'vitpose-h-multi-coco.pth')
        self.pose_model = init_pose_model(pose_model_cfg, pose_model_ckpt, device=device.lower())
        
        # When a dataset supplies one validated target bbox per frame, bypass
        # generic multi-person detection entirely. This both enforces the
        # target-human contract and avoids loading YOLO for constrained runs.
        self.target_bboxes = target_bboxes
        if target_bboxes is None:
            bbox_model_ckpt = osp.join(ROOT_DIR, 'checkpoints', 'yolov8x.pt')
            self.bbox_model = YOLO(bbox_model_ckpt)
        else:
            self.bbox_model = None
        
        self.device = device
        self.initialize_tracking()
        
    def initialize_tracking(self, ):
        self.next_id = 0
        self.frame_id = 0
        self.pose_results_last = []
        self.tracking_results = {
            'id': [],
            'frame_id': [],
            'bbox': [],
            'keypoints': []
        }
        
    def xyxy_to_cxcys(self, bbox, s_factor=1.05):
        cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200 * s_factor
        return np.array([[cx, cy, scale]])
        
    def compute_bboxes_from_keypoints(self, s_factor=1.2):
        X = self.tracking_results['keypoints'].copy()
        mask = X[..., -1] > VIS_THRESH

        bbox = np.zeros((len(X), 3))
        for i, (kp, m) in enumerate(zip(X, mask)):
            bb = [kp[m, 0].min(), kp[m, 1].min(),
                  kp[m, 0].max(), kp[m, 1].max()]
            cx, cy = [(bb[2]+bb[0])/2, (bb[3]+bb[1])/2]
            bb_w = bb[2] - bb[0]
            bb_h = bb[3] - bb[1]
            s = np.stack((bb_w, bb_h)).max()
            bb = np.array((cx, cy, s))
            bbox[i] = bb
        
        bbox[:, 2] = bbox[:, 2] * s_factor / 200.0
        self.tracking_results['bbox'] = bbox
    
    def track(self, img, fps, length):
        
        # bbox detection
        if self.target_bboxes is None:
            bboxes = self.bbox_model.predict(
                img, device=self.device, classes=0, conf=BBOX_CONF, save=False, verbose=False
            )[0].boxes.xyxy.detach().cpu().numpy()
        else:
            if self.frame_id not in self.target_bboxes:
                raise ValueError(f'Missing target bbox for video frame {self.frame_id}')
            bboxes = np.asarray([self.target_bboxes[self.frame_id]], dtype=np.float32)
        bboxes = [{'bbox': bbox} for bbox in bboxes]
        
        # keypoints detection
        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_results=bboxes,
            format='xyxy',
            return_heatmap=False,
            outputs=None)
        
        # person identification
        if self.target_bboxes is None:
            pose_results, self.next_id = get_track_id(
                pose_results,
                self.pose_results_last,
                self.next_id,
                use_oks=False,
                tracking_thr=TRACKING_THR,
                use_one_euro=True,
                fps=fps)
        else:
            # The annotation already identifies exactly one intended person.
            # A stable ID prevents OKS noise from fragmenting that target into
            # multiple short tracks.
            for pose_result in pose_results:
                pose_result['track_id'] = 0
            self.next_id = 1
        
        for pose_result in pose_results:
            n_valid = (pose_result['keypoints'][:, -1] > VIS_THRESH).sum()
            if self.target_bboxes is None and n_valid < MINIMUM_JOINTS: continue
            
            _id = pose_result['track_id']
            xyxy = pose_result['bbox']
            bbox = self.xyxy_to_cxcys(xyxy)
            
            self.tracking_results['id'].append(_id)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bbox if self.target_bboxes is None else bbox[0])
            self.tracking_results['keypoints'].append(pose_result['keypoints'])
        
        self.frame_id += 1
        self.pose_results_last = pose_results
    
    def process(self, fps):
        for key in ['id', 'frame_id', 'keypoints', 'bbox']:
            self.tracking_results[key] = np.array(self.tracking_results[key])
        if self.target_bboxes is None:
            self.compute_bboxes_from_keypoints()
        else:
            expected_frames = np.arange(self.frame_id)
            if (
                not np.array_equal(self.tracking_results['frame_id'], expected_frames)
                or not np.all(self.tracking_results['id'] == 0)
            ):
                raise RuntimeError(
                    'Target-bbox inference did not produce exactly one stable pose per video frame'
                )
            
        output = defaultdict(lambda: defaultdict(list))
        ids = np.unique(self.tracking_results['id'])
        for _id in ids:
            idxs = np.where(self.tracking_results['id'] == _id)[0]
            for key, val in self.tracking_results.items():
                if key == 'id': continue
                output[_id][key] = val[idxs]
        
        # Smooth bounding box detection
        ids = list(output.keys())
        for _id in ids:
            if len(output[_id]['bbox']) < MINIMUM_FRMAES:
                del output[_id]
                continue
            
            kernel = int(int(fps/2) / 2) * 2 + 1
            smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in output[_id]['bbox'].T]).T
            output[_id]['bbox'] = smoothed_bbox
        
        return output
