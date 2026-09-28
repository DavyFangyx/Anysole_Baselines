import glob
import json
import os.path as osp
import logging
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from lib.utils.depth_utils import depth_to_pointcloud

from icecream import ic
from tqdm import tqdm

log = logging.getLogger(__name__)

# Trim applied at the two ends of every contiguous segment, replacing the
# former cross-modal ``[2:-2]`` positional crop (upstream crops 2 frames at
# each end of a gapless sequence; with fake-gap segments the crop is applied
# per segment, keyed by frame ids, never by array position).
TRIM_END = 2


def _frame_index(frame_ids, query):
    """Position of each query frame id inside a sorted unique array."""
    order = np.searchsorted(frame_ids, query)
    if np.any(order >= len(frame_ids)) or not np.array_equal(
            frame_ids[np.clip(order, 0, len(frame_ids) - 1)], query):
        raise ValueError('joined frame ids do not match the NPZ frame ids')
    return order


def _savgol_per_segment(values, window, poly, segments):
    """savgol_filter applied per contiguous segment (upstream semantics)."""
    filtered = values.copy()
    for start, end in segments:
        length = end - start
        if length < window:
            raise ValueError(
                f'segment [{start},{end}) length {length} < savgol window '
                f'{window}; the adapter should have dropped short segments')
        filtered[start:end] = savgol_filter(
            values[start:end], window, poly, axis=0)
    return filtered


def _trimmed_segments(segments):
    """Index-space segments of the array after per-segment end trimming."""
    lengths = [end - start - 2 * TRIM_END for start, end in segments]
    result = []
    cursor = 0
    for length in lengths:
        result.append([cursor, cursor + length])
        cursor += length
    return result


class Dataset():
    def __init__(self, cfg):
        self.task_cfg = cfg['task']
        self.method_cfg = cfg['method']

        self.input_path_base = self.task_cfg['input_path_base']
        # The adapter-layer join manifest is the single alignment authority:
        # streams are joined on shared frame ids, fake/invalid frames are
        # dropped with reasons, and contiguous segments are explicit.  No
        # stream is zipped to another by array position.
        manifest_path = osp.join(self.input_path_base, 'join_manifest.json')
        with open(manifest_path, encoding='utf-8') as handle:
            self.join_manifest = json.load(handle)
        self.frame_ids = np.asarray(
            self.join_manifest['joined_frames'], dtype=np.int64)
        if len(self.frame_ids) == 0:
            raise ValueError(
                f'empty join for {self.input_path_base}: no frames survive '
                'the frame-id inner join (see join_manifest.json)')
        self.segments = [
            (int(start), int(end))  # [start, end) half-open
            for start, end in self.join_manifest['segments']]

        self.input_path_image = osp.join(self.input_path_base, 'color')
        self.input_path_RTMPose = osp.join(self.input_path_base, 'keypoints')
        self.input_path_npz = self.join_manifest['cliff_npz']

        self.input_path_template_scene_rgbd = self.task_cfg['scene_rgbd']
        self.point_cloud, _ = depth_to_pointcloud(self.input_path_template_scene_rgbd, fx=cfg['task']['focal_length'], fy=cfg['task']['focal_length'], scale=cfg['task']['depth_scale'], cx=int(cfg['task']['image_width']/2), cy=int(cfg['task']['image_height']/2))

        self.pose_filter = self.method_cfg['pose_filter']
        self.kp_filter = self.method_cfg['kp_filter']
        self.kp_score = self.method_cfg['kp_score']

        # Load every stream on the joined frame ids, then apply the
        # per-segment end trim exactly once to all of them (the upstream
        # [2:-2] crop applied to each loader of a gapless sequence trims the
        # same 2 frames twice; here the trim is a single shared mask).
        self.target_halpe_fil, self.target_halpe_score = self.load_2d_keypoints()
        self.pose, self.betas, self.trans_origin = self.load_pose()
        keep = np.zeros(len(self.frame_ids), dtype=bool)
        for start, end in self.segments:
            keep[start + TRIM_END:end - TRIM_END] = True
        self.target_halpe_fil = self.target_halpe_fil[keep]
        self.target_halpe_score = self.target_halpe_score[keep]
        self.pose = self.pose[keep]
        self.betas = self.betas[keep]
        self.trans_origin = self.trans_origin[keep]
        self.frame_ids = self.frame_ids[keep]
        self.segments = _trimmed_segments(self.segments)
        expected = np.asarray(
            self.join_manifest.get('final_frames', []), dtype=np.int64)
        if expected.size and not np.array_equal(self.frame_ids, expected):
            raise ValueError(
                f'post-trim frame ids differ from the adapter final_frames: '
                f'{self.frame_ids[:5]}... vs {expected[:5]}...')

        self.input_path_pred_contact = osp.join(self.input_path_base, 'pred_contact_smpl')
        self.thres_contact = self.task_cfg['thres_contact']
        self.contact, self.vertex_contact = self.load_pred_contact_smpl()

        max_frames = int(self.task_cfg.get('max_frames', 0) or 0)
        if max_frames > 0:
            self.target_halpe_fil = self.target_halpe_fil[:max_frames]
            self.target_halpe_score = self.target_halpe_score[:max_frames]
            self.pose = self.pose[:max_frames]
            self.betas = self.betas[:max_frames]
            self.trans_origin = self.trans_origin[:max_frames]
            self.contact = self.contact[:max_frames]
            self.vertex_contact = self.vertex_contact[:max_frames]
            self.frame_ids = self.frame_ids[:max_frames]

    def load_pred_contact_smpl(self):
        """Load FPP-Net pred_contact_smpl sidecars in joined frame-id order.

        The native upstream contract is kept: the per-vertex contact head
        output is thresholded at 0.5 and split into toe/heel four zones
        (footL_ids/footR_ids first 48 = toe, last 48 = heel).  A zone whose
        contacting-vertex count exceeds ``thres_contact`` gates the 3-D
        ground anchor losses; the zone vertex set is zeroed otherwise.
        """
        contact_flag_all = []
        vertex_contact_all = []
        ic("Loading pred_contact_smpl npy (frame-id join) ...")
        for frame_id in tqdm(self.frame_ids.tolist()):
            path = osp.join(self.input_path_pred_contact, f'{frame_id:06d}.npy')
            payload = np.load(path, allow_pickle=True).item()
            stored = int(payload.get('frame_id', -1))
            if stored != frame_id:
                raise ValueError(
                    f'pred_contact_smpl frame id mismatch: {path} stored '
                    f'{stored} != requested {frame_id}')
            vertex_contact_prob = payload['contact_smpl']['pred']
            vertex_contact = np.zeros_like(vertex_contact_prob)

            for i in range(vertex_contact_prob.shape[1]):
                if vertex_contact_prob[0, i] > 0.5:
                    vertex_contact[0, i] = 1
                if vertex_contact_prob[1, i] > 0.5:
                    vertex_contact[1, i] = 1

            LT_contact = np.sum(vertex_contact[0, :48])
            RT_contact = np.sum(vertex_contact[1, :48])
            LH_contact = np.sum(vertex_contact[0, 48:])
            RH_contact = np.sum(vertex_contact[1, 48:])

            _contact_flag = np.zeros(4)

            if LT_contact > self.thres_contact:
                _contact_flag[0] = 1
            else:
                vertex_contact[0, :48] = np.zeros_like(vertex_contact[0, :48])
            if LH_contact > self.thres_contact:
                _contact_flag[1] = 1
            else:
                vertex_contact[0, 48:] = np.zeros_like(vertex_contact[0, 48:])
            if RT_contact > self.thres_contact:
                _contact_flag[2] = 1
            else:
                vertex_contact[1, :48] = np.zeros_like(vertex_contact[1, :48])
            if RH_contact > self.thres_contact:
                _contact_flag[3] = 1
            else:
                vertex_contact[1, 48:] = np.zeros_like(vertex_contact[1, 48:])

            contact_flag_all.append(_contact_flag)
            vertex_contact_all.append(vertex_contact)

        return np.array(contact_flag_all), np.array(vertex_contact_all)

    def load_pose(self):
        npz = np.load(self.input_path_npz)

        # CLIFF rows are selected by frame id; the NPZ contract guarantees
        # one row per canonical frame (single-person, mask-recomputed bbox).
        npz_frame = np.asarray(npz['frame_id'], dtype=np.int64)
        order = _frame_index(npz_frame, self.frame_ids)

        pred_betas = npz['shape'][order]
        pred_rotvec = npz['pose'][order]
        pred_cam_full = npz['global_t'][order]

        # betas are averaged over this unique-subject sequence (the CLIFF
        # contract contains no bystander rows any more).
        pred_betas = np.mean(pred_betas, axis=0).reshape(1, 10).repeat(pred_betas.shape[0], axis=0)

        if self.pose_filter:
            if pred_rotvec.shape[1] != 72:
                raise ValueError('The shape of pred_rotvec is not 72')
            pred_rotmat_fil = []
            for i in range(0, 72, 3):
                q_origin = R.from_rotvec(pred_rotvec[:, i:i+3]).as_quat()
                if i == 0:
                    # quaternion continuity fix, applied within each
                    # contiguous segment (never across a fake gap)
                    for start, end in self.segments:
                        for j in range(start + 1, end):
                            similarity = np.dot(q_origin[j], q_origin[j-1])
                            if similarity < 0:
                                q_origin[j, 0] = -q_origin[j, 0]
                                q_origin[j, 1] = -q_origin[j, 1]
                                q_origin[j, 2] = -q_origin[j, 2]
                                q_origin[j, 3] = -q_origin[j, 3]
                            similarity = np.dot(q_origin[j], q_origin[j-1])

                            if similarity < 0.5:
                                q_origin[j, 0] = q_origin[j-1, 0]
                                q_origin[j, 1] = q_origin[j-1, 1]
                                q_origin[j, 2] = q_origin[j-1, 2]
                                q_origin[j, 3] = q_origin[j-1, 3]
                for dim in range(4):
                    q_origin[:, dim] = _savgol_per_segment(
                        q_origin[:, dim], 17, 5, self.segments)
                temp = R.from_quat(q_origin).as_matrix()
                pred_rotmat_fil.append(temp)
            pred_rotmat_fil = np.stack(pred_rotmat_fil, axis=1)
            pred_rotmat = pred_rotmat_fil

            for dim in (0, 1, 2):
                window, poly = (17, 5) if dim < 2 else (17, 3)
                pred_cam_full[:, dim] = _savgol_per_segment(
                    pred_cam_full[:, dim], window, poly, self.segments)
        elif self.pose_filter == False:
            pred_rotmat = []
            for i in range(0, 72, 3):
                pred_rotmat.append(R.from_rotvec(pred_rotvec[:, i:i+3]).as_matrix())
            pred_rotmat = np.stack(pred_rotmat, axis=1)

        # flip x-axis (upstream-native)
        pred_rotmat[:, 0] = R.from_rotvec([np.pi, 0, 0]).as_matrix() @ pred_rotmat[:, 0]
        pred_rotmat[:, 0] = R.from_rotvec([0, np.pi, 0]).as_matrix() @ pred_rotmat[:, 0]

        return pred_rotmat, pred_betas, pred_cam_full

    def load_2d_keypoints(self):
        kp_2d_path = self.input_path_RTMPose
        ic("Loading 2d keypoints (frame-id join) ...")
        kp_2d_all = []
        for frame_id in tqdm(self.frame_ids.tolist()):
            path = osp.join(kp_2d_path, f'{frame_id:06d}.npy')
            payload = np.load(path, allow_pickle=True).item()
            stored = int(payload.get('frame_id', -1))
            if stored != frame_id:
                raise ValueError(
                    f'keypoint frame id mismatch: {path} stored '
                    f'{stored} != requested {frame_id}')
            kp_2d_all.append(payload)

        target_halpe = np.array([kp_2d_all[i]['keypoints'] for i in range(len(kp_2d_all))])
        target_halpe_score = np.array([kp_2d_all[i]['keypoint_scores'] for i in range(len(kp_2d_all))])

        if self.kp_filter:
            target_halpe_fil = np.zeros_like(target_halpe)

            # use simple savgol_filter, applied per contiguous segment
            for i in range(target_halpe.shape[1]):
                for dim in (0, 1):
                    target_halpe_fil[:, i, dim] = _savgol_per_segment(
                        target_halpe[:, i, dim], 15, 3, self.segments)

        elif self.kp_filter == False:
            target_halpe_fil = target_halpe

        # No score
        if self.kp_score == False:
            target_halpe_score = np.ones_like(target_halpe_score)

        return target_halpe_fil, target_halpe_score
