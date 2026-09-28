import cv2
import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset


def create_dataset(
        basdir,
        dataset_name=None,
        sub_ids=None,
        seq_name=None,
        start_img_idx=0,
        end_img_idx=-1,
        init_root=None,
        stage='init_shape',  # init_pose, tracking
):

    return Pressure_Dataset(
        basdir,
        dataset_name,
        sub_ids,
        seq_name,
        start_img_idx=start_img_idx,
        end_img_idx=end_img_idx,
        init_root=init_root,
        stage=stage)


class ContractError(ValueError):
    """输入契约错误：缺失/歧义数据，不猜测、不回绕。"""


def read_rtm_kpts(keypoint_fn):
    data = np.load(keypoint_fn, allow_pickle=True)
    # HALPE-26 单受试者契约：dict {'keypoints': (26,2), 'keypoint_scores': (26,)}。
    # 不再保留旧的多人数组 [0] 回退——多人行静默取第一行是 P5 禁止的行为。
    if data.dtype == object and data.shape == ():
        payload = data.item()
    else:
        raise ContractError(
            f'{keypoint_fn}: expected a single-subject keypoint dict, '
            f'got array of shape {data.shape}; multi-person [0] fallback removed')
    if 'keypoints' not in payload or 'keypoint_scores' not in payload:
        raise ContractError(
            f'{keypoint_fn}: missing keypoints/keypoint_scores keys')

    points = np.array(payload['keypoints'])
    points_score = np.array(payload['keypoint_scores'])
    points_score = points_score[:, np.newaxis]
    target_keypoints = np.concatenate([points, points_score], axis=1)

    return target_keypoints


def load_contact(contact_fn):
    insole_data = dict(np.load(contact_fn, allow_pickle=True).item())
    region_l, region_r = insole_data['insole'][0], insole_data['insole'][1]
    contact_label = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    row_range = [range(0, 8), range(8, 15), range(15, 23), range(23, 31)]
    # left foot
    for row in range(region_l.shape[0]):
        for col in range(region_l.shape[1]):
            if region_l[row][col] != 0:
                # 0, 1
                if row in row_range[0]:
                    if col in range(0, 7):
                        contact_label[0][1] = 1
                    if col in range(7, 11):
                        contact_label[0][0] = 1
                # 2, 3, 4
                if row in row_range[1]:
                    if col in range(0, 4):
                        contact_label[0][4] = 1
                    if col in range(4, 8):
                        contact_label[0][3] = 1
                    if col in range(8, 11):
                        contact_label[0][2] = 1
                # 5, 6
                if row in row_range[2]:
                    if col in range(0, 4):
                        contact_label[0][6] = 1
                    if col in range(4, 11):
                        contact_label[0][5] = 1
                # 7, 8
                if row in row_range[3]:
                    if col in range(0, 4):
                        contact_label[0][8] = 1
                    if col in range(4, 11):
                        contact_label[0][7] = 1
    # right foot
    for row in range(region_r.shape[0]):
        for col in range(region_r.shape[1]):
            if region_r[row][col] != 0:
                # 0, 1
                if row in row_range[0]:
                    if col in range(0, 4):
                        contact_label[1][0] = 1
                    if col in range(4, 11):
                        contact_label[1][1] = 1
                # 2, 3, 4
                if row in row_range[1]:
                    if col in range(0, 4):
                        contact_label[1][2] = 1
                    if col in range(4, 8):
                        contact_label[1][3] = 1
                    if col in range(8, 11):
                        contact_label[1][4] = 1
                # 5, 6
                if row in row_range[2]:
                    if col in range(0, 8):
                        contact_label[1][5] = 1
                    if col in range(8, 11):
                        contact_label[1][6] = 1
                # 7, 8
                if row in row_range[3]:
                    if col in range(0, 8):
                        contact_label[1][7] = 1
                    if col in range(8, 11):
                        contact_label[1][8] = 1
    return contact_label


def load_init_pose(data_fn, form='cliff', frame_id=None):
    """加载单受试者 CLIFF 初始姿态。

    P5 契约：pose 每行对应一个 canonical frame，按 frame id 精确取行；
    不得从旧多人 NPZ 默认取第一行。
    """
    init_global_rot = None
    init_transl = None

    if form == 'cliff':
        init_data = dict(np.load(data_fn).items())

        if 'pose' not in init_data:
            raise ContractError(f'{data_fn}: missing pose key')
        pose = np.asarray(init_data['pose'])
        if pose.ndim != 2 or pose.shape[1] != 72:
            raise ContractError(
                f'{data_fn}: expected pose (T,72), got {pose.shape}')
        if frame_id is None:
            if pose.shape[0] != 1:
                raise ContractError(
                    f'{data_fn}: pose has {pose.shape[0]} rows but no frame '
                    'id was given; refusing to pick row 0 of a multi-row '
                    'single-subject archive')
            init_pose = pose
        else:
            ids = init_data.get('frame_id')
            if ids is None:
                raise ContractError(
                    f'{data_fn}: no frame_id key; cannot select the row for '
                    f'frame {frame_id}')
            ids = np.asarray(ids).reshape(-1)
            if ids.shape[0] != pose.shape[0]:
                raise ContractError(
                    f'{data_fn}: frame_id length {ids.shape[0]} != pose rows '
                    f'{pose.shape[0]}')
            matches = np.flatnonzero(ids == frame_id)
            if matches.size != 1:
                raise ContractError(
                    f'{data_fn}: expected exactly one row for frame '
                    f'{frame_id}, found {matches.size}')
            init_pose = pose[matches]
        return init_pose[:, 3:], init_global_rot, init_transl
    elif form == 'tracking':
        init_data = dict(np.load(data_fn).items())
        init_pose = init_data['body_pose']
        init_global_rot = init_data['global_rot']
        init_transl = init_data['transl']
        return init_pose, init_global_rot, init_transl
    else:
        print('unsupported format when load init pose')
        raise TypeError


def load_init_shape(data_fn):
    shape_param = dict(np.load(data_fn).items())
    return shape_param['shape'], shape_param['model_scale_opt']


class Pressure_Dataset(Dataset):

    def __init__(
            self,
            basdir,
            dataset_name,
            sub_ids,
            seq_name,
            start_img_idx=0,
            end_img_idx=-1,
            init_root=None,
            dtype=torch.float32,
            stage='init_shape',  # init_pose, tracking
    ):
        super(Pressure_Dataset, self).__init__()

        self.dtype = dtype

        self.basdir = basdir
        self.dataset_name = dataset_name
        self.sub_ids = sub_ids
        self.seq_name = seq_name
        self.init_root = init_root

        self.start_idx = max(0, int(start_img_idx))
        # ``-1`` is the public CLI sentinel for "through the last frame".
        requested_end = int(end_img_idx)
        self.end_idx = requested_end if requested_end >= 0 else 10 ** 9

        self.stage = stage

        self.cnt = 0

        self.rgbd_path = osp.join(basdir, 'images', self.dataset_name,
                                  self.sub_ids, self.seq_name)

        # 拟合帧由 adapter 的 frame_ids.npy 决定：shared frame id，只含
        # valid 且非 fake 的帧。fake/invalid 帧不生成优化任务（P4）。
        frame_ids_path = osp.join(self.rgbd_path, 'frame_ids.npy')
        if not osp.isfile(frame_ids_path):
            raise FileNotFoundError(
                f'missing frame_ids.npy: {frame_ids_path} (run '
                'AnysoleWorkspace/tool/adapters/pressure_toolkit/build_inputs.py)')
        all_frame_ids = np.load(frame_ids_path)
        self.frame_ids = [
            int(x) for x in all_frame_ids
            if int(x) >= self.start_idx and int(x) < self.end_idx
        ]
        if not self.frame_ids:
            raise ContractError(
                f'no fitting frames in [{self.start_idx}, {self.end_idx}) '
                f'for {self.seq_name}')

        # 逐帧路径均以 shared frame id 命名（%06d），排序即 frame id 序
        self.depth_paths = [osp.join(self.rgbd_path, 'depth',
                                     f'{x:06d}.png') for x in self.frame_ids]
        self.dmask_paths = [osp.join(self.rgbd_path, 'depth_mask',
                                     f'{x:06d}.png') for x in self.frame_ids]
        kp_root = osp.join(basdir, 'input', self.sub_ids, self.seq_name,
                           'keypoints')
        self.kp_paths = [osp.join(kp_root, f'{x:06d}.npy')
                         for x in self.frame_ids]

        # 全 session 拟合帧的 insole 映射（含 start_idx 之前的帧，供
        # tracking 首帧找上一有效帧接触；P7 禁止负索引回绕）
        self.insole_path_by_frame = {
            int(x): osp.join(self.rgbd_path, 'insole', f'{int(x):06d}.npy')
            for x in all_frame_ids
        }
        self.session_frame_ids = [int(x) for x in all_frame_ids]
        self.pressure_paths = [
            self.insole_path_by_frame[x] for x in self.frame_ids
        ]

        # init shape data
        if stage == 'init_shape':
            self.shape_path = None
            self.cliff_path = None
        else:
            if not self.init_root:
                raise ValueError('init_root is required for init_pose/tracking')
            shape_path = osp.join(self.init_root, 'fitting', 'results',
                                  self.dataset_name, self.sub_ids,
                                  f'init_shape_{self.sub_ids}.npz')
            if not osp.isfile(shape_path):
                raise FileNotFoundError(
                    f'Missing init_shape for {self.sub_ids}: {shape_path}')
            self.shape_path = shape_path
            cliff_path = osp.join(self.init_root, 'initialization',
                                  self.dataset_name, self.sub_ids,
                                  self.seq_name,
                                  f'{self.seq_name}_cliff_hr48.npz')
            if not osp.isfile(cliff_path):
                raise FileNotFoundError(
                    f'Missing single-subject CLIFF initialization for '
                    f'{self.seq_name}: {cliff_path}')
            self.cliff_path = cliff_path

        # joint mapper
        self.joint_mapper = self.init_joint_mapper()

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(25, dtype=np.float32)
        # These joints are ignored because SMPL has no neck.
        optim_weights[1] = 0
        # put higher weights on knee and elbow joints for mimic'ed poses
        optim_weights[[10, 13]] = 2
        optim_weights[[3, 6, 4, 7]] = 200

        optim_weights[[17, 18]] = 0

        return torch.tensor(optim_weights)

    def init_joint_mapper(self):
        openposemap = np.array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24
        ])[np.newaxis, :]
        halpemap = np.array([
            0, 18, 6, 8, 10, 5, 7, 9, 19, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3,
            20, 22, 24, 21, 23, 25
        ])[np.newaxis, :]
        return np.concatenate([openposemap, halpemap], axis=0).tolist()

    def _previous_frame_id(self, frame_id):
        """tracking 帧的前一有效拟合帧 id；缺失时返回契约错误，不回绕。"""
        position = self.session_frame_ids.index(frame_id)
        if position == 0:
            raise ContractError(
                f'frame {frame_id} has no previous fitting frame in '
                f'{self.seq_name}; refusing negative-index wrap-around '
                '(P7)')
        return self.session_frame_ids[position - 1]

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt >= len(self.frame_ids):
            raise StopIteration

        self.cnt += 1

        return self.read_item(self.cnt - 1)

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        return self.read_item(idx)

    def read_item(self, idx):
        # load data
        frame_id = self.frame_ids[idx]

        # RGB is retained as a None placeholder for API compatibility. The
        # active pressure_toolkit loss consumes keypoints/depth/contact only.
        img = None
        # depth
        depth_path = self.depth_paths[idx]
        depth_map = cv2.imread(depth_path, -1).astype(np.float32) / 1000.
        # depth_mask（adapter 已做 human ∩ finite ∩ [0.4m,5m] 交集；
        # 此处的 3×3 膨胀是上游原生行为，作用于交集后的人体 mask）
        dmask_path = self.dmask_paths[idx]
        mask_ori = cv2.imread(dmask_path)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask_ori, kernel, 1)
        dmask = np.mean(mask, axis=-1)
        # keypoints
        kp_path = self.kp_paths[idx]
        keypoints = read_rtm_kpts(kp_path)
        frame_kp = torch.from_numpy(keypoints).float()
        # insole pressure
        if self.stage == 'tracking':
            pressure_path = self.pressure_paths[idx]
            contact_label = load_contact(pressure_path)
        elif self.stage == 'init_shape':
            # contact_label = None
            # we assume that people should keep A-pose when init shape
            contact_label = np.ones((2, 9)).tolist()
        else:  # init pose
            pressure_path = self.pressure_paths[idx]
            contact_label = load_contact(pressure_path)
            # optionally
            # contact_label = np.ones_like(np.array(contact_label)).tolist()

        # temp insole pressure
        if self.stage == 'tracking':
            # P7：前帧接触在进入 optimizer 前验证；缺失时契约错误，不回绕
            previous_frame = self._previous_frame_id(frame_id)
            pre_pressure_path = self.insole_path_by_frame.get(previous_frame)
            if not pre_pressure_path or not osp.isfile(pre_pressure_path):
                raise ContractError(
                    f'previous-frame contact for frame {frame_id} is missing '
                    f'(expected insole of frame {previous_frame}); cannot '
                    'enter the optimizer (P7)')
            pre_contact_label = load_contact(pre_pressure_path)
        else:
            pre_contact_label = None

        # load initial pose data
        # init pose data
        # TODO: combine different format data
        if self.stage == 'init_shape':
            init_pose, init_betas, init_scale, init_global_rot, init_transl = \
                None, None, None, None, None
        if self.stage == 'init_pose':
            # P5：按 shared frame id 选择单受试者 CLIFF 行
            init_pose, init_global_rot, init_transl = load_init_pose(
                self.cliff_path, form='cliff', frame_id=frame_id)
            init_betas, init_scale = load_init_shape(self.shape_path)
        if self.stage == 'tracking':
            previous_frame = self._previous_frame_id(frame_id)
            init_path = osp.join(self.init_root, 'fitting', 'results',
                                 self.dataset_name, self.sub_ids,
                                 self.seq_name,
                                 f'smpl_{previous_frame:06d}.npz')
            if not osp.isfile(init_path):
                raise FileNotFoundError(
                    f'Missing tracking init for frame {frame_id} '
                    f'(previous fitting frame {previous_frame}): {init_path}')
            init_pose, init_global_rot, init_transl = load_init_pose(
                init_path, form='tracking')
            init_betas, init_scale = load_init_shape(self.shape_path)

        output_dict = {
            'root_path': self.rgbd_path,
            'frame_id': frame_id,
            'depth_map': depth_map,
            'img': img,
            'depth_mask': dmask,
            'kp': frame_kp,
            'contact_label': contact_label,
            'pre_contact_label': pre_contact_label,
            'init_pose': init_pose,
            'init_betas': init_betas,
            'init_scale': init_scale,
            'init_global_rot': init_global_rot,
            'init_transl': init_transl
        }

        return output_dict
