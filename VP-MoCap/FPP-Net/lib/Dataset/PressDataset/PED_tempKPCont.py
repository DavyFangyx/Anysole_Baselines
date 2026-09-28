import json
import cv2
import os
import os.path as osp
import numpy as np
import torch
import random
import pickle, glob
import re
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from icecream import ic

from lib.Dataset.ImageModule import ImageModule
from lib.Dataset.InsoleModule import InsoleModule


class ContDataset(Dataset):
    def __init__(self, opt, phase):
        self.phase = phase
        self.basedir = opt.datadir
        self.seq_name = opt.seq_name

        self.insole_module = InsoleModule(
            self.basedir, getattr(opt, 'essentials_root', None))
        self.image_module = ImageModule(opt)
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

        # 31x11 insole: the single public MMVP representation produced by the
        # merged mmvp_series adapter (insole_31x11).  Read-only here; no
        # private per-model copy exists under model_inputs (the FPP adapter
        # tree carries color/keypoints/sub_info only).
        self.tactile_root = Path(getattr(opt, 'tactile_root', ''))
        self._session_frame_ids = {}

        self.w_sc = opt.w_sc
        self.w_nc = opt.w_nc

        self.img_res = opt.img_res
        self.is_aug = opt.aug.is_aug
        self.scale_factor = opt.aug.scale_factor  # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.noise_factor = opt.aug.noise_factor
        self.rot_factor = opt.aug.rot_factor  # Random rotation in the range [-rot_factor, rot_factor]

        self.is_sam = opt.aug.is_sam
        self.img_W = 1280
        self.img_H = 720

        # Subject pressure weight.  The value is computed live by the
        # mmvp_series FPP adapter (build_metadata): the raw 31x11 insole sum
        # over the insole masks at the min-velocity **double-support**
        # standing frame of the session.  Upstream's static table is
        # falsified (61-168x off, non-proportional) and is not used; the
        # per-date files live in the FPP adapter tree (one per recording day,
        # subject keyed by subject id).
        self.sub_info = {}
        for date_dir in sorted(Path(self.basedir).iterdir()):
            info_path = date_dir / 'sub_info.npy'
            if date_dir.is_dir() and info_path.is_file():
                self.sub_info[date_dir.name] = np.load(
                    info_path, allow_pickle=True).item()
        if not self.sub_info:
            raise FileNotFoundError(
                f'No date-level sub_info.npy under {self.basedir}')

        match = re.search(r'temporal(\d+)', str(opt.tv_fn))
        self.seqlen = int(match.group(1)) if match else 5
        self.halfseqlen = int((self.seqlen - 1) / 2)

        self.images_fn = []
        train_val = np.load(opt.tv_fn, allow_pickle=True).item()
        data_ls = train_val[self.phase]

        for data_id in data_ls.keys():
            sub_ls = data_ls[data_id]
            for sub_id in sub_ls.keys():
                seq_ls = sub_ls[sub_id]
                for seq_name in seq_ls.keys():
                    insole_ls = seq_ls[seq_name]
                    images_fn = sorted([os.path.join(data_id, sub_id, seq_name, x[:-4])
                                        for x in insole_ls])
                    self.images_fn += images_fn

        # every split subject must carry a usable weight, otherwise
        # sigmoidNorm silently normalizes with a wrong (or zero) scale
        for data_id in data_ls.keys():
            if data_id not in self.sub_info:
                raise FileNotFoundError(
                    f'split phase "{self.phase}" uses date {data_id} but no '
                    f'sub_info.npy for it exists under {self.basedir}')
            for sub_id in data_ls[data_id].keys():
                if sub_id not in self.sub_info[data_id]:
                    raise KeyError(
                        f'{data_id}: sub_info.npy has no entry for {sub_id} '
                        f'(has {sorted(self.sub_info[data_id])})')
                weight = self.sub_info[data_id][sub_id]['weight']
                if not np.isfinite(weight) or weight <= 0:
                    raise ValueError(
                        f'{data_id}/{sub_id}: subject pressure weight is '
                        f'{weight}, expected a positive finite sum')

        if self.phase == 'train':
            random.shuffle(self.images_fn)

        self.insole_mask = torch.from_numpy(self.insole_module.maskImg)

    def augm_params(self, flip, sc):
        """Get augmentation parameters."""
        # We flip with probability 1/2
        if np.random.uniform() <= 0.5:
            flip = 1

        # The scale is multiplied with a number
        # in the area [1-scaleFactor,1+scaleFactor]
        sc = min(1 + self.scale_factor,
                 max(1 - self.scale_factor, np.random.randn() * self.scale_factor + 1))
        if sc > 1:
            sc = 2 - sc
        # but it is zero with probability 3/5
        # rebg = np.random.uniform()
        return flip, sc

    def __len__(self):
        return len(self.images_fn)

    def check_session_frame_ids(self, data_id, sub_ids, seq_name):
        '''Verify the public representation is addressed by the shared frame id.

        The split tree names a window centre by its canonical frame id and the
        insole (and keypoint) sidecars are looked up with that same id.  The
        public tree resolves the id positionally, so an id map that is not
        ``0..n-1`` would silently shift every frame of the session.
        '''
        key = (data_id, sub_ids, seq_name)
        if key in self._session_frame_ids:
            return self._session_frame_ids[key]
        frame_id_path = self.tactile_root / data_id / sub_ids / seq_name / 'frame_id.npy'
        if not frame_id_path.is_file():
            raise FileNotFoundError(
                f'public MMVP 31x11 session has no frame_id.npy: {frame_id_path}')
        frame_ids = np.load(frame_id_path)
        if not np.array_equal(frame_ids, np.arange(frame_ids.shape[0])):
            raise ValueError(
                f'{frame_id_path}: frame ids are not positional 0..n-1 '
                f'({frame_ids[:5]}...); the insole address would be shifted')
        self._session_frame_ids[key] = int(frame_ids.shape[0])
        return self._session_frame_ids[key]

    def __getitem__(self, index):
        case_name = self.images_fn[index]
        # case_name = 'S01/MoCap_20230422_092324/000' #debug
        data_id, sub_ids, seq_name, frame_ids = case_name.split('/')
        frame_idx = int(frame_ids)
        seq_dir = osp.join(self.basedir, data_id, sub_ids, seq_name)

        # load insole from the single public MMVP 31x11 representation
        frame_count = self.check_session_frame_ids(data_id, sub_ids, seq_name)
        if frame_idx >= frame_count:
            raise IndexError(
                f'{data_id}/{sub_ids}/{seq_name}: frame {frame_idx} outside the '
                f'public representation ({frame_count} frames)')
        insole_path = self.tactile_root / data_id / sub_ids / seq_name \
            / 'insole' / ('%06d.npy' % frame_idx)
        insole_press = np.load(insole_path)
        if insole_press.shape != (2, 31, 11):
            raise ValueError(
                f'unexpected public insole shape: {insole_path} '
                f'{insole_press.shape}, expected (2,31,11)')

        # load insole pressure weight of the subject (adapter-computed)
        sub_weight = self.sub_info[data_id][sub_ids]['weight']

        insole = self.insole_module.sigmoidNorm(insole_press, sub_weight)
        # cv2.imwrite('debug/tmp.png',self.insole_module.showNormalizedInsole(insole))
        insole = np.concatenate([insole[0], insole[1]], axis=1)
        insole = torch.from_numpy(insole).float()

        # calc contact
        # #1 ruling (2026-09-28): the native call site threshold is 0.5 and
        # stays 0.5.  InsoleModule.press2Cont's 0.7 is only the function
        # default; the 0.7 detour that took the default is retired and must
        # not come back through this call site.
        contact_map = self.insole_module.press2Cont(insole_press, sub_weight, th=0.5)
        # cv2.imwrite('debug/cont.png',self.insole_module.showContact(contact_map))

        # Aug
        flip = 0  # flipping
        sc = 1  # scaling
        if self.is_aug and self.phase == 'train':
            flip, sc = self.augm_params(flip, sc)

        kp_ls = []
        for frame_i in range(-self.halfseqlen, self.halfseqlen + 1):
            req_frame = frame_idx + frame_i
            kp_fn = osp.join(seq_dir, 'keypoints/%06d.npy' % req_frame)
            kp_data = np.load(kp_fn, allow_pickle=True).item()
            # frame-id join: every keypoint sidecar carries its canonical
            # frame id; a positional read would silently shift the window
            # whenever the id map is not 0..n-1.
            stored_frame = int(kp_data.get('frame_id', -1))
            if stored_frame != req_frame:
                raise ValueError(
                    f'keypoint frame id mismatch: {kp_fn} stored '
                    f'{stored_frame} != requested {req_frame}')
            keypoints = np.array(kp_data['keypoints'])  # [width,height]
            keypoints[:, 0] = keypoints[:, 0]  # /self.img_W
            keypoints[:, 1] = keypoints[:, 1]  # /self.img_H
            keypoint_scores = np.array(kp_data['keypoint_scores']).reshape([-1, 1])
            kps = np.concatenate([keypoints, keypoint_scores], axis=1)
            kps = torch.from_numpy(kps).float()
            kp_ls.append(kps)

        # normalized keypoints
        kp_temp = torch.stack(kp_ls)
        kp_h = torch.max(kp_temp[..., 0]) - torch.min(kp_temp[..., 0])
        kp_w = torch.max(kp_temp[..., 1]) - torch.min(kp_temp[..., 1])
        kp_center = kp_temp[2, 19, :2]  # root keypoints
        kp_temp[:, :, 0] = kp_temp[:, :, 0] - kp_center[0]
        kp_temp[:, :, 1] = kp_temp[:, :, 1] - kp_center[1]
        kp_temp[..., 0] = kp_temp[..., 0] / kp_h
        kp_temp[..., 1] = kp_temp[..., 1] / kp_w

        if flip and self.phase == 'train':
            # if True:
            kp_temp[:, :, 0] = -kp_temp[:, :, 0]
            insole = torch.flip(insole, dims=[1])
            # contact_map = torch.flip(torch.from_numpy(contact_map).float(), dims=[1])
            contact_map = np.flip(contact_map, axis=1)
            contact_map = contact_map.copy()

        _smpl_cont = self.insole_module.getVertsPress(np.stack([contact_map[:, :11], contact_map[:, 11:]]))
        smpl_cont = np.concatenate([_smpl_cont[0], _smpl_cont[1]])

        contact_map = torch.from_numpy(contact_map).float()
        smpl_cont = torch.from_numpy(smpl_cont).float()

        if self.phase == 'train':
            kp_temp[:, :, :2] = kp_temp[:, :, :2] * sc

        res = {
            'case_name': case_name,
            'frame_id': frame_idx,  # temporal-5 centre frame id
            'keypoints': kp_temp,
            'insole': insole[self.insole_mask],
            'contact_label': contact_map[self.insole_mask],
            'contact_smpl': smpl_cont,
        }
        return res
