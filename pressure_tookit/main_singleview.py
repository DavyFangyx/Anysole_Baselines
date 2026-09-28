import os
# for cluster rendering
import os.path as osp
import sys
import time
import torch
import yaml
try:
    from icecream import ic
except ImportError:  # keep --help/entrypoint validation usable in lean envs
    def ic(value):
        return value

from lib.config.config import parse_config
from lib.core.camera import create_camera
from lib.core.fit_single_frame import fit_single_frame
from lib.core.smpl_mmvp import SMPL_MMVP
from lib.dataextra.data_loader import create_dataset
from lib.utils.workspace import WORKSPACE_ROOT, resolve_path

WORK_ROOT = WORKSPACE_ROOT / "work" / "pressure_toolkit"


def _under_work_root(folder: str) -> bool:
    """过程产物必须写入 work://pressure_toolkit/，不得进入 results/。"""
    folder = os.path.abspath(os.path.expanduser(folder))
    root = os.path.abspath(str(WORK_ROOT))
    return folder == root or folder.startswith(root + os.sep)


def main(**args):
    start = time.time()

    # init common param
    demo_basdir = resolve_path(args.pop('basdir'))
    demo_dsname = args.pop('dataset')
    demo_subids = args.pop('sub_ids')
    demo_seqname = args.pop('seq_name')
    demo_essential_root = resolve_path(args.pop('essential_root'))
    demo_init_root = resolve_path(args.pop('init_data_dir'))
    # fitting stage
    stage = args.pop('fitting_stage')
    save_corr_debug = args.pop('save_corr_debug', False)
    no_export_obj = args.pop('no_export_obj', False)
    skip_mesh_export = args.pop('skip_mesh_export', False)
    skip_gt_depth_export = args.pop('skip_gt_depth_export', False)
    skip_mesh_export = no_export_obj or skip_mesh_export
    skip_gt_depth_export = no_export_obj or skip_gt_depth_export
    maxiters = args.pop('maxiters')
    depth_approx_stride = args.pop('depth_approx_stride', 1)
    if depth_approx_stride > 1:
        print('WARNING: depth_approx_stride>1 is an explicit approximation '
              'mode; its outputs are not the formal baseline',
              file=sys.stderr)
    # frame range
    start_idx = args.pop('start_idx')
    end_idx = args.pop('end_idx')

    # create output folders
    output_folder = resolve_path(args.pop('output_dir'))
    # 过程产物只允许落在 work://pressure_toolkit/ 下
    if not _under_work_root(output_folder):
        raise SystemExit(
            f'output_dir {output_folder} is outside work://pressure_toolkit; '
            'fitting process artifacts must live in the work tree, formal '
            'results are exported separately')
    if not osp.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for f in ['results', 'meshes', 'yamls']:
            os.makedirs(osp.join(output_folder, f), exist_ok=True)
            print(osp.join(output_folder, f))

    # create subdir to write output
    for f in ['results', 'meshes', 'yamls', 'temp', 'gt_depths']:
        curr_output_folder = osp.join(output_folder, f, demo_dsname,
                                      demo_subids, demo_seqname)
        os.makedirs(curr_output_folder, exist_ok=True)
        print(f'{f} will be saved in {curr_output_folder}')

    # save arguments of current experiment
    conf_fn = osp.join(output_folder, 'yamls', f'{demo_dsname}',
                       f'{demo_subids}', f'{demo_seqname}',
                       f'{stage}_conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)

    # get device and set dtype
    dtype = torch.float32
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create Dataset from folders
    dataset_obj = create_dataset(
        basdir=demo_basdir,
        dataset_name=demo_dsname,
        sub_ids=demo_subids,
        seq_name=demo_seqname,
        init_root=demo_init_root,
        start_img_idx=start_idx,
        end_img_idx=end_idx,
        stage=stage)

    # read gender and select model
    body_model = SMPL_MMVP(
        essential_root=demo_essential_root,
        gender=args.pop('model_gender'),
        stage=stage,
        dtype=dtype).to(device)

    # Create the camera object
    rgbd_cam = create_camera(
        basdir=demo_basdir,
        dataset_name=demo_dsname,
        sub_ids=demo_subids,
        seq_name=demo_seqname,
    )
    rgbd_cam.to(device=device)

    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights()\
        .to(device=device, dtype=dtype)

    depth_size, color_size = args.pop('depth_size'), args.pop('color_size')
    reuse_session_resources = args.pop('reuse_session_resources', False)
    resource_cache = {} if reuse_session_resources else None

    for idx in range(len(dataset_obj)):
        # whole tracking
        if idx > 0 and stage == 'init_pose':
            # switch to tracking automatically
            stage = 'tracking'
            dataset_obj.stage = 'tracking'

        # read data
        data = dataset_obj[idx]
        rgbd_path = data['root_path']
        frame_id = data['frame_id']
        print(f'Processing: {rgbd_path}, frame id: {frame_id}, {stage}')

        # fundamental data
        # The active loss uses 2-D keypoints, depth and pressure contacts. It
        # does not consume the RGB image; keep the field for API compatibility.
        img = data['img']
        depth_mask = data['depth_mask']
        keypoints = data['kp']
        depth_map = data['depth_map']

        # optional data
        contact_label = data['contact_label']
        pre_contact_label = data['pre_contact_label']
        init_pose = data['init_pose']
        init_betas = data['init_betas']
        init_scale = data['init_scale']
        init_global_rot = data['init_global_rot']
        init_transl = data['init_transl']

        # prepare output path（以 shared frame id 命名，替代旧全局序号）
        curr_mesh_fn = None if skip_mesh_export else osp.join(
            output_folder, 'meshes', demo_dsname, demo_subids, demo_seqname,
            f'smpl_{frame_id:06d}.obj')
        curr_result_fn = osp.join(output_folder, 'results', demo_dsname,
                                  demo_subids, demo_seqname,
                                  f'smpl_{frame_id:06d}.npz')
        curr_shape_fn = None if stage != 'init_shape' else osp.join(
            output_folder, 'results', demo_dsname, demo_subids,
            f'init_shape_{demo_subids}.npz')
        curr_temp_fn = None if stage == 'init_shape' else osp.join(
            output_folder, 'temp', demo_dsname, demo_subids, demo_seqname,
            f'init_pose_{demo_subids}.npz')
        curr_gt_depths_fn = None if skip_gt_depth_export else osp.join(
            output_folder, 'gt_depths', demo_dsname, demo_subids, demo_seqname,
            f'depth_{frame_id:06d}.obj')
        fit_single_frame(
            img=img,
            depth_mask=depth_mask,
            keypoints=keypoints,
            depth_map=depth_map,
            contact_label=contact_label,
            pre_contact_label=pre_contact_label,
            init_pose=init_pose,
            init_shape=init_betas,
            init_scale=init_scale,
            init_global_rot=init_global_rot,
            init_transl=init_transl,
            essential_root=demo_essential_root,
            body_model=body_model,
            camera=rgbd_cam,
            depth_size=depth_size,
            color_size=color_size,
            joint_weights=joint_weights,
            joint_mapper=dataset_obj.joint_mapper,
            stage=stage,
            output_mesh_fn=curr_mesh_fn,
            output_shape_fn=curr_shape_fn,
            output_result_fn=curr_result_fn,
            output_temp_fn=curr_temp_fn,
            output_gt_depth_fn=curr_gt_depths_fn,
            save_corr_debug=save_corr_debug,
            maxiters=maxiters,
            depth_approx_stride=depth_approx_stride,
            resource_cache=resource_cache,
        )

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))


if __name__ == '__main__':
    args = parse_config()
    ic(args)
    main(**args)
