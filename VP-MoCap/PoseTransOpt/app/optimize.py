import hydra
import torch
import numpy as np
import copy
import json
import logging
import os
import os.path as osp
from pathlib import Path
from tqdm import tqdm

from lib.dataset.dataset_mmvp import Dataset
from lib.optimize_util import perspective_projection, encode, decode
from lib.loss.losses import OPENPOSE, HALPE, loss_2d, loss_3d_t
from lib.initial_trans.initial_trans import initial_trans
from lib.utils.visualize_utils import Camera, Visualizer
from lib.utils.render_utils import Renderer
from lib.utils.filter_utils import fil_pose, fil_trans
from models.smpl import SMPL
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser


log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_path(value):
    text = str(value or '')
    if '://' in text and text.startswith(('shared://', 'model-input://', 'work://', 'asset://')):
        from AnysoleWorkspace.tool.workspace import resolve_uri
        return str(resolve_uri(text))
    roots = {
        'workspace://': Path(os.environ.get('ANYSOLE_WORKSPACE', REPO_ROOT / 'AnysoleWorkspace')),
        'results://': Path(os.environ.get('ANYSOLE_RESULTS', REPO_ROOT / 'results')),
        'display://': Path(os.environ.get('ANYSOLE_RESULTSDISPLAY', REPO_ROOT / 'results_display')),
    }
    for prefix, root in roots.items():
        if text.startswith(prefix):
            return str(root / text[len(prefix):])
    if not text:
        return text
    path = Path(text)
    return str(path if path.is_absolute() else BASELINE_ROOT / path)


def resolve_cfg_paths(cfg):
    for key in ('input_path_base', 'scene_rgbd'):
        cfg['task'][key] = resolve_path(cfg['task'][key])
    for key in ('smpl_file', 'smpl_male_file', 'vposer_path'):
        cfg['method'][key] = resolve_path(cfg['method'][key])
    return cfg

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    cfg = resolve_cfg_paths(cfg)
    device = torch.device('cuda:{}'.format(cfg['gpu'])) if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    task_cfg = cfg['task']
    method_cfg = cfg['method']

    configured_output = str(task_cfg.get('output_path', '') or '')
    if configured_output:
        opt_result_path = resolve_path(configured_output)
    else:
        # canonical work root: work://VP-MoCap/v1/pose_optimization/<date>/<subject>/<session>
        session_parts = Path(task_cfg['input_path_base']).parts[-3:]
        opt_result_path = str(
            REPO_ROOT / 'AnysoleWorkspace' / 'work' / 'VP-MoCap' / 'v1'
            / 'pose_optimization' / session_parts[0] / session_parts[1]
            / session_parts[2])
    if not os.path.isdir(opt_result_path):
        os.makedirs(opt_result_path)

    smpl_model = SMPL(method_cfg['smpl_file'], scale=task_cfg['scale']).to(device)

    camera = Camera(cfg, dtype=dtype, device=device)
    dataset = Dataset(cfg)
    write_visualization = bool(task_cfg.get('write_visualization', True))
    visualizer = Visualizer(cfg, opt_result_path) if write_visualization else None
    renderer = Renderer(focal_length=camera.focal_length, img_w=camera.img_w, img_h=camera.img_h,
                            faces=smpl_model.faces, same_mesh_color=False,
                            rotation=camera.rotation, translation=camera.translation) if write_visualization else None

    pose = torch.tensor(dataset.pose,dtype=dtype,device=device)
    betas = torch.tensor(dataset.betas,dtype=dtype,device=device)
    trans = initial_trans(cfg, dataset).to(device)

    kp_2d = torch.tensor(dataset.target_halpe_fil,dtype=dtype,device=device)
    kp_2d_score = torch.tensor(dataset.target_halpe_score,dtype=dtype,device=device)
    contact = dataset.contact

    point_cloud = torch.tensor(dataset.point_cloud, dtype=dtype, device=device)

    vp, ps = load_model(cfg['method'].get('vposer_path', 'models/V02_05'), model_code=VPoser,
                        remove_words_in_model_weights='vp_model.',
                        disable_grad=True)
    vp = vp.to(device)

    mse = torch.nn.MSELoss()
    pose_arr = []
    beta_arr = []
    trans_arr = []

    last_output_joints = None
    for frame in range(pose.shape[0]):
        log.info(f'frame {frame}')

        pose_single = pose[frame].unsqueeze(0)
        beta_single = betas[frame].unsqueeze(0)
        trans_single = trans[frame].unsqueeze(0)
        kp_2d_single = kp_2d[frame].unsqueeze(0)
        kp_2d_score_single = kp_2d_score[frame].unsqueeze(0)
        contact_single = contact[frame]

        pred_body_poZ = encode(pose_single,vp,device)
        pred_body_poZ.requires_grad = True

        trans_root = copy.deepcopy(trans_single)
        trans_root.requires_grad = True

        params = [pred_body_poZ,trans_root]
        optimizer = torch.optim.Rprop(params,lr=method_cfg['lr'])
        # optimizer = torch.optim.Adam(params,lr=method_cfg['lr'])

        last_loss = 0

        for i in range(cfg['method']['max_iter']):
            
            optimizer.zero_grad()
            pose_single_rec = decode(pred_body_poZ,vp,device,pose_single)

            output_opt = smpl_model(betas=beta_single,
                                    body_pose=pose_single_rec[:,1:],
                                    global_orient=pose_single_rec[:,[0]],
                                    pose2rot=False,
                                    transl=trans_root)
                        
            kp_2d_opt = perspective_projection(output_opt.joints[:,:25],camera.rotation,camera.translation,camera.focal_length,camera.camera_center)
            source = kp_2d_opt[:,torch.tensor(OPENPOSE),:]
            target = kp_2d_single[:,torch.tensor(HALPE),:]

            loss1 = loss_2d(source, target, kp_2d_score_single, method_cfg['lambda_2d_limbs'], method_cfg['lambda_2d_feet'], method_cfg['lambda_2d_others'])
            loss2 = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
            loss3 = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)

            loss2, loss3 = loss_3d_t(contact_single, point_cloud, target, output_opt, camera, mse, loss2, loss3, last_output_joints)
            
            loss4 = mse(pose_single,pose_single_rec) 

            loss = loss1 + loss2 * method_cfg['lambda_3d'] + loss3 * method_cfg['lambda_timing'] + loss4 * method_cfg['lambda_pose']
            loss.requires_grad_(True)        
            loss.backward(retain_graph=True)

            optimizer.step()
            if abs(loss - last_loss) < 1e-3:
                log.info('loss_delta almost equal to ZERO!')
                log.info(f'end at {i}, final loss {loss}')
                log.info(f'{loss1}, {loss2}, {loss3}, {loss4}')
                break
            last_loss = loss


        pose_arr.append(pose_single_rec.detach().cpu())
        beta_arr.append(beta_single.detach().cpu())
        trans_arr.append(trans_root.detach().cpu())

        last_output_joints = output_opt.joints
    
    pose_result = torch.from_numpy(fil_pose(np.stack(pose_arr)))
    # pose_result = torch.from_numpy(np.stack(pose_arr))

    beta_result = torch.cat(beta_arr)
    trans_result = torch.from_numpy(fil_trans(np.stack(trans_arr)))
    # trans_result = torch.from_numpy(np.stack(trans_arr))


    save_result = {
        'pose': pose_result,
        'beta': beta_result,
        'trans': trans_result,
        # shared frame ids of the optimized (post-join, post-trim) frames
        'frame_ids': np.asarray(dataset.frame_ids, dtype=np.int64),
    }

    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    torch.save(save_result, osp.join(opt_result_path, "opt_result.pth"))
    torch.save(save_result, osp.join(opt_result_path, "opt_result_" + Path(hydra_path).name.split('.')[0] + ".pth"))

    with open(osp.join(opt_result_path, 'artifact.json'), 'w',
              encoding='utf-8') as handle:
        json.dump({
            "schema_version": "posetransopt.result.v1",
            "producer": "app.optimize",
            "input_path_base": str(task_cfg['input_path_base']),
            "frame_ids_min": int(min(dataset.frame_ids)),
            "frame_ids_max": int(max(dataset.frame_ids)),
            "frame_count": int(len(dataset.frame_ids)),
            "segments": [[int(s), int(e)] for s, e in dataset.segments],
        }, handle, ensure_ascii=False, indent=2)

    if write_visualization:
        for frame in tqdm(range(pose_result.shape[0])):
            result = smpl_model(betas=beta_result[frame].unsqueeze(0).type(dtype).to(device),
                                body_pose=pose_result[frame,1:].unsqueeze(0).type(dtype).to(device),
                                global_orient=pose_result[frame,[0]].unsqueeze(0).type(dtype).to(device),
                                pose2rot=False,
                                transl=trans_result[frame].unsqueeze(0).type(dtype).to(device))

            visualizer.visual_smpl_2d_single(result.vertices, renderer, frame)
            visualizer.save_o3d_mesh(result.vertices, smpl_model.faces, frame)


if __name__ == "__main__":
    main()
