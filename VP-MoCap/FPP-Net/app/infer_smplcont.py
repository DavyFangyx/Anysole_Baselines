from lib.config.config import config_cont as config
import argparse
import torch
import os,cv2
from pathlib import Path
from tqdm import tqdm,trange
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from icecream import ic
import numpy as np

from lib.Dataset import make_dataset
from lib.Networks import make_network
from lib.Dataset.InsoleModule import InsoleModule
import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--phase', choices=('train', 'val', 'test'), default='test')
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--batch_size', default=1,type=int)
    parser.add_argument('--max_batches', type=int, default=0,
                        help='CPU/debug smoke limit; 0 means the complete split')
    parser.add_argument('--no_visualization', action='store_true',
                        help='skip per-frame PNG and only export the prediction sidecars')
    parser.add_argument('--gpus', type=str, default='cpu', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
    arg = parser.parse_args()

    cfg = config()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()
    cfg.defrost()
    if os.environ.get('FPP_CHECKPOINT'):
        cfg.load_net_checkpoint = os.environ['FPP_CHECKPOINT']
    cfg.num_threads = arg.num_threads
    cfg.gpus = arg.gpus
    cfg.batch_size = arg.batch_size
    # Inference must use the original keypoints/pressure coordinate frame,
    # including when exporting the train split.  Training keeps augmentation
    # enabled; this entrypoint must not write augmented predictions back
    # under the original frame names.
    cfg.dataset.aug.is_aug = False
    cfg.freeze()
    ic(cfg)

    if not os.path.isfile(cfg.load_net_checkpoint):
        raise FileNotFoundError(
            'FPP-Net checkpoint does not exist: %s. Train FPP-Net first or '
            'set FPP_CHECKPOINT to a trained checkpoint.'
            % cfg.load_net_checkpoint)

    os.makedirs(cfg.result_path, exist_ok=True)
    os.makedirs('%s/%s' % (cfg.result_path, cfg.name), exist_ok=True)

    dataset_bsc = make_dataset.make_dataset(cfg.dataset, phase=arg.phase)
    dataloader = DataLoader(dataset_bsc, batch_size=cfg.batch_size, shuffle=False,
               num_workers=cfg.num_threads, pin_memory=cfg.dataset.pin_memory)

    gc_net = make_network.make_network(cfg.networks)

    if cfg.gpus != 'cpu':
        gpu_ids = [int(i) for i in cfg.gpus.split(',')]
        device = torch.device('cuda:%d' % gpu_ids[0])
        gc_net = gc_net.to(device)
        gc_net = DataParallel(gc_net, gpu_ids)

    else:
        gpu_ids = None
        device = torch.device('cpu')

    print('gc_net : loading from %s' % cfg.load_net_checkpoint)
    state = torch.load(cfg.load_net_checkpoint, map_location=device)
    if isinstance(state, dict):
        state_keys = list(state.keys())
        model_keys = list(gc_net.state_dict().keys())
        has_module = bool(state_keys) and all(key.startswith('module.') for key in state_keys)
        model_has_module = bool(model_keys) and all(key.startswith('module.') for key in model_keys)
        if has_module and not model_has_module:
            state = {key[len('module.'):]: value for key, value in state.items()}
        elif not has_module and model_has_module:
            state = {f'module.{key}': value for key, value in state.items()}
    gc_net.load_state_dict(state)
    gc_net.eval()

    mse_press_ls,bce_cont_ls,mse_cont_ls = [],[],[]
    insole_ratio = 10
    m_insole = InsoleModule(
        cfg.dataset.datadir, getattr(cfg.dataset, 'essentials_root', None))
    for batch_index, data in enumerate(tqdm(dataloader)):
        for data_item in ['keypoints']:
            data[data_item] = data[data_item].to(device=device)
        pred_press, pred_cont = gc_net(keypoints=data['keypoints'])
        pred_press = pred_press.detach().cpu()
        pred_cont = pred_cont.detach().cpu()

        mse_press = F.mse_loss(pred_press, data['insole'])
        mse_cont = F.mse_loss(pred_cont, data['contact_smpl'])
        # The contact head emits a probability per foot vertex.  The metrics
        # and the exported sidecars must be computed on that continuous
        # probability: thresholding first would turn the BCE into a value
        # about the binarized array instead of the head's calibration.
        bce_cont = F.binary_cross_entropy(pred_cont, data['contact_smpl'])
        mse_press_ls.append(mse_press.numpy())
        bce_cont_ls.append(bce_cont.numpy())
        mse_cont_ls.append(mse_cont.numpy())

        pred_press_batch = pred_press.numpy()
        press_gt_batch = data['insole'].detach().cpu().numpy()
        # The visualization is one image per batch and keeps the first
        # sample; the metrics and the per-frame export below use every
        # sample of the batch.
        pred_press = pred_press_batch[0].reshape([-1])
        press_gt = press_gt_batch[0].reshape([-1])

        pred_cont_np = pred_cont.numpy()
        pred_cont_binary = (pred_cont_np > 0.5).astype(np.float32)
        cont_gt = data['contact_smpl'][0].detach().cpu().numpy()
        cont_pred = pred_cont_binary[0]

        if not arg.no_visualization:
            press_gt_img, press_gt_data = m_insole.visMaskedPressure(press_gt)
            press_pred_img, press_pred_data = m_insole.visMaskedPressure(pred_press)
            press_gt_img = cv2.resize(press_gt_img,(press_gt_img.shape[1]*insole_ratio,press_gt_img.shape[0]*insole_ratio))
            press_pred_img = cv2.resize(press_pred_img,(press_pred_img.shape[1]*insole_ratio,press_pred_img.shape[0]*insole_ratio))

            image_path = data['case_name'][0]
            data_id,sub_ids,seq_name,frame_ids = (image_path).split('/')
            color_root = Path(cfg.dataset.datadir) / data_id / sub_ids / seq_name / 'color'
            color_candidates = sorted(color_root.glob(frame_ids + '.*'))
            img_fn = str(color_candidates[0]) if color_candidates else str(color_root / (frame_ids + '.png'))
            img = cv2.imread(img_fn)

            insole_img = np.concatenate([press_gt_img,press_pred_img],axis=1)
            img = cv2.resize(img, (insole_img.shape[1], int(img.shape[0] / (img.shape[1] / insole_img.shape[1]))))
            save_img = np.concatenate([img,insole_img],axis=0)
            save_fn = os.path.join(cfg.result_path, cfg.name,'%s_%s_%s_%s.png'%(data_id,sub_ids,seq_name,frame_ids))
            cv2.imwrite(save_fn,save_img)

            results={
                'pressure':{
                    'gt':press_gt_data,
                    'pred':press_pred_data,},
                'contact_smpl':{
                    'gt':cont_gt,
                    'pred':cont_pred,}}
            np.save(save_fn[:-3]+'npy',results)

        # PoseTransOpt consumes one file per real frame.  The exported map
        # keeps the head's continuous probability (never the thresholded
        # visualization array) and each sidecar carries the shared frame id,
        # so the consumer joins on frame ids instead of file order.
        case_names = data['case_name']
        for batch_i, case in enumerate(case_names):
            parts = str(case).split('/')
            if len(parts) != 4:
                raise ValueError('unexpected FPP case_name: %s' % case)
            frame = int(parts[-1])
            out_dir = Path(cfg.dataset.prediction_root) / parts[0] / parts[1] / parts[2] / 'pred_contact_smpl'
            out_dir.mkdir(parents=True, exist_ok=True)
            _, p_pred_map = m_insole.visMaskedPressure(pred_press_batch[batch_i])
            _, p_gt_map = m_insole.visMaskedPressure(press_gt_batch[batch_i])
            np.save(out_dir / ('%06d.npy' % frame), {
                'frame_id': frame,
                'pressure': {
                    'pred': p_pred_map.astype(np.float32),
                    'gt': p_gt_map.astype(np.float32),
                },
                'contact_smpl': {
                    'pred': pred_cont_np[batch_i].reshape(2, -1).astype(np.float32),
                    'gt': data['contact_smpl'][batch_i].detach().cpu().numpy().reshape(2, -1).astype(np.float32),
                },
            })

        if arg.max_batches > 0 and batch_index + 1 >= arg.max_batches:
            print('Stopping after max_batches=%d' % arg.max_batches)
            break

    print("pressure mse:",np.mean(mse_press_ls),'cont mse:',np.mean(mse_cont_ls),'cont bce:',np.mean(bce_cont_ls))
