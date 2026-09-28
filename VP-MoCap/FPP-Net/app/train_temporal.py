from lib.config.config import config_cont as config
import argparse
import torch
from torch.nn import DataParallel
import os
from torch.utils.data import DataLoader
from icecream import ic

from lib.Dataset import make_dataset
from lib.Networks import make_network
from lib.Trainer import make_trainer
from lib.Record.record import ContRecorder



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--num_threads', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--gpus', type=str, default='cpu', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override total epochs (default: config value)')
    parser.add_argument('--early-stop-patience', type=int, default=None,
                        help='stop after N epochs without improvement > '
                        'min-delta (default: config value; 0 disables)')
    parser.add_argument('--early-stop-min-delta', type=float, default=None,
                        help='loss improvement threshold (default: config value)')
    parser.add_argument('--early-stop-lr-floor', type=float, default=None,
                        help='stop once LR decays below this (default: config value)')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='disable early stopping entirely')
    arg = parser.parse_args()

    cfg = config()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()
    cfg.defrost()
    cfg.num_threads = arg.num_threads
    cfg.gpus = arg.gpus
    cfg.batch_size = arg.batch_size
    if arg.epochs is not None:
        cfg.trainer.epochs = arg.epochs
    if arg.no_early_stop:
        cfg.trainer.early_stop_patience = 0
        cfg.trainer.early_stop_lr_floor = 0.0
    else:
        if arg.early_stop_patience is not None:
            cfg.trainer.early_stop_patience = arg.early_stop_patience
        if arg.early_stop_min_delta is not None:
            cfg.trainer.early_stop_min_delta = arg.early_stop_min_delta
        if arg.early_stop_lr_floor is not None:
            cfg.trainer.early_stop_lr_floor = arg.early_stop_lr_floor
    cfg.freeze()
    ic(cfg)

    os.makedirs(cfg.checkpoint_path, exist_ok=True)
    os.makedirs(cfg.result_path, exist_ok=True)
    os.makedirs('%s/%s' % (cfg.checkpoint_path, cfg.name), exist_ok=True)
    os.makedirs('%s/%s' % (cfg.result_path, cfg.name), exist_ok=True)

    device = None

    dataset_bsc = make_dataset.make_dataset(cfg.dataset, phase='train')
    # dataset_bsc.__getitem__(0)
    # exit()

    dataloader = DataLoader(dataset_bsc, batch_size=cfg.batch_size, shuffle=not cfg.dataset.serial_batches,
                            num_workers=cfg.num_threads, pin_memory=cfg.dataset.pin_memory)

    gc_net = make_network.make_network(cfg.networks)

    if cfg.gpus != 'cpu':
        gpu_ids = [int(i) for i in cfg.gpus.split(',')]
        device = torch.device('cuda:%d' % gpu_ids[0])

        gc_net = gc_net.to(device)
        gc_net = DataParallel(gc_net, gpu_ids)

    else:
        gpu_ids = None

    if os.path.exists(cfg.load_net_checkpoint):
        print('gc_net : loading from %s' % cfg.load_net_checkpoint)
        gc_net.load_state_dict(torch.load(cfg.load_net_checkpoint))
    else:
        print('can not find checkpoint %s' % cfg.load_net_checkpoint)

    optimizer = torch.optim.Adam(gc_net.parameters(), lr=cfg.trainer.lr)
    recorder = ContRecorder(cfg)

    trainer = make_trainer.make_trainer(dataloader, gc_net, optimizer, recorder, gpu_ids, cfg.trainer)

    trainer.train()
    print("=========================Training completed!!!!!=========================")

