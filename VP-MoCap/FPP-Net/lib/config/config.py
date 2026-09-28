import os
import sys
from pathlib import Path
from yacs.config import CfgNode as CN


REPO_ROOT = Path(__file__).resolve().parents[5]


def resolve_workspace_path(value):
    text = str(value or '')
    if '://' in text:
        # canonical workspace URIs (shared://, model-input://, work://) go
        # through the single public resolver; the legacy roots below are
        # resolved locally so a config can still be read without importing
        # the workspace package.
        if text.startswith(('shared://', 'model-input://', 'work://')):
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
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
    return text


class config_cont():

    def __init__(self):
        self.cfg = CN()
        self.cfg.name = ''
        self.cfg.logdir = ''
        self.cfg.load_net_checkpoint = ''
        self.cfg.checkpoint_path = ''
        self.cfg.result_path = ''

        self.cfg.trainer = CN()
        self.cfg.trainer.lr = 1e-4
        self.cfg.trainer.epochs = 400
        self.cfg.trainer.module = ''
        self.cfg.trainer.path = ''
        self.cfg.trainer.num_train_epochs = 100
        self.cfg.trainer.w_press = 0.4
        self.cfg.trainer.w_cont = 0.6

        self.cfg.dataset = CN()
        self.cfg.dataset.module = ''
        self.cfg.dataset.path = ''
        self.cfg.dataset.datadir = ''
        self.cfg.dataset.seq_name = ''
        self.cfg.dataset.tv_fn = ''
        self.cfg.dataset.tactile_root = ''
        self.cfg.dataset.prediction_root = ''
        self.cfg.dataset.essentials_root = ''
        self.cfg.dataset.w_sc = 1.0
        # self.cfg.dataset.w_bsc = 1.0
        self.cfg.dataset.w_nc = 1.0
        self.cfg.dataset.img_res = 224
        self.cfg.dataset.serial_batches = False
        self.cfg.dataset.pin_memory = False
        self.cfg.dataset.aug = CN()
        self.cfg.dataset.aug.is_aug = True
        self.cfg.dataset.aug.is_sam = False
        self.cfg.dataset.aug.noise_factor = 0.4
        self.cfg.dataset.aug.scale_factor = 0.15  # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.cfg.dataset.aug.rot_factor = 30  # Random rotation in the range [-rot_factor, rot_factor]
        self.cfg.dataset.g2g_num = 100  # Random rotation in the range [-rot_factor, rot_factor]

        self.cfg.networks = CN()
        self.cfg.networks.module = ''
        self.cfg.networks.path = ''
        self.cfg.networks.seqlen = 5

        self.cfg.record = CN()
        self.cfg.record.save_freq = 1
        self.cfg.record.show_freq = 1
        self.cfg.record.print_freq = 1

        self.cfg.gpus = ''
        self.cfg.batch_size = 0

    def get_cfg(self):
        return self.cfg.clone()

    def load(self, config_file):
        self.cfg.defrost()
        self.cfg.merge_from_file(config_file)
        for key in ('datadir', 'tv_fn', 'tactile_root', 'prediction_root', 'essentials_root'):
            self.cfg.dataset[key] = resolve_workspace_path(self.cfg.dataset[key])
        if self.cfg.dataset.essentials_root and not Path(self.cfg.dataset.essentials_root).is_absolute():
            self.cfg.dataset.essentials_root = str(REPO_ROOT / self.cfg.dataset.essentials_root)
        for key in ('load_net_checkpoint', 'checkpoint_path', 'result_path', 'logdir'):
            value = str(self.cfg.get(key, ''))
            if value.startswith(('workspace://', 'results://', 'display://')):
                self.cfg[key] = resolve_workspace_path(value)
        self.cfg.freeze()
