import os
import sys

import numpy as np


def findAllFilesWithSpecifiedName(target_dir, target_name):
    find_res = []
    walk_generator = os.walk(target_dir, followlinks=True)
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        for file in files:
            if target_name in file:
                find_res.append(os.path.join(root_path, file))
    find_res.sort()
    return find_res


def _allow_numpy2_pickle():
    """Let NumPy 1.x unpickle arrays written by NumPy 2.x."""
    try:
        import numpy.core.multiarray as multiarray
        sys.modules.setdefault('numpy._core', sys.modules['numpy.core'])
        sys.modules.setdefault('numpy._core.multiarray', multiarray)
    except Exception:
        pass


def load_smpl_npy(path):
    _allow_numpy2_pickle()
    return np.load(path, allow_pickle=True).item()
