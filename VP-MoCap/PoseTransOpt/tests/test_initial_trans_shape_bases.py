"""PoseTransOpt initial-translation SMPL loading tests (registration #7).

Official SMPL files may ship 300 PCA shape bases while this baseline's
interface (dataset ``betas`` -> ``initial_trans`` -> TranSolver) supplies the
standard 10-dimensional beta vector.  ``transpose/model.py`` therefore keeps
the file untouched and uses the first 10 bases, raising when the file cannot
supply 10.

Run directly:

    /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python \
        tests/test_initial_trans_shape_bases.py

The native env is ``mmvp``; ``touch_gait`` lacks icecream, so this file
installs a minimal stub for it.
"""
from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

POSETRANSOPT_ROOT = Path(__file__).resolve().parents[1]
if str(POSETRANSOPT_ROOT) not in sys.path:
    sys.path.insert(0, str(POSETRANSOPT_ROOT))


def _install_light_stubs():
    try:
        __import__("icecream")
    except ModuleNotFoundError:
        module = types.ModuleType("icecream")
        module.ic = lambda *args, **kwargs: None
        sys.modules["icecream"] = module


_install_light_stubs()

from lib.initial_trans.transpose.model import ParametricModel  # noqa: E402

N_JOINT = 24
N_VERT = 12
# SMPL-24 parent ids (root first)
KINTREE = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
# a fixed spatial pattern shared by every basis: makes the shape offset
# non-uniform over vertices, so a wrong basis selection cannot cancel out
SPATIAL = np.random.default_rng(1).normal(size=(N_VERT, 3))


def _write_smpl_pickle(path: Path, n_bases: int, shape_ndim: int = 3,
                       basis_offset: int = 0) -> None:
    """Write a minimal SMPL pickle with a distinguishable basis per index.

    ``shapedirs[:, :, k] = SPATIAL * (k + 1 + basis_offset)``: basis index k is
    identifiable from the mesh offset it produces, and ``basis_offset`` lets a
    test build a file whose first 10 bases differ from another file's.
    """
    rng = np.random.default_rng(0)
    shapedirs = np.zeros((N_VERT, 3, n_bases), dtype=np.float64)
    for k in range(n_bases):
        shapedirs[:, :, k] = SPATIAL * float(k + 1 + basis_offset)
    if shape_ndim == 2:
        shapedirs = shapedirs.reshape(N_VERT, 3 * n_bases)
    payload = {
        "J_regressor": sp.csr_matrix(np.eye(N_JOINT, N_VERT, dtype=np.float64)),
        "weights": np.full((N_VERT, N_JOINT), 1.0 / N_JOINT, dtype=np.float64),
        "posedirs": np.zeros((N_VERT * 3, 9 * (N_JOINT - 1)), dtype=np.float64),
        "shapedirs": shapedirs,
        "v_template": rng.normal(size=(N_VERT, 3)),
        "J": rng.normal(size=(N_JOINT, 3)),
        "f": [[0, 1, 2], [1, 2, 3]],
        "kintree_table": np.asarray([KINTREE, list(range(N_JOINT))], dtype=np.int64),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _meshes(model: ParametricModel, betas: torch.Tensor):
    joints, vertices = model.get_zero_pose_joint_and_vertex(betas)
    return joints.detach().numpy(), vertices.detach().numpy()


def test_shapedirs_truncated_to_ten_bases(tmp_path):
    path = tmp_path / "SMPL_300.pkl"
    _write_smpl_pickle(path, n_bases=300)
    model = ParametricModel(str(path))
    assert model._shapedirs.shape == (N_VERT, 3, 10), model._shapedirs.shape
    # the first 10 bases in order: not the last 10, not a strided selection
    expected = np.stack([SPATIAL * float(k + 1) for k in range(10)], axis=-1)
    assert np.allclose(model._shapedirs.numpy(), expected)


def test_shape_offset_composes_with_the_ten_betas(tmp_path):
    """The mesh offset equals sum_k betas[k] * basis_k over the first 10 bases."""
    path = tmp_path / "SMPL_300.pkl"
    _write_smpl_pickle(path, n_bases=300)
    model = ParametricModel(str(path))
    betas = torch.arange(1, 11, dtype=torch.float32).view(1, 10) * 0.1  # k -> 0.1 k
    _, vertices = _meshes(model, betas)
    _, vertices_zero = _meshes(model, torch.zeros(1, 10))
    # vertex 0 is the joint-0 anchor (offsets are taken relative to it)
    offset = vertices[0] - vertices_zero[0]
    expected = (SPATIAL - SPATIAL[0]) * sum(0.1 * (i + 1) ** 2 for i in range(10))
    assert np.allclose(offset, expected, rtol=1e-4, atol=1e-3), (offset[1], expected[1])


def test_a_300_base_file_matches_its_10_base_equivalent(tmp_path):
    """A 300-base file and a 10-base file must yield the same mesh."""
    path_wide = tmp_path / "SMPL_300.pkl"
    path_ten = tmp_path / "SMPL_10.pkl"
    _write_smpl_pickle(path_wide, n_bases=300)
    _write_smpl_pickle(path_ten, n_bases=10)
    wide = ParametricModel(str(path_wide))
    ten = ParametricModel(str(path_ten))
    betas = torch.arange(1, 11, dtype=torch.float32).view(1, 10) * 0.1
    wide_j, wide_v = _meshes(wide, betas)
    ten_j, ten_v = _meshes(ten, betas)
    assert np.allclose(wide_v, ten_v, atol=1e-5)
    assert np.allclose(wide_j, ten_j, atol=1e-5)
    # sensitivity: pruning to a *different* 10 bases would change the mesh, so
    # the check above really discriminates the first 10 from any other slice
    path_shifted = tmp_path / "SMPL_300_shifted.pkl"
    _write_smpl_pickle(path_shifted, n_bases=300, basis_offset=10)
    shifted = ParametricModel(str(path_shifted))
    _, shifted_v = _meshes(shifted, betas)
    assert not np.allclose(wide_v, shifted_v, atol=1e-3)


def test_too_few_bases_raise(tmp_path):
    path = tmp_path / "SMPL_9.pkl"
    _write_smpl_pickle(path, n_bases=9)
    try:
        ParametricModel(str(path))
    except ValueError as error:
        assert "at least 10 bases" in str(error), str(error)
        return
    raise AssertionError("a file with fewer than 10 shape bases must raise")


def test_non_three_dimensional_shapedirs_raise(tmp_path):
    path = tmp_path / "SMPL_flat.pkl"
    _write_smpl_pickle(path, n_bases=300, shape_ndim=2)
    try:
        ParametricModel(str(path))
    except ValueError as error:
        assert "shapedirs" in str(error), str(error)
        return
    raise AssertionError("shapedirs must be a 3-D (V, 3, K) array")


TESTS = (
    test_shapedirs_truncated_to_ten_bases,
    test_shape_offset_composes_with_the_ten_betas,
    test_a_300_base_file_matches_its_10_base_equivalent,
    test_too_few_bases_raise,
    test_non_three_dimensional_shapedirs_raise,
)


if __name__ == "__main__":
    import tempfile

    for test in TESTS:
        with tempfile.TemporaryDirectory() as folder:
            test(Path(folder))
        print(f"ok: {test.__name__}")
    print("test_initial_trans_shape_bases: all passed")
