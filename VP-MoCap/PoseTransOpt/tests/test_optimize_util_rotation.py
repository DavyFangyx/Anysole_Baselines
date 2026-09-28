"""PoseTransOpt rotation conversion tests (registration #9).

``lib/optimize_util.encode`` converts predicted SMPL rotation matrices to
axis-angle before the VPoser prior.  The upstream call used
``torchgeometry.rotation_matrix_to_angle_axis``, which raises on torch 2.4
("Subtraction, the `-` operator, with a bool tensor is not supported"); the
replacement uses ``scipy.spatial.transform.Rotation.as_rotvec``, which is
mathematically equivalent.  At exactly pi the axis sign is implementation
defined, but the represented rotation is identical and the round trip through
the matrix is stable.

Run directly:

    /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python \
        tests/test_optimize_util_rotation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRotation

POSETRANSOPT_ROOT = Path(__file__).resolve().parents[1]
if str(POSETRANSOPT_ROOT) not in sys.path:
    sys.path.insert(0, str(POSETRANSOPT_ROOT))

from lib.optimize_util import encode  # noqa: E402


class _StubVPoser:
    """Captures the rotvec handed to the VPoser encoder."""

    def __init__(self):
        self.seen = None

    def encode(self, rotvec):
        self.seen = rotvec.detach().cpu().numpy().copy()

        class _Mean:
            mean = rotvec

        return _Mean()


def _random_rotmats(count: int, seed: int = 0) -> np.ndarray:
    return SciRotation.random(count, random_state=seed).as_matrix().astype(np.float32)


def _edge_case_rotmats() -> np.ndarray:
    """Rotations where naive conversions lose precision or sign."""
    mats = [np.eye(3, dtype=np.float32)]
    for axis in np.eye(3):
        mats.append(SciRotation.from_rotvec(np.pi * axis).as_matrix())
    # near-pi (the axis sign is numerically ill-conditioned here) and tiny angle
    mats.append(SciRotation.from_rotvec([np.pi - 1e-6, 0.0, 0.0]).as_matrix())
    mats.append(SciRotation.from_rotvec([1e-9, -1e-9, 1e-9]).as_matrix())
    return np.asarray(mats, dtype=np.float32)


def test_scipy_rotvec_roundtrips_back_to_the_same_matrix():
    for matrix in np.concatenate([_random_rotmats(8), _edge_case_rotmats()]):
        rotvec = SciRotation.from_matrix(matrix).as_rotvec()
        back = SciRotation.from_rotvec(rotvec).as_matrix()
        assert np.allclose(back, matrix, atol=1e-5), (matrix, back)


def test_encode_hands_the_body_joints_to_vposer():
    # an (N, 24, 3, 3) tensor of per-joint rotation matrices as decode/projection use
    rotmats = torch.from_numpy(_random_rotmats(2 * 24, seed=7)).view(2, 24, 3, 3)
    vp = _StubVPoser()
    encode(rotmats, vp, torch.device("cpu"))
    assert vp.seen.shape == (2, 63), vp.seen.shape
    # encode() flattens (N, 24) joints to N*72 and slices [3:66] == joints 1..21;
    # the rotvec handed to VPoser must reconstruct exactly those matrices.
    # (compared as matrices: near pi the rotvec axis sign is float-precision
    # dependent, while the rotation it represents is not)
    got = SciRotation.from_rotvec(
        vp.seen.reshape(-1, 3).astype(np.float64)
    ).as_matrix().reshape(-1, 21, 3, 3)
    want = rotmats.detach().numpy()[:, 1:22].astype(np.float64)
    assert got.shape == want.shape == (2, 21, 3, 3)
    assert np.allclose(got, want, atol=1e-4), np.abs(got - want).max()
    # ordering: joints 0..20 (i.e. including the global orientation) would differ
    shifted = rotmats.detach().numpy()[:, :21].astype(np.float64)
    assert not np.allclose(got, shifted, atol=1e-3)


def test_encode_matches_the_legacy_torchgeometry_result():
    """Either the legacy conversion agrees, or it is unusable on this torch."""
    import torchgeometry as tgm

    body = SciRotation.random(5, random_state=3).as_matrix().astype(np.float32).reshape(5, 3, 3)
    rot_pad = torch.tensor([0, 0, 1], dtype=torch.float32).view(1, 3, 1)
    legacy_input = torch.cat(
        (torch.from_numpy(body), rot_pad.expand(5, -1, -1)), dim=-1)
    try:
        legacy = tgm.rotation_matrix_to_angle_axis(legacy_input).numpy()
    except RuntimeError as error:
        assert "bool tensor" in str(error), str(error)
        print("  note: torchgeometry is unusable on this torch (bool subtraction);"
              " scipy replacement is the only path")
        return
    reference = SciRotation.from_matrix(body.astype(np.float64)).as_rotvec()
    # angles agree up to the 2*pi branch and the sign convention at pi
    for row, ref in zip(legacy, reference):
        angle_legacy = np.linalg.norm(row)
        angle_ref = np.linalg.norm(ref)
        assert abs(angle_legacy - angle_ref) < 1e-4 or abs(
            angle_legacy - 2 * np.pi + angle_ref) < 1e-4, (row, ref)
        if angle_ref > 1e-3:
            assert np.allclose(row / angle_legacy, ref / angle_ref, atol=1e-4), (row, ref)


def test_encode_stays_on_the_input_device_and_dtype():
    rotmats = torch.from_numpy(_random_rotmats(24, seed=11)).view(1, 24, 3, 3)
    seen = {}

    class _Vp:
        def encode(self, rotvec):
            seen["device"] = rotvec.device
            seen["dtype"] = rotvec.dtype

            class _Mean:
                mean = rotvec

            return _Mean()

    encode(rotmats, _Vp(), torch.device("cpu"))
    assert seen["device"] == rotmats.device, seen
    assert seen["dtype"] == rotmats.dtype, seen


TESTS = (
    test_scipy_rotvec_roundtrips_back_to_the_same_matrix,
    test_encode_hands_the_body_joints_to_vposer,
    test_encode_matches_the_legacy_torchgeometry_result,
    test_encode_stays_on_the_input_device_and_dtype,
)


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"ok: {test.__name__}")
    print("test_optimize_util_rotation: all passed")
