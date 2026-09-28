"""#7 evidence: the inference entrypoint keeps probabilities and exports them.

Upstream thresholded the contact head (``pred_cont[pred_cont>0.5] = 1``)
*before* computing the BCE, so the reported number described a binarized
array rather than the head's calibration; and it wrote only a PNG plus a
per-batch ``.npy`` without any frame id.

This test runs the real entrypoint (shipped config + real dataset + real
network, random init checkpoint) on a small fixture tree and checks the
exported contract against what the PoseTransOpt adapter reads:
``prediction_root/<date>/<sub>/<seq>/pred_contact_smpl/%06d.npy`` carrying
``frame_id``, ``pressure.{pred,gt}`` (31,22) float32 and
``contact_smpl.{pred,gt}`` (2,96) float32 with a *continuous* prediction.
The printed BCE is re-derived from the sidecar to prove it is computed on
that continuous probability.

Run directly:

    python tests/test_infer_export_contract.py
"""
from __future__ import annotations

import contextlib
import io
import os
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

FPP_ROOT = Path(__file__).resolve().parents[1]
if str(FPP_ROOT) not in sys.path:
    sys.path.insert(0, str(FPP_ROOT))

from lib.config.config import config_cont  # noqa: E402
from lib.Networks import make_network  # noqa: E402

CONFIG = FPP_ROOT / "configs/temporalKPSMPLCont_series5_mlp.yaml"
ENTRYPOINT = FPP_ROOT / "app/infer_smplcont.py"
N_JOINTS = 26
N_FRAMES = 8
FRAMES = [2, 3]
WEIGHT = 2000.0


def _write_case(root):
    seq = root / "datadir/20260808/S10/S1011"
    (seq / "keypoints").mkdir(parents=True)
    np.save(root / "datadir/20260808/sub_info.npy", {"S10": {"weight": WEIGHT}})
    rng = np.random.default_rng(11)
    for frame in range(N_FRAMES):
        np.save(seq / "keypoints" / ("%06d.npy" % frame), {
            "keypoints": (rng.random((N_JOINTS, 2)) * 400 + 200).astype(np.float32),
            "keypoint_scores": np.full((N_JOINTS,), 0.9, dtype=np.float32),
            "frame_id": frame,
        })
    tactile = root / "tactile/20260808/S10/S1011"
    (tactile / "insole").mkdir(parents=True)
    for frame in range(N_FRAMES):
        np.save(tactile / "insole" / ("%06d.npy" % frame),
                (rng.random((2, 31, 11)) * 3000).astype(np.float32))
    np.save(tactile / "frame_id.npy", np.arange(N_FRAMES))
    split = root / "split.npy"
    np.save(split, {"test": {"20260808": {"S10": {"S1011":
                 ["%06d.npy" % f for f in FRAMES]}}}})
    return split


def _temp_config(root, split):
    """The shipped config with every path pointed into the fixture.

    The plugin paths (``dataset.path`` etc.) are relative to the repo, so
    they are rewritten to absolute paths: the entrypoint is run from a
    scratch working directory.
    """
    text = CONFIG.read_text()
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("path:"):
            relative = stripped[len("path:"):].strip().strip("'\"")
            prefix = line[:len(line) - len(line.lstrip())]
            line = "%spath: '%s'" % (prefix, FPP_ROOT / relative)
        lines.append(line)
    text = "\n".join(lines) + "\n"
    for key, value in (
            ("datadir", "'%s'" % (root / "datadir")),
            ("tv_fn", "'%s'" % split),
            ("tactile_root", "'%s'" % (root / "tactile")),
            ("prediction_root", "'%s'" % (root / "predictions")),
            ("essentials_root", "'%s'" % (FPP_ROOT / "essentials")),
            ("result_path", "'%s'" % (root / "metrics")),
    ):
        lines = []
        for line in text.splitlines():
            if line.strip().startswith(key + ":"):
                prefix = line[:len(line) - len(line.lstrip())]
                line = "%s%s: %s" % (prefix, key, value)
            lines.append(line)
        text = "\n".join(lines) + "\n"
    path = root / "config.yaml"
    path.write_text(text)
    return path


def _fake_checkpoint(cfg, path):
    net = make_network.make_network(cfg.networks)
    torch.save(net.state_dict(), path)
    return path


def _run_entrypoint(cfg_path):
    argv = ["infer_smplcont.py", "--config", str(cfg_path), "--num_threads", "0",
            "--batch_size", "1", "--max_batches", "%d" % len(FRAMES),
            "--no_visualization"]
    stdout = io.StringIO()
    cwd = os.getcwd()
    saved_argv = sys.argv
    sys.argv = argv
    with tempfile.TemporaryDirectory() as run_dir:
        os.chdir(run_dir)
        try:
            with contextlib.redirect_stdout(stdout):
                runpy.run_path(str(ENTRYPOINT), run_name="__main__")
        finally:
            os.chdir(cwd)
            sys.argv = saved_argv
    return stdout.getvalue()


def test_entrypoint_exports_the_frame_id_sidecar_contract():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        split = _write_case(root)
        cfg_path = _temp_config(root, split)
        cfg = config_cont()
        cfg.load(str(cfg_path))
        cfg = cfg.get_cfg()
        checkpoint = _fake_checkpoint(cfg, root / "net.pth")
        os.environ["FPP_CHECKPOINT"] = str(checkpoint)
        try:
            out = _run_entrypoint(cfg_path)
        finally:
            os.environ.pop("FPP_CHECKPOINT", None)

        out_dir = root / "predictions/20260808/S10/S1011/pred_contact_smpl"
        payloads = {}
        for frame in FRAMES:
            path = out_dir / ("%06d.npy" % frame)
            assert path.is_file(), f"missing export: {path}"
            payload = np.load(path, allow_pickle=True).item()
            payloads[frame] = payload
            assert int(payload["frame_id"]) == frame
            for name, arr in (("pressure pred", payload["pressure"]["pred"]),
                              ("pressure gt", payload["pressure"]["gt"])):
                assert arr.shape == (31, 22) and arr.dtype == np.float32, name
            for name, arr in (("contact pred", payload["contact_smpl"]["pred"]),
                              ("contact gt", payload["contact_smpl"]["gt"])):
                assert arr.shape == (2, 96) and arr.dtype == np.float32, name

        # the exported prediction is the continuous probability, not a mask
        pred = payloads[FRAMES[0]]["contact_smpl"]["pred"]
        gt = payloads[FRAMES[0]]["contact_smpl"]["gt"]
        assert pred.min() > 0.0 and pred.max() < 1.0, "prediction must be raw probability"
        assert set(np.unique(gt)) <= {0.0, 1.0}, "the GT stays a binary mask"
        assert not np.array_equal(payloads[FRAMES[0]]["pressure"]["pred"],
                                  payloads[FRAMES[1]]["pressure"]["pred"])

        # the printed bce must be the one computed on that probability
        printed = float(out.split("cont bce:")[1].split()[0])
        # one batch per frame: the printed value is their mean
        bce_continuous, bce_binarized = [], []
        for frame in FRAMES:
            p = payloads[frame]["contact_smpl"]["pred"]
            t = torch.from_numpy(payloads[frame]["contact_smpl"]["gt"])
            bce_continuous.append(float(F.binary_cross_entropy(
                torch.from_numpy(p), t)))
            bce_binarized.append(float(F.binary_cross_entropy(
                torch.from_numpy((p > 0.5).astype(np.float32)), t)))
        continuous = float(np.mean(bce_continuous))
        binarized = float(np.mean(bce_binarized))
        assert abs(printed - continuous) < 1e-6, (
            f"printed bce {printed} is not the continuous-probability bce "
            f"{continuous}")
        assert abs(printed - binarized) > 1e-6, (
            "the printed bce still looks like the thresholded variant")


def test_missing_checkpoint_is_rejected_before_any_work():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        split = _write_case(root)
        cfg_path = _temp_config(root, split)
        assert not (root / "predictions").exists()
        try:
            _run_entrypoint(cfg_path)
        except FileNotFoundError as err:
            assert "checkpoint does not exist" in str(err), err
        else:
            raise AssertionError("a missing checkpoint must stop the run")
        assert not (root / "predictions").exists()


if __name__ == "__main__":
    for fn in (test_entrypoint_exports_the_frame_id_sidecar_contract,
               test_missing_checkpoint_is_rejected_before_any_work):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_infer_export_contract: all passed")
