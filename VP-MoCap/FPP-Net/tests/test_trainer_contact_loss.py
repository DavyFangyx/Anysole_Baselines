"""#2 evidence: the effective contact loss target is the 192-dim vertex GT.

Upstream keeps ``w_cont*BCE(pred_cont, contact_label)`` live while the head
emits 192 values and ``contact_label`` holds the 484 masked insole pixels —
the shapes cannot broadcast.  The retired line ``MSE(pred, contact_smpl)``
was the matching 192-dim target; the ruling uses BCE on that target.

This test drives the real trainer (real loss code path) with one synthetic
batch and the real network, so the assertion is about the executed line, not
about a source-text match.

Run directly:

    python tests/test_trainer_contact_loss.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

FPP_ROOT = Path(__file__).resolve().parents[1]
if str(FPP_ROOT) not in sys.path:
    sys.path.insert(0, str(FPP_ROOT))

from lib.Networks import make_network  # noqa: E402
from lib.Trainer import make_trainer  # noqa: E402

BATCH = 3
SEQLEN = 5
N_JOINTS = 26


class _Recorder:
    """Minimal recorder; keeps the test free of tensorboardX/checkpoints."""

    def __init__(self):
        self.epochs_logged = 0
        self.checkpoint_path = ""
        self.name = "test"

    def init(self):
        pass

    def logPressNetTensorBoard(self, log):
        pass

    def log(self, log):
        self.epochs_logged += 1


def _one_batch():
    torch.manual_seed(0)
    return {
        "case_name": ["20260810/S14/S14073/%06d" % (100 + i) for i in range(BATCH)],
        "keypoints": torch.randn(BATCH, SEQLEN, N_JOINTS, 3),
        "insole": torch.rand(BATCH, 484),
        # kept at its native 484 width on purpose: if the trainer consumed
        # this tensor the step would raise on the 192-vs-484 mismatch
        "contact_label": torch.randint(0, 2, (BATCH, 484)).float(),
        "contact_smpl": torch.randint(0, 2, (BATCH, 192)).float(),
    }


def _network_cfg():
    return SimpleNamespace(
        module="lib.Networks.TemporalKpSMPLNet_Series_mlp",
        path=str(FPP_ROOT / "lib/Networks/TemporalKpSMPLNet_Series_mlp.py"),
        seqlen=SEQLEN)


def test_head_outputs_192_and_target_is_192():
    network = make_network.make_network(_network_cfg())
    with torch.no_grad():
        press, cont = network(keypoints=torch.randn(BATCH, SEQLEN, N_JOINTS, 3))
    assert press.shape == (BATCH, 484)
    assert cont.shape == (BATCH, 192), "cont head stays at the 2x96 vertex GT"


def test_484_target_cannot_serve_the_192_head():
    pred = torch.rand(BATCH, 192)
    with torch.no_grad():
        try:
            F.binary_cross_entropy(pred, torch.rand(BATCH, 484))
        except (ValueError, RuntimeError):
            return
    raise AssertionError("484-wide target must not be consumable by the head")


def _run_step(batch):
    """One deterministic training step (same init and dropout stream)."""
    torch.manual_seed(7)
    network = make_network.make_network(_network_cfg())
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    opts = SimpleNamespace(lr=1e-4, num_train_epochs=200, epochs=1,
                           w_press=0.2, w_cont=0.8,
                           module="lib.Trainer.trainer_tempkpSMPLCont_mse",
                           path=str(FPP_ROOT / "lib/Trainer/trainer_tempkpSMPLCont.py"))
    trainer = make_trainer.make_trainer([batch], network, optimizer, _Recorder(),
                                        None, opts)
    torch.manual_seed(11)
    trainer.train()
    return {k: v.clone() for k, v in network.state_dict().items()}


def test_one_batch_step_uses_the_vertex_target():
    batch = _one_batch()
    reference = _run_step(batch)

    # same inputs, different 484-dim contact_label: with the vertex target
    # the step cannot see it, so the updated weights are identical
    other_label = {k: (v.clone() if torch.is_tensor(v) else list(v))
                   for k, v in batch.items()}
    other_label["contact_label"] = 1.0 - other_label["contact_label"]
    same = _run_step(other_label)
    for key in reference:
        assert torch.equal(reference[key], same[key]), (
            f"parameter {key} changed with contact_label: the 484-dim tensor "
            "must not participate in the contact loss")

    # control: the vertex target itself does drive the step
    other_target = {k: (v.clone() if torch.is_tensor(v) else list(v))
                    for k, v in batch.items()}
    other_target["contact_smpl"] = 1.0 - other_target["contact_smpl"]
    different = _run_step(other_target)
    assert any(not torch.equal(reference[key], different[key])
               for key in reference), (
        "the contact_smpl target must move the parameters")


if __name__ == "__main__":
    for fn in (test_head_outputs_192_and_target_is_192,
               test_484_target_cannot_serve_the_192_head,
               test_one_batch_step_uses_the_vertex_target):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_trainer_contact_loss: all passed")
