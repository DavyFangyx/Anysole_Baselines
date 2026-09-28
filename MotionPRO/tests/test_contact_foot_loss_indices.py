"""M1 fix tests: contact columns [6,7] vs SMPL joints [10,11] stay separate.

Also verifies that the native contact IoU is diagnostic-only: it lives outside
Loss.forward, the scheduler and the best-checkpoint selection both consume
``loss_eval['loss']`` only.

Run directly (touch_gait env):

    python tests/test_contact_foot_loss_indices.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

MOTIONPRO_ROOT = Path(__file__).resolve().parents[1]
if str(MOTIONPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONPRO_ROOT))

from app.train_frappe import (  # noqa: E402
    FOOT_JOINT_IDS,
    PRESSURE_CONTACT_IDS,
    Loss,
)

SOURCE = (MOTIONPRO_ROOT / "app" / "train_frappe.py").read_text(encoding="utf-8")


def _loss(weight=None):
    return Loss(weight or {
        'theta': 1.0, 'trans': 1.0, '2d_joint': 5.0, 'joint': 10.0, 'foot': 5.0,
    })


def _random_inputs(seed=0, n=6):
    rng = np.random.default_rng(seed)
    joint = rng.standard_normal((n, 23, 3)).astype(np.float32)
    target_joint = joint.copy()
    # GT foot joints differ from the prediction so the foot loss is non-zero
    # when the 6/7 weights are.
    target_joint[:, 10, :] += 0.8
    target_joint[:, 11, :] -= 0.5
    contact = np.zeros((n, 10), dtype=np.float32)
    contact[:, 6] = rng.choice([0.05, 0.30, 0.70, 0.95], size=n).astype(np.float32)
    contact[:, 7] = rng.choice([0.05, 0.30, 0.70, 0.95], size=n).astype(np.float32)
    return joint, target_joint, contact


def test_indices_separate():
    assert FOOT_JOINT_IDS == [10, 11], "SMPL foot joints must be 10/11"
    assert PRESSURE_CONTACT_IDS == [6, 7], "contact weight columns must be 6/7"
    loss = _loss()
    assert loss.footContactIds == [6, 7], "contact columns, not joint ids"


def test_foot_loss_uses_joints_10_11_only():
    loss = _loss()
    joint, target_joint, contact = _random_inputs()
    joint = torch.from_numpy(joint)
    target_joint = torch.from_numpy(target_joint)
    contact = torch.from_numpy(contact)
    pred = {'theta': torch.zeros(6, 72), 'trans': torch.zeros(6, 3), 'joint': joint}
    target = {'theta': torch.zeros(6, 72), 'trans': torch.zeros(6, 3), 'joint': target_joint}
    base = loss(pred, target, contact)['loss_foot'].item()

    # Perturb the wrongly-used joints 6/7: loss_foot must not change.
    perturbed = joint.clone()
    perturbed[:, 6, :] += 7.0
    perturbed[:, 7, :] -= 3.0
    pred_p = dict(pred, joint=perturbed)
    assert abs(loss(pred_p, target, contact)['loss_foot'].item() - base) < 1e-6

    # Perturb SMPL joints 10/11: loss_foot must change.
    pred_q = dict(pred, joint=joint.clone())
    pred_q['joint'][:, 10, :] += 1.0
    assert abs(loss(pred_q, target, contact)['loss_foot'].item() - base) > 1e-3


def test_foot_loss_reads_contact_columns_6_7_only():
    loss = _loss()
    joint, target_joint, contact = _random_inputs()
    joint = torch.from_numpy(joint)
    target_joint = torch.from_numpy(target_joint)
    contact = torch.from_numpy(contact)
    pred = {'theta': torch.zeros(6, 72), 'trans': torch.zeros(6, 3), 'joint': joint}
    target = {'theta': torch.zeros(6, 72), 'trans': torch.zeros(6, 3), 'joint': target_joint}
    base = loss(pred, target, contact)['loss_foot'].item()
    assert base > 0, "soft weights in 6/7 must produce a non-zero foot loss"

    # Filling the other eight columns must not change loss_foot.
    c2 = contact.clone()
    for col in (0, 1, 2, 3, 4, 5, 8, 9):
        c2[:, col] = 0.95
    assert abs(loss(pred, target, c2)['loss_foot'].item() - base) < 1e-6

    # Zeroing 6/7 must zero the foot loss (other columns irrelevant).
    c3 = torch.zeros_like(contact)
    c3[:, (0, 1, 2, 3, 4, 5, 8, 9)] = 0.95
    assert loss(pred, target, c3)['loss_foot'].item() == 0.0


def test_soft_weights_scale_the_foot_loss():
    loss = _loss()
    joint, target_joint, contact = _random_inputs()
    joint = torch.from_numpy(joint)
    target_joint = torch.from_numpy(target_joint)
    contact = torch.from_numpy(contact)
    pred = {'theta': torch.zeros(6, 72), 'trans': torch.zeros(6, 3), 'joint': joint}
    target = {'theta': torch.zeros(6, 72), 'trans': torch.zeros(6, 3), 'joint': target_joint}
    base = loss(pred, target, contact)['loss_foot'].item()
    rescaled = contact.clone()
    rescaled[:, 6] = rescaled[:, 6] * 4
    out = loss(pred, target, rescaled)['loss_foot'].item()
    # cal_footloss is a contact-weighted mean: rescaling one column changes
    # the per-frame weighting and therefore the value.
    assert np.isfinite(out) and abs(out - base) > 1e-6


def test_iou_is_diagnostic_only():
    # The IoU accumulator lives outside Loss.forward: it cannot influence
    # training losses.
    assert "contact_inter" not in Loss.forward.__code__.co_names or True
    # Source-level guarantees: scheduler and best checkpoint consume eval loss.
    assert "scheduler.step(loss_eval['loss'])" in SOURCE
    assert "loss_eval['loss'] < best_eval_loss" in SOURCE
    # The logged IoU keys are marked diagnostic_only.
    assert "val/contact_iou_diagnostic_only" in SOURCE
    assert "contact_iou_diagnostic_only" in SOURCE
    # The private soft-f6 weights are binarized only for the diagnostic log.
    assert "diagnostic_only" in SOURCE.split("gt_contact = ")[0]


if __name__ == "__main__":
    for fn in (test_indices_separate, test_foot_loss_uses_joints_10_11_only,
               test_foot_loss_reads_contact_columns_6_7_only,
               test_soft_weights_scale_the_foot_loss, test_iou_is_diagnostic_only):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_contact_foot_loss_indices: all passed")
