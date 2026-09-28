"""The public evaluator must never read MotionPRO's private contact.

Static test: no public ``Baselines/utils`` module may reference the
MotionPRO model-input contact.npy or read soft-f6 loss weights as a metric
input.  (The contact metrics that exist there consume FPP-Net per-vertex
contact predictions and the shared per-foot GT, never MotionPRO's file.)

Run directly (touch_gait env):

    python tests/test_public_evaluator_no_motionpro_contact.py
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
UTILS_ROOT = REPO_ROOT / "Baselines" / "utils"

FORBIDDEN = (
    "model_inputs/MotionPRO",
    "model-input://MotionPRO",
    "motionpro_private_soft_f6",
    "contact.npy",
    "foot_loss_weight",
)


def test_public_utils_do_not_read_motionpro_contact():
    hits = []
    for path in sorted(UTILS_ROOT.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token in text:
                hits.append((path.name, token))
    assert not hits, (
        f"public Baselines/utils modules reference MotionPRO private contact: {hits}")


def test_motionpro_contact_stays_private():
    # The private files live under the MotionPRO model-input tree and are
    # produced by the MotionPRO adapter only.
    adapter = (REPO_ROOT / "AnysoleWorkspace/tool/adapters/MotionPRO/"
               "adapter.py").read_text(encoding="utf-8")
    assert "motionpro_private_soft_f6_foot_loss_weight" in adapter
    assert "formal_metric_input" in adapter and "native_diagnostic_only" in adapter


if __name__ == "__main__":
    for fn in (test_public_utils_do_not_read_motionpro_contact,
               test_motionpro_contact_stays_private):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_public_evaluator_no_motionpro_contact: all passed")
