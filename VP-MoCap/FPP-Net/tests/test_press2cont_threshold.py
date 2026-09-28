"""#1 evidence: press2Cont threshold at the FPP-Net call site.

Ruling (2026-09-28): the contact GT is binarised at the native call-site
threshold 0.5.  ``InsoleModule.press2Cont`` keeps its 0.7 function default
(upstream code), but the dataset must never fall back to it — the 0.7 detour
changes the contact label distribution, and the vertex GT (``contact_smpl``)
derived from it feeds both the training loss and the exported
``contact_smpl.gt`` sidecar.

Run directly:

    python tests/test_press2cont_threshold.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

FPP_ROOT = Path(__file__).resolve().parents[1]
if str(FPP_ROOT) not in sys.path:
    sys.path.insert(0, str(FPP_ROOT))

from lib.Dataset.InsoleModule import InsoleModule  # noqa: E402

DATASET = FPP_ROOT / "lib/Dataset/PressDataset/PED_tempKPCont.py"


def test_call_site_threshold_is_explicit_05():
    source = DATASET.read_text(encoding="utf-8")
    assert "press2Cont(insole_press, sub_weight, th=0.5)" in source, (
        "the dataset must binarise contact at the native call-site threshold 0.5")
    assert "th=0.7" not in source, "the 0.7 detour must not be reintroduced"


def test_function_default_stays_upstream():
    import inspect

    signature = inspect.signature(InsoleModule.press2Cont)
    assert signature.parameters["th"].default == 0.7, (
        "press2Cont keeps its upstream 0.7 default; only the call site rules 0.5")


def test_05_and_07_label_sets_differ_as_derived():
    module = InsoleModule()
    # sigmoid norm: sigmoid(x / (w/pixel_num) - 1); contact iff the sigmoid
    # exceeds th, i.e. x > (1 + logit(th)) * w/pixel_num.
    weight = 26935.169921875  # S13 double-support weight, adapter_v1
    unit = weight / module.pixel_num
    x_07 = (1.0 + math.log(0.7 / 0.3)) * unit
    x_05 = unit

    insole = np.zeros((2, 31, 11), dtype=np.float32)
    insole[0, 0, 0] = 0.5 * x_05                    # below both thresholds
    insole[0, 0, 1] = 0.5 * (x_05 + x_07)           # only th=0.5 contacts
    insole[0, 0, 2] = 1.5 * x_07                    # both contact

    at_05 = module.press2Cont(insole, weight, th=0.5)
    at_07 = module.press2Cont(insole, weight, th=0.7)

    assert at_05.shape == (31, 22) == at_07.shape
    assert at_05[0, 0] == 0 and at_07[0, 0] == 0
    assert at_05[0, 1] == 1 and at_07[0, 1] == 0, (
        "the mid-band cell is where the ruling is observable")
    assert at_05[0, 2] == 1 and at_07[0, 2] == 1
    assert at_05.sum() > at_07.sum(), "0.5 labels are a strict superset of 0.7"


if __name__ == "__main__":
    for fn in (test_call_site_threshold_is_explicit_05,
               test_function_default_stays_upstream,
               test_05_and_07_label_sets_differ_as_derived):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_press2cont_threshold: all passed")
