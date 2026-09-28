"""#5 evidence: pixel_weight is the adapter's live double-support sum.

The pristine dataset hard-loaded three date-level ``sub_info.npy`` files from
a data root that no longer exists (the upstream static table).  The ported
version discovers every date directory of the adapter tree, requires the
phase's split dates to be covered, and rejects non-finite / non-positive
weights.  The values it consumes are the ones the mmvp_series FPP adapter
computed on the **double-support** standing frame; this test cross-checks the
loaded numbers against the producer's own report and shows they are not the
upstream static values.

Run directly:

    python tests/test_pixel_weight_source.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

FPP_ROOT = Path(__file__).resolve().parents[1]
if str(FPP_ROOT) not in sys.path:
    sys.path.insert(0, str(FPP_ROOT))

from lib.config.config import config_cont  # noqa: E402
from lib.Dataset.PressDataset.PED_tempKPCont import ContDataset  # noqa: E402

CONFIG = FPP_ROOT / "configs/temporalKPSMPLCont_series5_mlp.yaml"
# documented upstream static values (registry FPP #5); the ported code must
# not reproduce them
UPSTREAM_STATIC = {"S10": 992.17, "S11": 603.48}


def _report():
    """The adapter's own metadata report, addressed through the config."""
    return Path(_cfg().dataset.datadir) / "fpp_metadata_report.json"


def _cfg(datadir=None, tv_fn=None, phase_agnostic=True):
    cfg = config_cont()
    cfg.load(str(CONFIG))
    cfg = cfg.get_cfg()
    cfg.defrost()
    if datadir is not None:
        cfg.dataset.datadir = str(datadir)
    if tv_fn is not None:
        cfg.dataset.tv_fn = str(tv_fn)
    cfg.dataset.aug.is_aug = False
    cfg.freeze()
    return cfg


def test_real_tree_weights_are_positive_and_cover_the_split():
    for phase in ("train", "val", "test"):
        cfg = _cfg()
        dataset = ContDataset(cfg.dataset, phase)
        split = np.load(cfg.dataset.tv_fn, allow_pickle=True).item()
        for date, subs in split[phase].items():
            assert date in dataset.sub_info, f"{phase}: date {date} not covered"
            for sub in subs:
                weight = dataset.sub_info[date][sub]["weight"]
                assert np.isfinite(weight), f"{date}/{sub}: weight not finite"
                assert weight > 0, f"{date}/{sub}: weight {weight} not positive"


def test_weights_match_the_producer_report_and_the_double_support_choice():
    report = json.loads(_report().read_text())
    standing = report["standing"]
    # the standing frame is picked by the double-support rule, not by a
    # single-support fallback (the adapter records both choices explicitly)
    selection_block = report.get("standing_selection")
    if isinstance(selection_block, dict):
        assert "double" in selection_block["rule"].lower()
        assert selection_block["subjects_single_support_fallback"] == []
        assert set(selection_block["subjects"].values()) == {"double_support"}
    seen = set()
    for phase in ("train", "val", "test"):
        dataset = ContDataset(_cfg().dataset, phase)
        for date, subs in dataset.sub_info.items():
            for sub, info in subs.items():
                if (date, sub) in seen:
                    continue
                seen.add((date, sub))
                assert sub in standing, f"{sub} missing from the adapter report"
                assert np.isclose(info["weight"], standing[sub]["weight"]), (
                    f"{date}/{sub}: dataset weight {info['weight']} != adapter "
                    f"report {standing[sub]['weight']}")
                assert standing[sub]["standing_selection"] == "double_support"
                assert standing[sub]["date"] == date
    assert len(seen) == 9, f"expected the 9 adapter subjects, saw {sorted(seen)}"


def test_upstream_static_values_are_not_reproduced():
    dataset = ContDataset(_cfg().dataset, "train")
    for sub, static in UPSTREAM_STATIC.items():
        date = "20260808"  # S10/S11 both record on 20260808
        live = dataset.sub_info[date][sub]["weight"]
        assert live / static > 50, (
            f"{sub}: live weight {live} is not clearly apart from the "
            f"upstream static {static}")


def _write_sub_info(root, date, subs):
    date_dir = root / date
    date_dir.mkdir(parents=True, exist_ok=True)
    np.save(date_dir / "sub_info.npy", subs)


def _write_split(path, phases):
    np.save(path, phases)


def _fixture(tmp, sub_info, split):
    root = Path(tmp) / "datadir"
    root.mkdir()
    for date, subs in sub_info.items():
        _write_sub_info(root, date, subs)
    split_path = Path(tmp) / "split.npy"
    _write_split(split_path, split)
    return root, split_path


def _expect(exc, fn, needle):
    try:
        fn()
    except exc as err:
        assert needle in str(err), f"unexpected message: {err}"
        return
    raise AssertionError(f"{exc.__name__} not raised ({needle})")


def test_missing_tree_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "empty"
        root.mkdir()
        split = Path(tmp) / "split.npy"
        _write_split(split, {"train": {"20260808": {"S10": {"S1": ["000000.npy"]}}}})
        _expect(FileNotFoundError,
                lambda: ContDataset(_cfg(root, split).dataset, "train"),
                "No date-level sub_info.npy")


def test_split_date_without_sub_info_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root, split = _fixture(
            tmp,
            {"20260808": {"S10": {"weight": 100.0}}},
            {"train": {"20990101": {"S10": {"S1": ["000000.npy"]}}}})
        _expect(FileNotFoundError,
                lambda: ContDataset(_cfg(root, split).dataset, "train"),
                "no sub_info.npy")


def test_split_subject_without_weight_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root, split = _fixture(
            tmp,
            {"20260808": {"S10": {"weight": 100.0}}},
            {"train": {"20260808": {"S99": {"S1": ["000000.npy"]}}}})
        _expect(KeyError,
                lambda: ContDataset(_cfg(root, split).dataset, "train"),
                "no entry for S99")


def test_non_positive_or_non_finite_weight_is_rejected():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with tempfile.TemporaryDirectory() as tmp:
            root, split = _fixture(
                tmp,
                {"20260808": {"S10": {"weight": bad}}},
                {"train": {"20260808": {"S10": {"S1": ["000000.npy"]}}}})
            _expect(ValueError,
                    lambda: ContDataset(_cfg(root, split).dataset, "train"),
                    "positive finite sum")


if __name__ == "__main__":
    for fn in (test_real_tree_weights_are_positive_and_cover_the_split,
               test_weights_match_the_producer_report_and_the_double_support_choice,
               test_upstream_static_values_are_not_reproduced,
               test_missing_tree_is_rejected,
               test_split_date_without_sub_info_is_rejected,
               test_split_subject_without_weight_is_rejected,
               test_non_positive_or_non_finite_weight_is_rejected):
        fn()
        print(f"ok: {fn.__name__}")
    print("test_pixel_weight_source: all passed")
