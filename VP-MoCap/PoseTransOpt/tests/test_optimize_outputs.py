"""PoseTransOpt optimize entrypoint path/output tests (registration #8).

``app/optimize.py`` no longer writes beside its input: it resolves ``://`` URIs
through the workspace resolver, derives the canonical output root
``work://VP-MoCap/v1/pose_optimization/<date>/<subject>/<session>``, records the
shared frame ids in ``opt_result.pth`` and writes an ``artifact.json`` sidecar,
with the (slow, pyrender/o3d based) visualization switched off by
``task.write_visualization=false``.

This file covers the path layer.  The artifact/pth payload is covered by the
one-frame fit smoke (see the T2 delivery report), which needs the native
``mmvp`` env.

Run directly:

    /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python \
        tests/test_optimize_outputs.py

``touch_gait`` lacks icecream/open3d/pyrender/human_body_prior, so this file
installs minimal stubs for them; no optimization code is executed here.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

POSETRANSOPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = POSETRANSOPT_ROOT.parents[2]
if str(POSETRANSOPT_ROOT) not in sys.path:
    sys.path.insert(0, str(POSETRANSOPT_ROOT))


def _stub_module(name: str, **attributes) -> None:
    module = types.ModuleType(name)
    module.__path__ = []  # namespace package, so dotted imports keep working
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _stub_load_model(*args, **kwargs):
    raise RuntimeError("human_body_prior stub: load_model is not available in this test")


def _install_light_stubs() -> None:
    """Provide the native-env extras that the slim test env may lack."""
    try:
        __import__("icecream")
    except ModuleNotFoundError:
        _stub_module("icecream", ic=lambda *args, **kwargs: None)
    for name in ("open3d", "pyrender"):
        try:
            __import__(name)
        except ModuleNotFoundError:
            _stub_module(name)
    try:
        __import__("human_body_prior.tools.model_loader")
    except ModuleNotFoundError:
        _stub_module("human_body_prior")
        _stub_module("human_body_prior.tools")
        _stub_module("human_body_prior.tools.model_loader", load_model=_stub_load_model)
        _stub_module("human_body_prior.models")
        _stub_module("human_body_prior.models.vposer_model", VPoser=object)


_install_light_stubs()

from app.optimize import resolve_cfg_paths, resolve_path  # noqa: E402

WORKSPACE = Path(os.environ.get("ANYSOLE_WORKSPACE", REPO_ROOT / "AnysoleWorkspace"))


def test_plain_and_absolute_paths():
    assert resolve_path("") == ""
    assert resolve_path(None) == ""
    # relative paths stay anchored on the baseline root (native config style)
    assert resolve_path("models/V02_05") == str(POSETRANSOPT_ROOT / "models/V02_05")
    absolute = "/tmp/posetransopt_abs"
    assert resolve_path(absolute) == absolute


def test_workspace_uris_resolve_to_the_workspace_roots():
    assert resolve_path("workspace://model_inputs") == str(WORKSPACE / "model_inputs")
    assert resolve_path("asset://third_party/smpl/SMPL_NEUTRAL.pkl") == str(
        WORKSPACE / "assets/third_party/smpl/SMPL_NEUTRAL.pkl")
    assert resolve_path("model-input://PoseTransOpt/adapter_v1") == str(
        WORKSPACE / "model_inputs/PoseTransOpt/adapter_v1")


def test_canonical_work_root_matches_the_literal_fallback():
    """The work:// URI and the in-code fallback must agree on one location."""
    resolved = resolve_path("work://VP-MoCap/v1/pose_optimization/20260804/S5/S5011")
    literal = str(REPO_ROOT / "AnysoleWorkspace" / "work" / "VP-MoCap" / "v1"
                  / "pose_optimization" / "20260804" / "S5" / "S5011")
    assert Path(resolved) == Path(literal), (resolved, literal)
    assert Path(resolved).parts[-3:] == ("20260804", "S5", "S5011")


def test_resolve_cfg_paths_resolves_inputs_and_checkpoints_only():
    cfg = {"task": {"input_path_base": "workspace://model_inputs/PoseTransOpt/adapter_v1",
                    "scene_rgbd": "/abs/scene.npy",
                    "thres_contact": 15, "max_frames": 0},
           "method": {"smpl_file": "asset://third_party/smpl/SMPL_NEUTRAL.pkl",
                      "smpl_male_file": "asset://third_party/VP-MoCap/smpl/SMPL_MALE.pkl",
                      "vposer_path": "models/V02_05", "max_iter": 2001}}
    out = resolve_cfg_paths(cfg)
    assert out["task"]["input_path_base"] == str(WORKSPACE / "model_inputs/PoseTransOpt/adapter_v1")
    assert out["task"]["scene_rgbd"] == "/abs/scene.npy"
    assert out["method"]["smpl_file"] == str(WORKSPACE / "assets/third_party/smpl/SMPL_NEUTRAL.pkl")
    assert out["method"]["smpl_male_file"] == str(
        WORKSPACE / "assets/third_party/VP-MoCap/smpl/SMPL_MALE.pkl")
    assert out["method"]["vposer_path"] == str(POSETRANSOPT_ROOT / "models/V02_05")
    # non-path knobs are untouched
    assert out["task"]["thres_contact"] == 15
    assert out["task"]["max_frames"] == 0
    assert out["method"]["max_iter"] == 2001
    # config is resolved in place (hydra hands main() the mutable dict)
    assert cfg["task"]["scene_rgbd"] == "/abs/scene.npy"


def test_input_session_uri_resolves_to_a_real_t1_package():
    session = WORKSPACE / "model_inputs/PoseTransOpt/adapter_v1/20260804/S5/S5011"
    if not (session / "join_manifest.json").is_file():
        print(f"  skip: no T1 session at {session}")
        return
    assert Path(resolve_path(
        "model-input://PoseTransOpt/adapter_v1/20260804/S5/S5011")) == session


TESTS = (
    test_plain_and_absolute_paths,
    test_workspace_uris_resolve_to_the_workspace_roots,
    test_canonical_work_root_matches_the_literal_fallback,
    test_resolve_cfg_paths_resolves_inputs_and_checkpoints_only,
    test_input_session_uri_resolves_to_a_real_t1_package,
)


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"ok: {test.__name__}")
    print("test_optimize_outputs: all passed")
