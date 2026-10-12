"""Capability contract for the baseline evaluation side.

Capability contract (评估整改 00 §5): what a model really outputs, not what
"exists as an array" in an archive.  Registry entries may use either the
00-schema names (direct_joint_rotations / predicted_surface /
pressure_prediction / contact_prediction) or the 03-schema aliases
(joint_rotations / surface / pressure / contact); both are accepted.

Metric eligibility is determined by the declared generation capability and
the presence of the corresponding prediction and GT fields. Provenance
fields remain available for audit, but they are not a second metric policy:
``sources`` and ``approved`` must not make an otherwise generated metric
appear or disappear.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]  # utils/ -> Baselines/ -> repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Baselines.utils.solver import METRICS, V2T_METRICS

CAPABILITY_FIELDS = (
    "joint_positions", "direct_joint_rotations", "root_translation",
    "root_rotation", "predicted_surface", "predicted_shape",
    "pressure_prediction", "contact_prediction",
)
CAPABILITY_ALIASES = {
    "joint_rotations": "direct_joint_rotations",
    "surface": "predicted_surface",
    "pressure": "pressure_prediction",
    "contact": "contact_prediction",
}

# Historical source labels are retained for audit/reporting only. They are
# intentionally not consulted by ``applicable_metrics``/``applicable_for``.
DENIED_SOURCES = (
    "ground_truth_fallback", "template_reconstruction", "method_fitting",
    "derived_for_visualization",
)

# Necessary generated outputs per public metric (评估整改 01 §4.2).
METRIC_REQUIRES = {
    "mpjpe_mm": ("joint_positions",),
    "pa_mpjpe_mm": ("joint_positions",),
    "w_mpjpe100_mm": ("joint_positions", "root_translation"),
    "wa_mpjpe100_mm": ("joint_positions", "root_translation"),
    "root_ate_mm": ("root_translation",),
    "root_rte_percent": ("root_translation",),
    "root_orientation_deg": ("root_rotation",),
    "root_orientation_drift_deg": ("root_rotation",),
    "mpjae_deg": ("direct_joint_rotations",),
    "accel_error_m_s2": ("joint_positions",),
    "jitter_pred_m_s3": ("joint_positions",),
    "jitter_gt_m_s3": ("joint_positions",),
    "foot_sliding_mm": ("joint_positions",),
    "pve_mm": ("predicted_surface",),
    "contact_f1": ("contact_prediction",),
    **{name: ("pressure_prediction",) for name in V2T_METRICS},
    # Contact level of the V2T hierarchy requires a trained, explicitly
    # exported contact prediction (FPP-Net contact head); overrides the
    # pressure_prediction spread above for these keys.
    "contact_smpl_mse": ("contact_prediction",),
    "contact_smpl_bce": ("contact_prediction",),
}


def read_capabilities(capabilities) -> dict[str, bool]:
    """Normalize a capability mapping to the 00-schema field names."""
    source = capabilities if isinstance(capabilities, dict) else {}
    caps: dict[str, bool] = {}
    for field in CAPABILITY_FIELDS:
        value = source.get(field, source.get(CAPABILITY_ALIASES.get(field, field)))
        caps[field] = bool(value)
    return caps


def applicable_metrics(capabilities, *, approved: Any = ()) -> set[str]:
    """Metrics whose necessary declared inputs are available.

    ``approved`` is accepted for compatibility with old callers but is
    intentionally ignored. A protocol exception must be represented by a
    real generated capability and matching prediction field; it may not
    inject a GT/template value into evaluation.
    """
    caps = read_capabilities(capabilities)
    return {
        metric
        for metric, needs in METRIC_REQUIRES.items()
        if all(caps.get(need, False) for need in needs)
    }


def denied_capabilities(entry: dict[str, Any], archive: dict[str, Any] | None = None) -> set[str]:
    """Return provenance findings for audit metadata only.

    The returned set is not used by ``applicable_for``. Metric eligibility is
    determined by the generation capability and actual solver inputs.
    """
    denied: set[str] = set()
    sources = entry.get("sources")
    if isinstance(sources, dict):
        for field in CAPABILITY_FIELDS:
            value = sources.get(field)
            for alias, canonical in CAPABILITY_ALIASES.items():
                if canonical == field and field not in sources:
                    value = sources.get(alias)
                    break
            if str(value or "").strip().lower() in DENIED_SOURCES:
                denied.add(field)
    meta = archive or {}
    if str(meta.get("provenance_source_type", "")).strip() in DENIED_SOURCES:
        # Archive-level fallback marks the whole payload as not a model output.
        denied.update(CAPABILITY_FIELDS)
    field_to_cap = {
        "root_translation_source": "root_translation",
        "surface_source": "predicted_surface",
        "shape_source": "predicted_shape",
        "pressure_source_type": "pressure_prediction",
    }
    for field, cap in field_to_cap.items():
        if str(meta.get(field, "")).strip() in DENIED_SOURCES:
            denied.add(cap)
    return denied


def applicable_for(caps: dict[str, bool], denied: set[str],
                   approved: Any = ()) -> set[str]:
    """Metrics this row may compute from generated capabilities and GT.

    ``denied`` and ``approved`` are legacy compatibility arguments. They are
    deliberately ignored here: provenance metadata is recorded for audit,
    while applicability is decided by the generation contract and the
    solver's actual-field checks.
    """
    del denied, approved
    return applicable_metrics(caps)


def mask_metrics(values: dict[str, Any], applicable: set[str],
                 reasons: dict[str, str], *, reason: str = "not_applicable") -> None:
    """Blank every metric outside the applicable set and record the reason.

    Missing values stay NaN in the row and are written as ``—``/blank by the
    CSV/PNG writers — never as 0.
    """
    for key in METRICS:
        if key not in applicable:
            values[key] = float("nan")
            reasons.setdefault(key, reason)
