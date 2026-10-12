"""Baseline-side shared data/evaluation protocols (2026-09-26 结构裁定).

- ``motion_io`` / ``bvh_aligner``: raw motion/BVH readers (canonical home).
- ``protocols``: joint tables, skeleton trees, protocol identifiers,
  pelvis/foot joint conventions, forward axes.
- ``gt_loading``: GT loaders on the canonical session grid, prediction
  archive readers, split/manifest helpers.
- ``capabilities``: capability schema, metric necessary-inputs, provenance
  gating.
- ``solver``: the protocol-aware metric assembly (formulas live only in
  ``anysole.utils.metrics``) and the V2T nine-metric solver.
- ``evaluate``: the baseline evaluation entry — predictions + this solver →
  per-session and aggregate results under ``results/baselines/<Model>/metrics/``.

``results_display`` consumes these modules (``compare_core`` re-exports them);
it is a consumer, not the place where baseline metrics are produced.

2026-10-12 (H13): became a real tracked directory of the Baselines repo,
replacing the ``Baselines_old/utils`` tree and the ``Baselines/utils``
symlink bridge; package name unchanged (``Baselines.utils``).
"""
