# Changelog

## Unreleased (branch feat/spect-alignment-qc)
- SPECT-to-anatomy alignment check (`alignment_qc` in results.json): translation fit against the simulated uptake inside caudate, caudate plus anterior putamen and whole striatum masks. Warns above 3 mm. Cannot separate SPECT misalignment from a caudate-model offset.
- Combined SPECT shift grid (±4 mm, all axes) in robustness.
- Profile figure scaled by voxel-weighted anterior mean per side, with the lowest normal-gradient curve drawn.
- `<SYNTHSEG_HOME>` placeholder via `DATSCAN_SYNTHSEG_HOME`.

## 0.1.0 (2026-11-07)
- Optional anatomical segmenters (SynthSeg 2.0, FastSurfer), with SynthSeg as the reference-case default.
- Normal-gradient null as primary comparator. Uniform-binding null kept as a lenient bound.
- Axis-aligned analysis grid, loud QC gates, randomised phantom with two-sided controls.
