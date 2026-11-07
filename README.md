# datscan-mri

MRI-guided regional analysis of ¹²³I-FP-CIT (DaTscan) SPECT/CT.

> **Research tool. Not a medical device. Not for diagnosis.** It has not been
> validated against a normal database, and its output must not be used to make
> or change clinical decisions. Interpretation of a DaTscan belongs to the
> reading physician. The visual read is primary, and semi-quantification is an
> adjunct (DaTscan SmPC).

## What it does

Standard DaTscan quantification (for example DaTQUANT) fits a template striatum
to the SPECT and compares binding ratios with a normal database. That works
poorly when the brain is atypical, for example with marked atrophy or enlarged
ventricles. A template VOI can then sit partly in CSF or white matter, and the
normal database no longer describes the anatomy.

This tool uses the patient's own MRI instead:

1. **io**: converts the SPECT, the low-dose CT from the same SPECT/CT session and a 3-D T1 MRI (MPRAGE) from DICOM. A folder holding several series must be disambiguated with `--spect-series` / `--ct-series` / `--mri-series` (the run stops and lists them).
2. **register**: rigidly registers the MRI to the CT by Mattes mutual information. SPECT and CT share a frame, so the MRI then lines up with the SPECT. N4 bias correction follows, and everything is resampled onto one 1-mm grid that is always axis-aligned in DICOM LPS (a CT stored with flipped or oblique axes is resampled, never trusted). A QC overlay figure and a CT-vs-MRI ventricle Dice are written.
3. **segment**: outlines putamen and caudate on the MRI, by default with the semi-automatic row-by-row threshold method (below) or, optionally, with an established anatomical segmenter (SynthSeg 2.0, or FastSurfer; the reference case uses SynthSeg since v3), and builds a full region map (CSF, white matter, cortical grey, deep grey, occipital-grey and cerebellar reference regions, anterior/posterior putamen split at the coronal plane of the anterior commissure, AC).
4. **profile**: specific binding ratio, SBR = (VOI − reference) / reference, per VOI, the uptake along each putamen in 2-mm coronal slabs from the AC, and the posterior/anterior putamen ratio (P/A). Only voxels inside the SPECT's reconstructed support count (zero padding is excluded).
5. **simulate**: two null images on the same anatomy, both with no disease-related posterior loss, measured exactly like the patient:
   * **normal gradient (primary comparator)**: putamen SBR falls linearly front to back so the true P/A is 0.84 (swept 0.80-0.88). 0.84 is the posterior/anterior putamen ratio of the DaTQUANT normal values for age 65 in Grisanti 2023 (PMC10338803, Table 1: 2.09/2.49 left, 2.10/2.51 right). That is a different pipeline, template and age, so only the order of magnitude transfers.
   * **uniform binding (lenient upper bound)**: the same SBR throughout the putamen. Real normal putamen is not uniform, so this null overstates what a normal scan would show.

   The simulated binding sits in an **anatomical putamen**, not in the measurement VOI: the VOI's grey-matter core (white matter removed when the VOI contains any), shrunk to at most 5 mL per side (4, 5, 6 mL and uncapped are all swept). The image is blurred with an isotropic Gaussian and then measured through the VOI exactly as applied to the patient, so a generous VOI lowers the expected ratio the same way it lowers the measured one. Expected ranges cover FWHM 9-15 mm, two binding levels, three background-activity assumptions and the four anatomical sizes.
6. **decision**: a posterior loss is called only if the measured P/A is below the lowest normal-gradient expectation by more than 0.05.
7. **sweeps**: the measurement VOI is grown or shrunk by −2 to +6 mm (whole putamen, and posterior VOI only) with the simulated binding held fixed, the anterior/posterior split is moved ±4 mm, and the putamen thresholds are shifted (lo ±10, hi ±6, which rebuilds both the VOI and the anatomical putamen). The gap between measured and expected is reported for every variant.
8. **pvc**: geometric transfer matrix (GTM) partial-volume correction at 9, 11 and 13 mm FWHM (assumes uniform activity within each VOI).
9. **robustness**: 12 rigid SPECT-vs-VOI perturbations (±2 mm, ±2° per axis) and a 1-mm VOI erosion (measured ratio only).
10. **report**: `results.json` and five figures (registration overlay, segmentation, profile vs both simulations, ratio table, VOI sweep).
11. **export**: de-identified full-field-of-view NIfTIs (MRI, SPECT, CT, labels, and the two simulated nulls as derived layers) on the common grid with the AC at the origin, plus `profile.json` and `geometry.json`. `add-flair` registers a same-session FLAIR into that space.

### Quality gates (the run stops with a non-zero exit code)

| gate | default | override |
|---|---|---|
| AC not given (only a rough centroid-based estimate) | stop | `--allow-estimated-ac` |
| CT-vs-MRI ventricle Dice after registration | < 0.6 stops | `--min-ventricle-dice` |
| putamen VOI per side (a collapsed segmentation) | < 4.0 mL stops | `--min-putamen-ml` |
| any striatal VOI or the occipital reference outside the SPECT support | > 1 % stops | `--max-outside-fov` |
| folder with several DICOM series and none chosen | stop, lists them | `--<modality>-series UID` |

The cerebellar (secondary) reference is not gated. The fraction of each region outside the SPECT support is in `results.json`.

### Putamen segmentation: external anatomical segmenter (optional)

`--segmenter synthseg` (or `"segmenter": {"method": "synthseg"}` in the params
JSON) runs SynthSeg 2.0 (Billot et al., PNAS 2023), the FreeSurfer 7.4.1
`mri_synthseg` script unchanged, on the registered, N4-corrected MRI on the
analysis grid. `--segmenter fastsurfer` runs FastSurferCNN (VINN) instead.
Neither tool is a dependency: each runs as a subprocess from its own
environment (`segmenter.<tool>.home` or `DATSCAN_SYNTHSEG_HOME` /
`DATSCAN_FASTSURFER_HOME`, layouts in `datscan_mri/external.py`), and the
pipeline and tests run without them. `--seg-labels file` uses a precomputed
FreeSurfer-LUT label map (`derived/segmenter_fs_labels_grid.nii.gz` from an
earlier run), and `--compare-labels NAME=file` adds another segmentation as a
comparison.

Only four labels are used: Right/Left-Putamen (51/12) become the putamen VOIs
(then split at the AC like before) and Right/Left-Caudate (50/11) the caudate
VOIs. Accumbens and pallidum are left out. Ventricles (registration QC), the
reference regions and the background compartments still come from the
intensity method. The threshold method is also run as a comparison: its Dice
against the new VOIs goes in `results.json` and it is one row of the
segmentation sweep. Tool, version, model file and model SHA-256 are recorded
under `segmentation` in `results.json`.

Extra quality gates for an external segmenter: any SynthSeg QC score below
0.65 stops the run (`--min-segmenter-qc`), a putamen outside 2.5-8 mL per side
stops it (`qc.external_putamen_ml`), and so does a left/right swap or a
missing structure. The threshold method's 4-mL gate does not apply, because an
atrophic putamen can genuinely be smaller than 4 mL. The threshold sweep is
replaced by the segmentation sweep (the other segmenters, each rebuilding both
the measurement VOI and the simulated putamen).

### Putamen segmentation: threshold method (semi-automatic)

On each axial row (slice z, row y) and each side, the method scans from 50 mm
lateral of the midline toward the midline. It remembers the most medial
"CSF-like" voxel lying 22-46 mm from the midline and away from the ventricles
(insular cortex / sylvian CSF). It then steps 3 mm further medially and
collects the run of voxels in the putamen grey range. The run stops at white
matter or 14 mm from the midline and is at most 13 mm wide. The 3-D cleanup is
an opening, the largest component, a closing and slice-wise hole filling. The
caudate is collected as up to 5 mm of grey lateral to the lateral-ventricle
wall.

Thresholds are expressed in reference intensity units: by default each MRI is
rescaled so the 90th percentile of its eroded intracranial intensities is
190.65 (`intensity.normalize_p90`). The thresholds and boxes were set by eye on
one case and are pinned, with that case's AC, in `configs/reference_case.json`.
`datscan_mri/params.py` documents every parameter. **On a new MRI you must check
`figures/fig2_segmentation.png` and adjust the parameters** with a JSON passed
to `--params`.

## Install

```bash
git clone <this repo> && cd datscan-mri-guided
python3.12 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest          # synthetic-phantom controls, about 3 minutes
```

Python 3.12 or later.

### Optional: SynthSeg (default segmenter for the reference case)

No FreeSurfer install, licence file or Docker is needed. The script, label
lists and weights come from the `freesurfer/freesurfer` repository at tag
`v7.4.1` (weights from its git-annex store, checked against the SHA-256 in the
annex key), and run in a separate Python 3.11 venv:

```bash
H=~/code/_vendor/synthseg-fs7.4.1; mkdir -p $H/models && cd $H
for f in mri_synthseg synthseg_{segmentation,qc}_{labels,names}_2.0.npy synthseg_denoiser_labels_2.0.npy \
         synthseg_topological_classes_2.0.npy synthseg_parcellation_{labels,names}.npy; do
  curl -sfLo models/$f https://raw.githubusercontent.com/freesurfer/freesurfer/v7.4.1/mri_synthseg/$f; done
A=https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects
K=SHA256E-s53079152--f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e.0.h5
curl -sfLo models/synthseg_2.0.h5 $A/bee/241/$K/$K
echo "f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e  models/synthseg_2.0.h5" | shasum -a 256 -c
K=SHA256E-s49878004--fb92e37daa0842bdd8a03bc06a3b36b0d38810e31c714659cedd09119c0abfb7.0.h5
curl -sfLo models/synthseg_qc_2.0.h5 $A/3ac/fe4/$K/$K
echo "fb92e37daa0842bdd8a03bc06a3b36b0d38810e31c714659cedd09119c0abfb7  models/synthseg_qc_2.0.h5" | shasum -a 256 -c
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python tensorflow==2.15.1 "numpy<2" nibabel scipy cython setuptools
uv pip install -p .venv/bin/python --no-build-isolation surfa==0.6.0
```

Then copy `scripts/run_synthseg.py` (a short launcher) to `$H`. It only aliases `np.float128`, which numpy
lacks on Apple silicon, and runs the unmodified script. One run takes about 6
minutes on 8 CPU threads (Apple M5).

FastSurfer (optional second segmenter): clone `Deep-MI/FastSurfer` at tag
`v2.5.4`, put the three `aparc_vinn_*_v2.0.0.pkl` checkpoints from Zenodo
record 10390573 (MD5 listed there) in `checkpoints/`, and install `torch==2.7.*`
and its listed requirements into `.venv`. About 10 minutes on CPU.

## Run on your own DICOMs

```bash
# find the series folders in an export
.venv/bin/datscan-mri inventory /path/to/dicom/export

.venv/bin/datscan-mri run \
  --spect /path/to/attenuation-corrected-SPECT-series \
  --ct    /path/to/low-dose-CT-series-from-the-same-SPECT-CT \
  --mri   /path/to/3D-T1-MPRAGE-series \
  --ac    x,y,z \
  --out   out/ \
  --viewer-out out/viewer
```

* `--ac` is the anterior commissure in DICOM LPS mm in the CT frame. Read it off the CT, or off the MRI after a first run, in any DICOM viewer.
* `--transform file.tfm` reuses a saved registration. `--skip-robustness` and `--skip-sweeps` save time.
* `datscan-mri export-viewer --run out/ --dest viewer/` re-exports from a previous run. `datscan-mri add-flair --run out/ --mri <T1> --flair <FLAIR> --dest viewer/` adds a registered FLAIR and its QC.

Outputs: `results.json`, `figures/`, `derived/` (grid volumes, simulated nulls, the transform), `params_used.json`.

## Reproduce the reference case

The reference case is a 2026 DaTscan SPECT/CT (0.96 mCi, Siemens Symbia,
attenuation-corrected OSEM) that was read clinically as normal, with a 3-D T1
MRI acquired 10.5 months earlier. The data are private and not in this repo.
With the DICOM export at hand:

```bash
.venv/bin/datscan-mri run --spect <SPECT IRAC-AC series> --ct <CT 2.0 mm H31s series> \
  --mri <MPRAGE series> --params configs/reference_case.json --out out/
```

With `configs/reference_case.json` the striatal VOIs come from SynthSeg
(install above). `--segmenter threshold` reproduces the v2 VOIs. The v3 run
took 13 minutes on an Apple M5, about 6 of them in SynthSeg.

### Results, v3 (SynthSeg VOIs, default)

v3 reused the v2 MRI-to-CT transform (`--transform`, ventricle Dice CT vs MRI
0.85), so the registered MRI is voxel-for-voxel identical and only the
segmentation changed. FastSurfer v2.5.4 labels were passed with
`--compare-labels`. P/A uses the occipital-grey reference. "Gap" is measured
P/A minus the lowest normal-gradient expectation. A posterior loss is called
when the gap is below −0.05.

| quantity | v2 threshold, R | v3 SynthSeg, R | v2 threshold, L | v3 SynthSeg, L |
|---|---|---|---|---|
| putamen VOI (anterior + posterior), mL | 8.3 | 4.2 (1.8 + 2.4) | 7.0 | 4.5 (1.8 + 2.8) |
| caudate VOI, mL | 3.0 | 4.8 | 2.7 | 4.2 |
| P/A measured (cerebellar reference in brackets) | 0.48 (0.48) | 0.56 (0.57) | 0.53 (0.54) | 0.50 (0.50) |
| expected, normal gradient 0.80-0.88, all assumptions (primary) | 0.67-0.82 | 0.68-0.79 | 0.71-0.82 | 0.73-0.79 |
| expected, normal gradient 0.84, nominal, FWHM 9-15 mm | 0.75-0.79 | 0.73-0.77 | 0.75-0.78 | 0.76-0.78 |
| expected, uniform binding, all assumptions (lenient) | 0.77-0.89 | 0.78-0.84 | 0.81-0.89 | 0.82-0.88 |
| gap vs normal-gradient low, base VOI | −0.19 | −0.12 | −0.18 | −0.23 |
| decision (gap < −0.05) | loss | loss | loss | loss |
| gap over all sweep variants | −0.27 to +0.04 | −0.20 to −0.00 | −0.19 to +0.02 | −0.28 to −0.05 |
| variants where a loss is NOT called | `hi` −6, posterior +6 mm | posterior +4 and +6 mm | posterior +4/+6 mm, `hi` +6 | none (posterior +6 mm: −0.052) |
| GTM 9 / 11 / 13 mm | 0.42 / 0.40 / 0.38 | 0.52 / 0.50 / 0.47 | 0.47 / 0.44 / 0.41 | 0.41 / 0.38 / 0.34 |
| 12 rigid perturbations, measured | 0.41-0.56 | 0.53-0.61 | 0.48-0.63 | 0.45-0.55 |
| posterior tail trimmed 8-13 mm: measured (normal-gradient expectation) | 0.66-0.79 (0.88-0.96) | 0.62-0.70 (0.79-0.84) | 0.67-0.72 (0.86-0.93) | 0.56-0.61 (0.80-0.86) |

v3 sweep, gap vs normal-gradient low: whole VOI −2..+6 mm R −0.17 to −0.07,
L −0.23 to −0.20. Posterior VOI −2..+6 mm R −0.12 to −0.00, L −0.24 to −0.05.
AC split ±4 mm R −0.19 / −0.09, L −0.27 / −0.16. Segmentation variants, each
rebuilding both the VOI and the simulated putamen: FastSurfer VOIs (3.8 / 3.9
mL) give R 0.52 (gap −0.20) and L 0.47 (gap −0.28). Threshold VOIs give R 0.48
(gap −0.19) and L 0.53 (gap −0.18). That reproduces v2 exactly inside the v3 run.

Segmentation agreement (Dice, putamen R / L): SynthSeg vs FastSurfer 0.87 /
0.83, SynthSeg vs threshold 0.53 / 0.66, FastSurfer vs threshold 0.52 / 0.60.
Caudate: SynthSeg vs FastSurfer 0.73 / 0.74, SynthSeg vs threshold 0.48 / 0.45.
SynthSeg QC scores were 0.72-0.96 (putamen+pallidum 0.81, gate 0.65). Typical
adult putamen is about 4-6 mL per side. The two learned segmenters (3.8-4.5 mL)
sit at the low end, as expected with age. The threshold VOI was nearly twice that.

What changed against v2: both sides are still called. The two anatomical
segmenters agree with each other far better than either agrees with the
threshold VOIs, and both give a loss on both sides. The right-side margin
shrank from −0.19 to −0.12. The right side no longer hinges on a threshold,
because there is none. Its only "not called" variants are the posterior VOI
grown 4-6 mm (11-17 mL), which on the phantom also hides a real loss. The
laterality flipped: v2 had the right side lower, v3 the left. Do not read a
left/right difference from this tool.

The simulated putamen is now the SynthSeg putamen itself. The grey-core step
rejected its Otsu split (1.2 / 1.4 mL) and the 5- and 6-mL caps do not bind, so
the anatomical-size sweep only spans 4.0 mL to the VOI size (4.2 / 4.5 mL).

### Results, v2 (threshold VOIs)

Kept for comparison as the v2 run, reproduced by `--segmenter threshold`.
v2 put the binding in an anatomically sized putamen and used the normal
front-to-back gradient. That lowered the expectation from 0.87-0.90 (v1) to
0.67-0.82, a margin of about 0.18-0.19. The right side flipped to normal (P/A
0.73) when the white-matter stop threshold `hi` moved by −6, which halved the
VOI. That fragility is why v3 moved to an anatomical segmenter. v2 also began
excluding the SPECT's zero padding (7 % of the cerebellar reference had lain
below the first reconstructed slice), and its analysis grid gained one voxel at
the far x/y edge.

## Limitations

* **Not a medical device.** It is not validated against a normal database or clinical outcome. It cannot distinguish Parkinson's disease from atypical or vascular parkinsonism or other causes of low posterior putamen binding, and it does not replace the visual read.
* **The comparator is a model, not a normal database.** The primary null assumes a linear front-to-back gradient with a true P/A of 0.80-0.88, taken from DaTQUANT normal values at age 65 (Grisanti 2023). That is a different pipeline, template and age group. The uniform-binding null is a lenient upper bound. Real normal putamen is not uniform, so comparing with it overstates any loss.
* **VOI size interacts with the simulation.** The expected ratio depends on the true putamen size (the simulated binding sits in a VOI capped at 4-6 mL, an assumption) and on how the measurement VOI is drawn. On the phantom, growing the posterior VOI 4-6 mm still behaves correctly for a normal scan but hides a real loss. Growing the whole VOI by 2 mm moves the gap down by up to about 0.03 (anterior/caudate spill is modelled imperfectly). The VOI and threshold sweep exist to show this.
* **Segmentation.** The threshold method was set by eye on one case and collapsed on 4 of 10 randomised phantoms. The 4-mL gate stopped all four, but a partial failure above 4 mL would not be caught. The reference case now uses SynthSeg 2.0, which is trained on real brains, not on this phantom. The phantom tests therefore do NOT validate the SynthSeg path. They test its plumbing with the true phantom labels, and its gates. SynthSeg and FastSurfer agree with each other (putamen Dice 0.83-0.87), but they share training data and label conventions, so their agreement is not independent ground truth. The reference brain has markedly enlarged lateral ventricles, which is outside the bulk of the data such tools are trained on. Always check figure 2.
* **Laterality is not robust.** v2 (threshold VOIs) gave the right side lower, v3 (SynthSeg) the left. Do not read a left/right difference from this tool.
* **The anatomical-size assumption is narrower with SynthSeg.** The simulated putamen is the SynthSeg putamen (4.2 / 4.5 mL), and the 5- and 6-mL caps cannot exceed it. If SynthSeg under-segments, the expectation never sees a larger true putamen.
* **Simulation physics.** The nulls use an isotropic Gaussian PSF with no attenuation, scatter, collimator depth-dependent resolution or OSEM non-linearity. The phantom SPECT does include anisotropic resolution, residual skull attenuation and Poisson noise, and against it the nulls sat about 0.01-0.09 above the noise-free measured ratio. That bias leans toward calling a loss.
* **The normal-gradient margin is thin.** On the 6 phantoms that segmented, the normal-gradient scan landed −0.02 to +0.03 from the lowest expectation (never called, rule −0.05). Per-scan noise SD is 0.011-0.023, so a normal scan will sometimes be called.
* **GTM depends on the assumed FWHM** and assumes uniform activity inside each VOI. On the phantom at 11 mm it over-corrects (posterior-loss P/A 0.59 / 0.48 vs true 0.61 / 0.57, uniform 0.96 / 0.80 vs 1.0), and more so at 13 mm. Treat GTM values as indicative only.
* **Caudate VOI.** With the threshold method it is narrow (at most 5 mm from the ventricle wall). With SynthSeg it is the whole caudate (4.2-4.8 mL). Caudate SBRs from the two are not comparable. Caudate spill into the anterior putamen is modelled imperfectly.
* **Low injected activity and large radius.** The reference scan used 0.96 mCi, so it is noisy, and its large radius of rotation blurs it (hence the 9-15 mm FWHM range). On the phantom at the reference case's noise level, the single-scan SD of P/A is 0.011-0.023. No noise estimate is derived from the patient data.
* **MRI not contemporaneous.** In the reference case the MRI is 10.5 months older than the SPECT. Atrophy in between is not modelled.
* **The AC is placed by hand.** Moving the anterior/posterior split ±4 mm is part of the sweep. The phantom tests give the pipeline an AC that is 2-3 mm off.
* **Rigid registration only.** Ventricle Dice (gated at 0.6) and the overlay figure are the QC. There is no non-linear correction.
* The analysis grid follows the CT axes, not AC-PC. "Coronal slabs" are CT-coronal, about 4° off AC-PC in the reference case.
* On the reference case the SPECT's axial field of view cuts 7 % of the cerebellar reference, so the occipital reference is primary.

## Validation (synthetic phantom, `tests/`)

A digital head phantom (`datscan_mri/phantom.py`) has scalp, skull, CSF,
cortex, white matter, ventricles, caudate, putamen, insular cortex with sylvian
CSF lateral to the putamen, and a cerebellum. It is randomised (seeded) so the
tests are not tuned to the segmentation thresholds. Tissue intensities vary by
±8 %, putamen and caudate size and position vary per side, and the capsule,
insular-cortex and CSF widths vary row by row (CSF missing on 15 % of rows). The
MRI is misaligned by a known rigid motion, and the AC given to the pipeline is
off by (2, −2.5, 2) mm. The phantom SPECT is made differently from the
analysis' own simulation: anisotropic resolution (10.5 / 12.5 / 13 mm FWHM),
skull attenuation left over after an approximate Chang correction, 3.9-mm
voxels and Poisson noise at the reference case's noise level.

Controls (decision rule above, 3 noise realisations averaged):

| control | true P/A | requirement |
|---|---|---|
| (a) uniform binding | 1.00 | no loss called |
| (b) smooth posterior loss (to 30 % at 18 mm behind the AC) | about 0.6 | loss called, base VOI and whole VOI ±2 mm |
| (c) all striatal binding × 0.6 | 1.00 | P/A within 0.05 of (a), absolute SBR down > 25 %, no loss called |
| (d) normal gradient | 0.84 | no loss called |
| (e) normal gradient, posterior VOI +4 / +6 mm, whole VOI ±2 mm | 0.84 | no loss called, and the expectation falls with posterior oversizing, as the measurement does |

Also tested: registration error under 1 mm, putamen VOI Dice above 0.7 and
grey-core Dice above 0.8 against the truth, identical results when the CT is
stored with its y axis flipped, and each quality gate (estimated AC, collapsed
segmentation, 10-mm registration error, VOIs outside the SPECT support,
ambiguous DICOM folder).

Seed sweep over phantoms 1-10 (one noise realisation each, outside the test
suite). 4 of 10 were stopped by the putamen-volume gate (true Dice 0.02-0.32).
On the other 6, gaps against the normal-gradient low were:

| scenario | base VOI | whole VOI +2 mm | posterior VOI +6 mm |
|---|---|---|---|
| uniform | +0.06 to +0.14 (never called) | +0.05 to +0.13 | +0.14 to +0.19 |
| normal gradient | −0.02 to +0.03 (never called) | −0.04 to +0.01 (never called) | +0.09 to +0.14 |
| posterior loss | −0.20 to −0.10 (always called) | −0.21 to −0.12 (always called) | +0.01 to +0.07 (never called: oversizing hides the loss) |

## Method references

* Rousset OG, Ma Y, Evans AC. Correction for partial volume effects in PET: principle and validation. *J Nucl Med* 1998;39:904-911 (GTM).
* Mattes D, Haynor DR, Vesselle H, Lewellen TK, Eubank W. PET-CT image registration in the chest using free-form deformations. *IEEE Trans Med Imaging* 2003;22:120-128 (mutual-information metric).
* Tustison NJ, et al. N4ITK: improved N3 bias correction. *IEEE Trans Med Imaging* 2010;29:1310-1320.
* Neill M, et al. Quantitative analysis of DaTscan SPECT with DaTQUANT (occipital reference, posterior-putamen thresholds). 2021, PMC8706562.
* Grisanti S, et al. DaTQUANT normal values (Table 1). 2023, PMC10338803.
* Billot B, et al. Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets. *PNAS* 2023;120:e2216399120 (SynthSeg 2.0).
* Henschel L, et al. FastSurfer: a fast and accurate deep learning based neuroimaging pipeline. *NeuroImage* 2020;219:117012.
* Lowekamp BC, et al. The design of SimpleITK. *Front Neuroinform* 2013;7:45.

## License

MIT. See `LICENSE`.
