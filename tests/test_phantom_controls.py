"""Two-sided controls on a RANDOMISED synthetic head phantom, run through the
full chain (registration, segmentation, simulated nulls, VOI sweep, GTM).

The phantom is not tuned to the method: tissue intensities, putamen/caudate
size and position and the tissue lateral of the putamen are random (seeded);
the AC given to the pipeline is off by (2, -2.5, 2) mm; the SPECT has an
anisotropic resolution, skull attenuation left over after an approximate
Chang correction, and Poisson noise at the reference case's noise level, none
of which the analysis' own simulation models.

Decision rule under test (pipeline.analyze): a posterior loss is called if
the measured P/A is below the LOWEST normal-gradient expectation by more than
the margin (0.05). Measured values are averaged over 3 noise realisations.

(a) uniform binding                    -> not called
(b) smooth posterior loss              -> called, with the base VOI and modest VOI errors
(c) global reduction                   -> P/A unchanged, not called
(d) normal gradient (true P/A 0.84)    -> not called
(e) normal gradient + posterior VOI oversized 4-6 mm -> not called (the key control
    for decoupling the simulated binding from the measurement VOI)
"""
import numpy as np
import pytest

MARGIN = 0.05


def gap(r, variant="base"):
    """noise-averaged measured P/A minus the lowest normal-gradient expectation."""
    row = r["sweep_by_name"][variant]
    return {sd: r["pa_mean"][variant][sd] - row["expected_normal_gradient"][sd][0] for sd in "RL"}


def test_registration_recovers_known_motion(phantom_case):
    A, motion = phantom_case["A"], phantom_case["motion"]
    z, y, x = A.ac
    for off in [(0, 0, 0), (40, 0, 0), (0, 50, 0), (0, 0, 40)]:
        p = A.grid.TransformIndexToPhysicalPoint((x + off[0], y + off[1], z + off[2]))
        err = np.linalg.norm(np.subtract(A.transform.TransformPoint(p), motion.TransformPoint(p)))
        assert err < 1.0, f"registration error {err:.2f} mm at offset {off}"


def test_putamen_segmentation_and_grey_core(phantom_case):
    from scipy import ndimage as nd

    from datscan_mri import segment
    A, t = phantom_case["A"], phantom_case["truth"]
    G = phantom_case["P"]["regions"]
    core = segment.anatomical_putamen(A.seg, nd.gaussian_filter(A.mg, 1.0), G["csf_hi"], G["gm_hi"])
    for side, codes, c in (("R", (3, 5), 3), ("L", (4, 6), 4)):
        v = np.isin(A.lab, codes)
        tr = t["put"][side]
        dice = 2 * (v & tr).sum() / (v.sum() + tr.sum())
        assert dice > 0.7, f"{side} putamen VOI Dice {dice:.2f}"
        assert v.sum() > tr.sum(), "the VOI is expected to be generous"
        k = core == c
        dk = 2 * (k & tr).sum() / (k.sum() + tr.sum())
        assert dk > 0.8 and dk > dice, f"{side} grey core Dice {dk:.2f} (VOI {dice:.2f})"


def test_a_uniform_binding_no_false_positive(scenario_results):
    r = scenario_results["uniform"]
    g = gap(r)
    for side in "RL":
        assert g[side] > -MARGIN, (side, g[side])
        assert not r["decision"]["posterior_loss_called"][side] or g[side] > -MARGIN
        # GTM-corrected: true ratio is 1.0 (assumes uniform VOIs, which holds here)
        assert abs(r["gtm"]["11"]["occ"][side]["post_ant"] - 1.0) < 0.25


def test_b_posterior_loss_detected(scenario_results):
    r = scenario_results["posterior_loss"]
    u = scenario_results["uniform"]
    for side in "RL":
        assert r["true_post_ant"][side] < 0.65
        for variant in ("base", "whole-2mm", "whole+2mm"):
            assert gap(r, variant)[side] < -MARGIN, (side, variant, gap(r, variant)[side])
        assert r["pa_mean"]["base"][side] < u["pa_mean"]["base"][side] - 0.15
        # GTM moves toward the true ratio
        assert r["gtm"]["11"]["occ"][side]["post_ant"] < r["uncorrected"]["occ"][side]["post_ant"]
        # the profile falls behind the AC more steeply than the normal-gradient simulation
        p = r["profile"]
        x = np.array(p["mm_from_ac"])
        meas = np.array(p[side], float)
        sim = np.array(p[f"simng_{side}_scaled"], float)
        post = (x > 4) & np.isfinite(meas)
        assert np.nanmean(meas[post] - sim[post]) < -0.2


def test_c_global_reduction_ratio_unchanged(scenario_results):
    g, u = scenario_results["global_loss"], scenario_results["uniform"]
    for side in "RL":
        assert abs(g["pa_mean"]["base"][side] - u["pa_mean"]["base"][side]) < 0.05
        assert gap(g)[side] > -MARGIN
        # but the absolute binding did fall
        assert g["uncorrected"]["occ"][side]["put"] < 0.75 * u["uncorrected"]["occ"][side]["put"]


def test_d_normal_gradient_not_called(scenario_results):
    r = scenario_results["normal_gradient"]
    for side in "RL":
        assert abs(r["true_post_ant"][side] - 0.84) < 0.005
        assert gap(r)[side] > -MARGIN, (side, gap(r)[side], r["pa_mean"]["base"][side])


@pytest.mark.parametrize("variant", ["posterior+4mm", "posterior+6mm", "whole-2mm", "whole+2mm"])
def test_e_oversized_voi_normal_gradient_not_called(scenario_results, variant):
    r = scenario_results["normal_gradient"]
    for side in "RL":
        assert gap(r, variant)[side] > -MARGIN, (variant, side, gap(r, variant)[side])


def test_e2_expectation_falls_with_posterior_oversizing(scenario_results):
    """With the binding held in the anatomical putamen, growing the posterior
    VOI must LOWER the expected ratio, as it lowers the measured one."""
    r = scenario_results["normal_gradient"]
    sw = r["sweep_by_name"]
    for side in "RL":
        e = [np.mean(sw[v]["expected_normal_gradient"][side]) for v in ("base", "posterior+2mm", "posterior+4mm",
                                                                         "posterior+6mm")]
        m = [r["pa_mean"][v][side] for v in ("base", "posterior+2mm", "posterior+4mm", "posterior+6mm")]
        assert all(np.diff(e) < 0) and all(np.diff(m) < 0), (side, e, m)


def test_noise_level(scenario_results):
    """Per-realisation SD of P/A at the calibrated count level stays small
    compared with the decision margin."""
    for sc, r in scenario_results.items():
        for side in "RL":
            assert r["pa_sd"]["base"][side] < 0.05, (sc, side, r["pa_sd"]["base"][side])
