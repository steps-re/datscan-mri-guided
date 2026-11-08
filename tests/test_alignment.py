"""SPECT-to-anatomy alignment check (datscan_mri/alignment.py) and the
profile-figure scaling.

Phantom: the phantom SPECT is translated by a known amount (its origin moved,
so the content moves physically) and the caudate-only fit must recover the
shift within 1 mm per axis, with a normal gradient and with a posterior loss
(the caudate fit must not be pulled by the putamen pattern)."""
import copy

import numpy as np
import pytest

from datscan_mri import alignment, phantom, pipeline
from datscan_mri import profile as prof

from conftest import NOISE_SEEDS, QUIET


def _shifted_spect(truth, scenario, shift_lps_mm):
    sp = phantom.make_spect(truth, scenario, np.random.default_rng(NOISE_SEEDS[0]))
    sp.SetOrigin(tuple(np.add(sp.GetOrigin(), shift_lps_mm)))
    return sp


def _params(P):
    P = copy.deepcopy(P)
    P["alignment_qc"]["masks"] = {"caudate only": [1, 2]}
    P["alignment_qc"]["targets"] = ["normal_gradient"]
    return P


@pytest.fixture(scope="session")
def truth_vois_case(phantom_case, tmp_path_factory):
    """Anatomy with the phantom's TRUE caudate/putamen as VOIs (loaded through the
    external-segmenter path the reference case uses), plus its simulated nulls."""
    from test_segmenter import _fs_file
    from test_segmenter import _params as seg_params
    d = tmp_path_factory.mktemp("truth_vois")
    P = seg_params(_fs_file(phantom_case["truth"], d / "truth_fs.nii.gz"))
    P["simulation"]["anatomical_putamen_cap_ml_sweep"] = [5.0]
    P["simulation"]["fwhm_expected"] = [11, 13]
    P["gtm"]["fwhm"] = [11]
    P["alignment_qc"]["enabled"] = False
    A = pipeline.prepare_anatomy(phantom_case["ct"], phantom_case["mri"], P, transform=phantom_case["motion"],
                                 workdir=d, **QUIET)
    sp = phantom.make_spect(phantom_case["truth"], "normal_gradient", np.random.default_rng(NOISE_SEEDS[0]))
    r = pipeline.analyze(A, sp, P, do_robustness=False, do_sweeps=False, **QUIET)
    low = {sd: r["simulation"]["normal_gradient"]["expected_post_ant_range"][sd][0] for sd in "RL"}
    P["alignment_qc"]["enabled"] = True
    return dict(A=A, P=_params(P), sims=r["_sim"], low=low)


def _fit(case, truth, scenario, post_mm):
    sp = _shifted_spect(truth, scenario, (0.0, float(post_mm), 0.0))
    return pipeline.alignment_qc(case["A"], sp, case["P"], case["sims"], case["low"])


@pytest.mark.parametrize("scenario", ["normal_gradient", "posterior_loss"])
@pytest.mark.parametrize("post_mm", [0, 4])
def test_caudate_fit_recovers_known_shift(phantom_case, truth_vois_case, scenario, post_mm):
    """Correct caudate VOIs: the SPECT is moved post_mm POSTERIOR (+y LPS); the
    fit must move it back (best shift y = -post_mm) within 1 mm on every axis,
    with a normal gradient and with a posterior loss."""
    aq = _fit(truth_vois_case, phantom_case["truth"], scenario, post_mm)
    x, y, z = aq["primary"]["best_shift_mm_xyz"]
    assert abs(y + post_mm) <= 1 and abs(x) <= 1 and abs(z) <= 1, (scenario, post_mm, (x, y, z))
    assert aq["primary"]["correlation"] > 0.8
    print(f"\n[alignment] truth VOIs, {scenario}, SPECT moved {post_mm} mm posterior: best shift (x,y,z) "
          f"{(x, y, z)} mm, r {aq['primary']['correlation']:.3f}")
    assert aq["shift_exceeds_limit"] == (aq["primary"]["best_shift_norm_mm"] > 3)
    assert ("warning" in aq) == aq["shift_exceeds_limit"]
    if post_mm == 0:
        assert not aq["shift_exceeds_limit"]


@pytest.mark.parametrize("scenario", ["normal_gradient", "posterior_loss"])
def test_check_also_flags_a_displaced_caudate_voi(phantom_case, scenario_results, scenario):
    """Two-sided: with perfect SPECT registration but the THRESHOLD caudate VOI
    (which on this phantom sits about 3.5 mm posterior of the true caudate),
    the check reports a posterior shift. It measures SPECT vs SIMULATED ANATOMY,
    not SPECT vs MRI alone. A known 4-mm shift is still recovered relative to that."""
    r = scenario_results[scenario]
    P = _params(phantom_case["P"])
    low = {sd: r["simulation"]["normal_gradient"]["expected_post_ant_range"][sd][0] for sd in "RL"}
    a0 = pipeline.alignment_qc(phantom_case["A"], _shifted_spect(phantom_case["truth"], scenario, (0, 0, 0)), P,
                               r["_sim"], low)["primary"]["best_shift_mm_xyz"]
    a4 = pipeline.alignment_qc(phantom_case["A"], _shifted_spect(phantom_case["truth"], scenario, (0, 4.0, 0)), P,
                               r["_sim"], low)["primary"]["best_shift_mm_xyz"]
    print(f"\n[alignment] threshold VOIs, {scenario}: best shift (x,y,z) mm at 0: {a0}, SPECT moved 4 mm: {a4}")
    assert a0[1] >= 2, a0
    assert abs((a4[1] - a0[1]) + 4) <= 1 and abs(a4[0] - a0[0]) <= 1 and abs(a4[2] - a0[2]) <= 1, (a0, a4)


def test_full_run_records_alignment_qc(scenario_results):
    """analyze() writes alignment_qc for all three masks and both nulls; at
    zero shift the re-measured P/A equals the decision value exactly."""
    for sc in ("normal_gradient", "posterior_loss"):
        r = scenario_results[sc]
        aq = r["alignment_qc"]
        assert {(f["mask"], f["target"]) for f in aq["fits"]} == {
            (m, t) for m in ("caudate only", "caudate + anterior putamen", "striatum")
            for t in ("normal_gradient", "uniform")}
        for sd in "RL":
            assert aq["as_registered"]["post_ant"][sd] == r["uncorrected"]["occ"][sd]["post_ant"]
            assert aq["as_registered"]["gap_vs_normal_gradient_low"][sd] == pytest.approx(
                r["decision"]["gap_vs_normal_gradient_low"][sd], abs=1e-12)
        assert aq["primary"]["mask"] == "caudate only" and aq["primary"]["target"] == "normal_gradient"
        for f in aq["fits"]:
            if f["best_shift_mm_xyz"] == [0, 0, 0]:
                assert f["post_ant"] == aq["as_registered"]["post_ant"]


def test_gate_stop_raises(scenario_results):
    r = copy.deepcopy({k: v for k, v in scenario_results["normal_gradient"].items() if not k.startswith("_")})
    r.pop("warnings", None)
    r["alignment_qc"]["shift_exceeds_limit"] = True
    r["alignment_qc"]["warning"] = "test"
    P = {"alignment_qc": {"gate": "warn"}}
    pipeline._alignment_gate(r, P, log=lambda *a: None)
    assert r["warnings"] == ["test"]
    P["alignment_qc"]["gate"] = "stop"
    with pytest.raises(pipeline.QCError):
        pipeline._alignment_gate(r, P, log=lambda *a: None)


def test_shifted_crop_convention():
    full = np.arange(5 * 6 * 7, dtype=float).reshape(5, 6, 7)
    crop = (slice(1, 4), slice(1, 5), slice(2, 6))
    assert np.array_equal(alignment.shifted_crop(full, crop, (0, 0, 0)), full[crop])
    s = alignment.shifted_crop(full, crop, (0, 1, 0))       # content moved +1 in y
    assert np.array_equal(s[:, 1:], full[1:4, 1:4, 2:6])
    assert np.array_equal(s[:, 0], full[1:4, 0, 2:6])
    e = alignment.shifted_crop(full, crop, (0, -5, 0), fill=-1)   # off the grid
    assert (e == -1).all()


def test_best_shift_recovers_synthetic_translation():
    rng = np.random.default_rng(0)
    from scipy import ndimage as nd
    tgt = nd.gaussian_filter(rng.random((40, 40, 40)), 2.0)
    full = np.zeros((60, 60, 60))
    full[10:50, 10:50, 10:50] = tgt
    full = np.roll(full, (2, -3, 1), axis=(0, 1, 2))      # content moved by (2, -3, 1)
    crop = (slice(10, 50),) * 3
    mask = np.zeros(tgt.shape, bool)
    mask[12:28, 12:28, 12:28] = True
    d, c, c0 = alignment.best_shift(full, crop, tgt, mask, 5)
    assert d == (-2, 3, -1) and c > 0.999 > c0


def test_profile_scaling_uses_voxel_weighted_anterior_mean():
    x = [-5, -3, -1, 1, 3, 5, 7]
    meas = [2.0, 3.0, np.nan, 2.0, 1.5, 1.0, 0.5]
    sim = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    n = [10, 30, 25, 40, 40, 30, 20]
    scaled, k = prof.scale_to_anterior(meas, sim, x, n)
    # anterior slabs with both curves finite: x=-5 (n 10) and x=-3 (n 30); the NaN slab is skipped
    assert k == pytest.approx((2.0 * 10 + 3.0 * 30) / 40)
    assert scaled == pytest.approx([k] * 7)
    # NOT the peak: peak scaling would give 3.0
    assert k != pytest.approx(3.0)
    # a shape that matches exactly is reproduced exactly
    s2 = [0.5, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1]
    m2 = [2 * v for v in s2]
    sc2, k2 = prof.scale_to_anterior(m2, s2, x, n)
    assert k2 == pytest.approx(2.0) and sc2 == pytest.approx(m2)
    # no usable anterior slab -> NaN, not a silent number
    _, k3 = prof.scale_to_anterior([np.nan] * 3 + [1.0] * 4, sim, x, n)
    assert np.isnan(k3)


def test_profile_records_lowest_normal_gradient_curve(scenario_results):
    p = scenario_results["normal_gradient"]["profile"]
    S = scenario_results["normal_gradient"]["simulation"]["normal_gradient"]
    for sd in "RL":
        assert len(p[f"simnglow_{sd}_scaled"]) == len(p["mm_from_ac"])
        assert p["simnglow_expected_post_ant"][sd] == pytest.approx(S["expected_post_ant_range"][sd][0])
        assert p["simnglow_assumption"][sd]["true_post_ant"] == min(S["true_post_ant_range"])


def test_combined_shift_grid_includes_decision_value(phantom_case, scenario_results):
    from datscan_mri import register, robustness
    import SimpleITK as sitk
    A, P = phantom_case["A"], phantom_case["P"]
    sp = phantom.make_spect(phantom_case["truth"], "normal_gradient", np.random.default_rng(NOISE_SEEDS[0]))
    SL = pipeline._crop(A.lab, P["analysis"]["crop_margin"])
    sg = register.to_grid(sp, A.grid, None, sitk.sitkLinear, 0)
    fov = register.spect_fov_mask(sp, A.grid)
    g = robustness.combined_shift_grid(sg, fov, SL, A.lab[SL], pipeline._measured_main, 2, 2)
    assert len(g["grid"]) == 27
    zero = [row for row in g["grid"] if row["shift_mm_xyz"] == [0, 0, 0]][0]
    r = scenario_results["normal_gradient"]
    for sd in "RL":
        assert zero[sd] == r["uncorrected"]["occ"][sd]["post_ant"]
        assert g["range"][sd][0] < zero[sd] < g["range"][sd][1]
