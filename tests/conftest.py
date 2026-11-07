import numpy as np
import pytest

from datscan_mri import params, phantom, pipeline

SEED = 1                       # first randomised phantom; see README for the seed sweep
AC_OFFSET = (2.0, -2.5, 2.0)   # the AC given to the pipeline is off by this much (x, y, z mm)
SCENARIOS = ("uniform", "normal_gradient", "posterior_loss", "global_loss")
NOISE_SEEDS = (1, 2, 3)
QUIET = dict(log=lambda *a: None)


def make_params():
    """Package defaults, except fewer anatomical-size variants, a shorter VOI
    sweep and no threshold sweep (runtime)."""
    P = params.load(None, ac_lps_mm=phantom.ac_lps(AC_OFFSET))
    P["simulation"]["anatomical_putamen_cap_ml_sweep"] = [4.0, 5.0, None]
    P["sweep"]["voi_mm"] = [-2, 2, 4, 6]
    P["sweep"]["threshold"] = {}
    P["gtm"]["fwhm"] = [11]
    return P


@pytest.fixture(scope="session")
def phantom_case():
    anat, mri, ct, motion = phantom.build(seed=SEED)
    P = make_params()
    A = pipeline.prepare_anatomy(ct, mri, P, **QUIET)
    return dict(truth=anat, mri=mri, ct=ct, motion=motion, P=P, A=A, cache={})


def _noise_rep(case, sc, seed):
    sp = phantom.make_spect(case["truth"], sc, np.random.default_rng(seed))
    return pipeline.measured_variants(case["A"], sp, case["P"], case["cache"])


@pytest.fixture(scope="session")
def scenario_results(phantom_case):
    """Full analysis on noise seed 1 (incl. nulls and VOI sweep), plus the
    measured P/A of every VOI variant averaged over 3 Poisson realisations
    ("pa_mean": {variant: {R, L}}) and their SD ("pa_sd")."""
    out = {}
    for sc in SCENARIOS:
        sp = phantom.make_spect(phantom_case["truth"], sc, np.random.default_rng(NOISE_SEEDS[0]))
        r = pipeline.analyze(phantom_case["A"], sp, phantom_case["P"], do_robustness=False,
                             cache=phantom_case["cache"], **QUIET)
        reps = [{row["variant"]: row["measured"] for row in r["sweep"]["voi"]}]
        reps += [_noise_rep(phantom_case, sc, s) for s in NOISE_SEEDS[1:]]
        r["pa_mean"] = {v: {sd: float(np.mean([x[v][sd] for x in reps])) for sd in "RL"} for v in reps[0]}
        r["pa_sd"] = {v: {sd: float(np.std([x[v][sd] for x in reps], ddof=1)) for sd in "RL"} for v in reps[0]}
        r["sweep_by_name"] = {row["variant"]: row for row in r["sweep"]["voi"]}
        r["true_post_ant"] = phantom.true_post_ant(phantom_case["truth"], sc)
        out[sc] = r
    return out
