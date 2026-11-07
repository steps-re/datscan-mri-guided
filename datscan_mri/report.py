"""results.json and figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import __version__
from .register import qc_figure

DISCLAIMER = ("Research tool, not a medical device, not for diagnosis. Results are exploratory "
              "and have not been validated against a normal database.")

COLORS = {1: "c", 2: "b", 3: "r", 4: "m", 5: "y", 6: "g"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_json(res: dict, path):
    doc = {"tool": "datscan-mri", "version": __version__, "disclaimer": DISCLAIMER}
    doc.update(_clean(res))
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return doc


def fig_segmentation(anat, path):
    plt = _plt()
    z0, y0, x0 = anat.ac
    m, seg = anat.mg, anat.lab
    zs = [z0 - 12, z0 - 6, z0, z0 + 6]
    ys = [y0 - 15, y0 - 5, y0 + 5, y0 + 15]
    ya, yb, xa, xb = y0 - 33, y0 + 42, x0 - 46, x0 + 48
    za, zb = z0 - 26, z0 + 34
    vmin, vmax = np.percentile(m[anat.ic], [5, 99.5]) if anat.ic.any() else (0, 1)
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    for j, z in enumerate(zs):
        ax[0, j].imshow(m[z, ya:yb, xa:xb], cmap="gray", vmin=vmin, vmax=vmax, extent=(xa - .5, xb - .5, yb - .5, ya - .5))
        for c, col in COLORS.items():
            s = seg[z, ya:yb, xa:xb] == c
            if s.any():
                ax[0, j].contour(np.arange(xa, xb), np.arange(ya, yb), s, levels=[0.5], colors=col, linewidths=0.9)
        ax[0, j].axhline(y0, color="w", lw=0.5, ls=":")
        ax[0, j].set_title(f"axial, {z - z0:+d} mm from AC", fontsize=10)
    for j, y in enumerate(ys):
        ax[1, j].imshow(m[za:zb, y, xa:xb], cmap="gray", vmin=vmin, vmax=vmax, origin="lower", extent=(xa - .5, xb - .5, za - .5, zb - .5))
        for c, col in COLORS.items():
            s = seg[za:zb, y, xa:xb] == c
            if s.any():
                ax[1, j].contour(np.arange(xa, xb), np.arange(za, zb), s, levels=[0.5], colors=col, linewidths=0.9)
        ax[1, j].set_title(f"coronal, {y - y0:+d} mm {'posterior' if y > y0 else 'anterior'} to AC", fontsize=10)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    si = getattr(anat, "seg_info", {}) or {}
    tool = si.get("version") or si.get("tool") or "threshold method"
    ml = si.get("putamen_ml")
    vol = f" Putamen R {ml['R']:.1f} / L {ml['L']:.1f} mL." if ml else ""
    fig.suptitle(f"MRI-defined VOIs from {tool} (patient right on image left).{vol} Caudate cyan/blue; anterior "
                 "putamen red/magenta; posterior putamen yellow/green. Dotted line: AC plane.", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=60)
    plt.close(fig)


def fig_profile(res, path):
    plt = _plt()
    p = res["profile"]
    x = np.array(p["mm_from_ac"])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for side, col, name in (("R", "tab:red", "Right"), ("L", "tab:purple", "Left")):
        ax.plot(x, p[side], "o-", color=col, label=f"{name} putamen, measured")
        ax.plot(x, p[f"simng_{side}_scaled"], "--", color=col, alpha=0.7,
                label=f"{name}, normal-gradient simulation (true P/A {p['simng_true_post_ant']}), scaled to peak")
        ax.plot(x, p[f"sim_{side}_scaled"], ":", color=col, alpha=0.5,
                label=f"{name}, uniform-binding simulation, scaled to peak")
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.text(0.5, ax.get_ylim()[0] + 0.05, "AC plane (anterior | posterior split)", fontsize=8)
    ax.set_xlabel("mm posterior to the anterior commissure (2-mm coronal slabs)")
    ax.set_ylabel("uncorrected SBR (occipital grey reference)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(f"Uptake along the MRI-defined putamen vs. simulations ({p['sim_fwhm_mm']:g} mm FWHM, "
                 "binding in the anatomical putamen)", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


def fig_sweep(res, path):
    """Measured P/A and the normal-gradient expectation across the VOI / threshold sweep."""
    if "sweep" not in res:
        return
    plt = _plt()
    rows = res["sweep"]["voi"] + res["sweep"]["threshold"] + res["sweep"].get("segmenter", [])
    names = [r["variant"] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(13, 0.32 * len(rows) + 1.8), sharey=True)
    y = np.arange(len(rows))
    for ax, side in zip(axs, "RL"):
        m = [r["measured"][side] for r in rows]
        g = np.array([r["expected_normal_gradient"][side] for r in rows])
        u = np.array([r["expected_uniform"][side] for r in rows])
        ax.hlines(y, u[:, 0], u[:, 1], color="0.7", lw=6, label="expected, uniform (lenient)")
        ax.hlines(y, g[:, 0], g[:, 1], color="tab:blue", lw=6, alpha=0.6, label="expected, normal gradient 0.80-0.88")
        ax.plot(m, y, "ko", label="measured")
        ax.set_title({"R": "Right", "L": "Left"}[side] + " putamen: posterior/anterior SBR ratio", fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(names, fontsize=7)
    axs[0].invert_yaxis()
    axs[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Measurement-VOI sweep (simulated binding held fixed); threshold / alternative-segmenter rows "
                 "rebuild both VOI and simulated binding; FWHM 9-15 mm", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


def ratio_rows(res):
    rows = []
    u = res["uncorrected"]
    for ref in ("occ", "cer"):
        rows.append((f"Measured, uncorrected ({'occipital' if ref == 'occ' else 'cerebellar'} ref)",
                     u[ref]["R"]["post_ant"], u[ref]["L"]["post_ant"]))
    S = res["simulation"]
    e = S["normal_gradient"]["expected_post_ant_range"]
    rows.append(("Expected, normal gradient, true P/A 0.80-0.88 (range, primary)", tuple(e["R"]), tuple(e["L"])))
    e = S["uniform"]["expected_post_ant_range"]
    rows.append(("Expected, uniform binding (range, lenient upper bound)", tuple(e["R"]), tuple(e["L"])))
    if "sweep" in res:
        g = res["sweep"]["gap_vs_normal_gradient_low_range"]
        rows.append(("Measured minus normal-gradient low, over VOI/segmentation sweep (range)",
                     tuple(g["R"]), tuple(g["L"])))
    for fw, g in res["gtm"].items():
        rows.append((f"GTM partial-volume corrected, {fw} mm FWHM", g["occ"]["R"]["post_ant"], g["occ"]["L"]["post_ant"]))
    if "robustness" in res:
        r = res["robustness"]["post_ant_ranges_uncorrected_occ"]
        rows.append(("Perturbations, rigid only (12), measured (range)", tuple(r["R"]["rigid_only"]), tuple(r["L"]["rigid_only"])))
        rows.append(("Perturbations, all (+split +/-4 mm, 1-mm erosion), measured (range)", tuple(r["R"]["all"]), tuple(r["L"]["all"])))
    for t in res["trimmed"]:
        rows.append((f"Posterior tail trimmed {t['trim_mm']} mm: measured / normal-gradient expectation",
                     (t["measured"]["R"]["post_ant"], t["expected_normal_gradient"]["R"]["post_ant"]),
                     (t["measured"]["L"]["post_ant"], t["expected_normal_gradient"]["L"]["post_ant"])))
    return rows


def _fmt(v, kind):
    if isinstance(v, tuple):
        sep = " - " if "range" in kind or "Perturb" in kind else " / "
        return sep.join(f"{a:.2f}" for a in v)
    return f"{v:.2f}"


def fig_ratio_table(res, path):
    plt = _plt()
    rows = ratio_rows(res)
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.3))
    ax.axis("off")
    cells = [[n, _fmt(r, n), _fmt(l, n)] for n, r, l in rows]
    t = ax.table(cellText=cells, colLabels=["Posterior / anterior putamen SBR ratio", "Right", "Left"],
                 loc="center", cellLoc="left", colWidths=[0.70, 0.15, 0.15])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.35)
    ax.set_title("Posterior/anterior putamen ratio. " + DISCLAIMER, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def write_all(anat, res, out: Path):
    out = Path(out)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    qc_figure(anat.ct, anat.mg, anat.ac, str(fig / "fig1_registration_overlay.png"))
    fig_segmentation(anat, fig / "fig2_segmentation.png")
    fig_profile(res, fig / "fig3_putamen_profile.png")
    fig_ratio_table(res, fig / "fig4_ratio_table.png")
    fig_sweep(res, fig / "fig5_voi_sweep.png")
    return write_json(res, out / "results.json")
