"""Generate the ten manuscript figures as vector PDF.

Run:  .venv/bin/python analysis/make_figures.py [--only figure8 figure10]

Every figure is produced from results/ artifacts; none contains hand-entered
numbers. Figures 1 and 2 need no Sobol output and are drawn whenever the script
runs; the rest are skipped with a notice if their inputs are missing.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import config as C
import figstyle as FS
from topology import load_topology

FIGDIR = C.REPO / "figures"
SEQ = FS.sequential_cmap()

# Figure labels carry the accented name; config.py and figstyle.py keep the
# ASCII form they use elsewhere.
ACCENTED = {"Castilla y Leon": "Castilla y Le\u00f3n"}
for _st in FS.SCENARIO_STYLE.values():
    _st["label"] = ACCENTED.get(_st["label"], _st["label"])

PRETTY = {
    "alpha": r"$\alpha$", "eta_det": r"$\eta_{\mathrm{det}}$",
    "p_dc": r"$p_{\mathrm{dc}}$", "e_d": r"$e_{\mathrm{d}}$",
    "mu": r"$\mu$", "f_EC": r"$f_{\mathrm{EC}}$", "L0": r"$L_0$",
}
OUT_PRETTY = {
    "median_skr_log10": r"$\widetilde{R}$", "reach_km": r"$d_{\max}$",
    "usable_frac": r"$\rho_E$", "lcc_frac": r"$S_{\mathrm{LCC}}$",
    "reachable_pairs": r"$\rho_P$", "global_eff": r"$E_G$",
    "bottleneck_log10": r"$\widetilde{B}$", "isolated_frac": r"$\rho_I$",
}
PRIMARY_TAG = f"{C.R_MIN_PRIMARY:.0e}".replace("+0", "").replace("+", "")


def rate_label(r_min: float) -> str:
    """A rate as mantissa x power of ten, e.g. 5e6 -> $5\\times10^{6}$.

    Truncating log10 to an integer, as the Figure 8 legend once did, printed
    the three thresholds 1e6, 5e6 and 2e7 as 10^6, 10^6 and 10^7.
    """
    exp = int(np.floor(np.log10(r_min) + 1e-9))
    mant = r_min / 10.0 ** exp
    if abs(mant - 1.0) < 1e-9:
        return f"$10^{{{exp}}}$"
    return f"${mant:g}\\times10^{{{exp}}}$"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_indices(path):
    if not path.exists():
        return None
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k in ("S1", "S1_conf", "ST", "ST_conf", "output_std"):
                r[k] = float(r[k]) if r[k] not in ("", "nan") else np.nan
            r["degenerate"] = r["degenerate"] == "True"
            r["n_base"] = int(r["n_base"])
            rows.append(r)
    return rows


def load_raw(scenario):
    p = C.RESULTS / f"raw_outputs_{scenario}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return d["X"], d["Y"], [str(c) for c in d["columns"]]


# --------------------------------------------------------------------------
# Figure 1 - pipeline schematic
# --------------------------------------------------------------------------
def figure1():
    fig, ax = plt.subplots(figsize=(FS.W2, 1.75))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 22)
    ax.grid(False)

    steps = ["Sobol\nparameter sample", "Decoy-state\nBB84 link model",
             "SKR-annotated\ncandidate links", "Usable graph\n$R \\geq R_{\\min}$",
             "Link & network\nmetrics", "Sobol indices\n$S_i$ and $S_{T_i}$"]
    w, gap = 13.5, 3.8
    x = 1.0
    for i, s in enumerate(steps):
        # One uniform tint for every stage. A ramp across the boxes would imply
        # a magnitude that does not exist here, and its darker steps leave the
        # label text below the contrast floor. Order is carried by the arrows.
        ax.add_patch(FancyBboxPatch(
            (x, 6), w, 10, boxstyle="round,pad=0.35,rounding_size=0.8",
            linewidth=0.7, edgecolor=FS.BLUE, facecolor=FS.SEQ_BLUE[0]))
        ax.text(x + w / 2, 11, s, ha="center", va="center", fontsize=7,
                color=FS.INK, linespacing=1.35)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w + 0.35, 11), (x + w + gap - 0.35, 11),
                arrowstyle="-|>", mutation_scale=7,
                linewidth=0.8, color=FS.INK_2))
        x += w + gap

    ax.text(1.0, 2.6,
            "Two geographical scenarios; three operational thresholds; "
            f"$N = {C.N_BASE}$ base samples, {C.N_BASE * 9:,} model evaluations "
            "per scenario.",
            fontsize=6.5, color=FS.INK_2)
    return FS.save(fig, FIGDIR / "figure1_pipeline.pdf")


# --------------------------------------------------------------------------
# Figure 2 - topologies
# --------------------------------------------------------------------------
def figure2():
    fig, axes = plt.subplots(1, 2, figsize=(FS.W2, 3.5))
    for ax, key in zip(axes, ("cyl", "spain")):
        topo = load_topology(key)
        ax.grid(False)
        segs = np.stack([
            np.column_stack([topo.lon[topo.edge_u], topo.lat[topo.edge_u]]),
            np.column_stack([topo.lon[topo.edge_v], topo.lat[topo.edge_v]]),
        ], axis=1)
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(
            segs, linewidths=0.18, colors=FS.BLUE, alpha=0.30, zorder=1))
        # The dense scenario needs smaller nodes or the markers merge into a
        # solid mass and the link structure disappears beneath them.
        ax.scatter(topo.lon, topo.lat, s=1.6 if key == "cyl" else 0.5,
                   color=FS.INK, zorder=2, linewidths=0)
        ax.set_aspect(1.0 / np.cos(np.radians(float(np.mean(topo.lat)))))
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_title(f"{ACCENTED.get(topo.label, topo.label)}: "
                     f"{topo.n_nodes} nodes, "
                     f"{topo.n_edges:,} candidate links", fontsize=8)

        inset = ax.inset_axes([0.60, 0.06, 0.38, 0.26])
        inset.hist(topo.edge_km, bins=30, color=FS.BLUE, alpha=0.85,
                   linewidth=0)
        inset.set_xlabel("link length (km)", fontsize=5.5, labelpad=1)
        inset.tick_params(labelsize=5, pad=1)
        inset.set_yticks([])
        inset.grid(False)
        for sp in ("top", "right", "left"):
            inset.spines[sp].set_visible(False)
    return FS.save(fig, FIGDIR / "figure2_topologies.pdf")


# --------------------------------------------------------------------------
# Figure 3 - output distributions (ECDF small multiples)
# --------------------------------------------------------------------------
def figure3():
    data = {k: load_raw(k) for k in C.SCENARIOS}
    if any(v is None for v in data.values()):
        return None
    cols = data["cyl"][2]
    show = ["median_skr_log10"] + [f"{n}@{PRIMARY_TAG}" for n, _, _ in
                                   C.OUTPUTS_THRESHOLDED]
    show = [c for c in show if c in cols]

    ncol = 4
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FS.W2, 1.75 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, show):
        j = cols.index(col)
        for key in ("cyl", "spain"):
            y = np.sort(data[key][1][:, j])
            ax.step(y, np.linspace(0, 1, len(y)), where="post",
                    color=FS.SCENARIO_STYLE[key]["color"],
                    ls=FS.SCENARIO_STYLE[key]["ls"],
                    label=FS.SCENARIO_STYLE[key]["label"], linewidth=1.2)
        base = col.split("@")[0]
        ax.set_title(OUT_PRETTY.get(base, base), fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("ECDF")
    for ax in axes[len(show):]:
        ax.set_axis_off()

    axes[0].legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    return FS.save(fig, FIGDIR / "figure3_output_distributions.pdf")


# --------------------------------------------------------------------------
# Figures 4 - link-level Sobol indices
# --------------------------------------------------------------------------
def _bars(ax, rows, output, scenario):
    sub = {r["parameter"]: r for r in rows
           if r["output"] == output and r["scenario"] == scenario}
    if not sub or all(s["degenerate"] for s in sub.values()):
        ax.text(0.5, 0.5, "output constant\n(indices undefined)",
                ha="center", va="center", fontsize=6.5, color=FS.INK_2,
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    names = C.PARAM_NAMES
    x = np.arange(len(names))
    s1 = [sub[n]["S1"] for n in names]
    s1e = [sub[n]["S1_conf"] for n in names]
    st = [sub[n]["ST"] for n in names]
    ste = [sub[n]["ST_conf"] for n in names]
    ax.bar(x - 0.2, s1, 0.38, yerr=s1e, color=FS.BLUE, label="$S_i$",
           linewidth=0, error_kw=dict(lw=0.6, ecolor=FS.INK_2))
    ax.bar(x + 0.2, st, 0.38, yerr=ste, color=FS.ORANGE, label="$S_{T_i}$",
           hatch="///", edgecolor=FS.SURFACE, linewidth=0.6,
           error_kw=dict(lw=0.6, ecolor=FS.INK_2))
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[n] for n in names])
    ax.axhline(0, color=FS.AXIS, linewidth=0.6)


def figure4(rows):
    outputs = ["median_skr_log10", f"reach_km@{PRIMARY_TAG}",
               f"usable_frac@{PRIMARY_TAG}"]
    fig, axes = plt.subplots(2, 3, figsize=(FS.W2, 4.0), sharey=True)
    for i, key in enumerate(("cyl", "spain")):
        for j, out in enumerate(outputs):
            ax = axes[i, j]
            _bars(ax, rows, out, key)
            if i == 0:
                ax.set_title(OUT_PRETTY.get(out.split("@")[0], out), fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{FS.SCENARIO_STYLE[key]['label']}\nSobol index")
    axes[0, 0].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    return FS.save(fig, FIGDIR / "figure4_sobol_link.pdf")


# --------------------------------------------------------------------------
# Figure 5 - network-level total-order heatmaps
# --------------------------------------------------------------------------
def figure5(rows):
    outs = [f"{n}@{PRIMARY_TAG}" for n, _, _ in C.OUTPUTS_THRESHOLDED]
    fig, axes = plt.subplots(1, 2, figsize=(FS.W2, 3.1))
    im = None
    for ax, key in zip(axes, ("cyl", "spain")):
        M = np.full((len(outs), len(C.PARAM_NAMES)), np.nan)
        for a, o in enumerate(outs):
            for b, p in enumerate(C.PARAM_NAMES):
                r = next((r for r in rows if r["scenario"] == key
                          and r["output"] == o and r["parameter"] == p), None)
                if r and not r["degenerate"]:
                    M[a, b] = r["ST"]
        im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(C.PARAM_NAMES)))
        ax.set_xticklabels([PRETTY[n] for n in C.PARAM_NAMES])
        ax.set_yticks(range(len(outs)))
        ax.set_yticklabels([OUT_PRETTY.get(o.split("@")[0], o) for o in outs])
        ax.set_title(FS.SCENARIO_STYLE[key]["label"], fontsize=8)
        ax.grid(False)
        # Printed values keep the panel readable in greyscale.
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                v = M[a, b]
                if np.isnan(v):
                    ax.text(b, a, "--", ha="center", va="center", fontsize=5.5,
                            color=FS.INK_2)
                else:
                    ax.text(b, a, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.5,
                            color=FS.SURFACE if v > 0.55 else FS.INK)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("$S_{T_i}$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return FS.save(fig, FIGDIR / "figure5_sobol_network.pdf")


# --------------------------------------------------------------------------
# Figures 6 and 7 - response surfaces and the connectivity transition
# --------------------------------------------------------------------------
def _surface(ax, d, scenario, xname, yname, metric_idx, levels=None):
    tag = f"{scenario}_{xname}_{yname}"
    Z = d[f"{tag}_Z"][:, :, metric_idx]
    xs, ys = d[f"{tag}_x"], d[f"{tag}_y"]
    mesh = ax.pcolormesh(xs, ys, Z, cmap=SEQ, vmin=0, vmax=1, shading="auto",
                         rasterized=True, linewidth=0, antialiased=False)
    if levels:
        cs = ax.contour(xs, ys, Z, levels=levels, colors=FS.INK,
                        linewidths=0.7)
        ax.clabel(cs, inline=True, fontsize=5.5, fmt="%.2f")
    if xname == "p_dc":
        ax.set_xscale("log")
    ax.set_xlabel(PRETTY[xname])
    ax.set_ylabel(PRETTY[yname])
    ax.grid(False)
    return mesh


def figure6(d):
    fig, axes = plt.subplots(2, 2, figsize=(FS.W2, 5.0))
    mesh = None
    for i, key in enumerate(("cyl", "spain")):
        for j, (xn, yn) in enumerate([("alpha", "eta_det"),
                                      ("p_dc", "eta_det")]):
            mesh = _surface(axes[i, j], d, key, xn, yn, 0)
            if i == 0:
                axes[i, j].set_title(f"{PRETTY[yn]} vs {PRETTY[xn]}",
                                     fontsize=8)
            if j == 0:
                axes[i, j].set_ylabel(
                    f"{FS.SCENARIO_STYLE[key]['label']}\n{PRETTY[yn]}")
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("$S_{\\mathrm{LCC}}$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return FS.save(fig, FIGDIR / "figure6_response_surfaces.pdf")


def figure7(d):
    fig, axes = plt.subplots(1, 2, figsize=(FS.W2, 2.9))
    mesh = None
    for ax, key in zip(axes, ("cyl", "spain")):
        mesh = _surface(ax, d, key, "alpha", "eta_det", 0,
                        levels=[0.50, 0.75, 0.90, 0.99])
        ax.set_title(FS.SCENARIO_STYLE[key]["label"], fontsize=8)
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("$S_{\\mathrm{LCC}}$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return FS.save(fig, FIGDIR / "figure7_connectivity_transition.pdf")


# --------------------------------------------------------------------------
# Figure 8 - convergence and threshold robustness
# --------------------------------------------------------------------------
def figure8(conv, rows):
    if not conv:
        return None
    target = f"lcc_frac@{PRIMARY_TAG}"
    fig, axes = plt.subplots(2, 4, figsize=(FS.W2, 3.8))
    axes = axes.ravel()

    for ax, p in zip(axes, C.PARAM_NAMES):
        for key in ("cyl", "spain"):
            pts = sorted(
                [(r["n_base"], r["ST"], r["ST_conf"]) for r in conv
                 if r["scenario"] == key and r["output"] == target
                 and r["parameter"] == p and not r["degenerate"]])
            if not pts:
                continue
            n, st, err = map(np.array, zip(*pts))
            st_style = FS.SCENARIO_STYLE[key]
            ax.errorbar(n, st, yerr=err, color=st_style["color"],
                        ls=st_style["ls"], marker=st_style["marker"],
                        label=st_style["label"], linewidth=1.1)
        ax.set_xscale("log", base=2)
        ax.set_title(PRETTY[p], fontsize=8)
        ax.set_ylim(bottom=0)

    # Final panel: threshold robustness of the parameter ranking.
    ax = axes[len(C.PARAM_NAMES)]
    tags = [f"{r:.0e}".replace("+0", "").replace("+", "") for r in C.R_MIN_SWEEP]
    width = 0.8 / len(tags)
    xs = np.arange(len(C.PARAM_NAMES))
    for t, tag in enumerate(tags):
        vals, errs = [], []
        for p in C.PARAM_NAMES:
            r = next((r for r in rows if r["scenario"] == "spain"
                      and r["output"] == f"lcc_frac@{tag}"
                      and r["parameter"] == p and not r["degenerate"]), None)
            vals.append(r["ST"] if r else np.nan)
            errs.append(r["ST_conf"] if r else np.nan)
        ax.bar(xs + (t - 1) * width, vals, width, yerr=errs,
               color=SEQ(0.30 + 0.28 * t), linewidth=0,
               error_kw=dict(lw=0.5, ecolor=FS.INK_2, capsize=0),
               label=rate_label(C.R_MIN_SWEEP[t]))
    ax.set_xticks(xs)
    ax.set_xticklabels([PRETTY[n] for n in C.PARAM_NAMES], fontsize=6)
    ax.set_title("$S_{T_i}$ by $R_{\\min}$, mainland Spain", fontsize=7.5)
    ax.set_ylabel("$S_{T_i}$", fontsize=7)
    # headroom so the legend clears the tallest bar
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1]) * 1.25)
    ax.legend(fontsize=5, ncol=3, loc="upper center",
              title="$R_{\\min}$ (bit/s)", title_fontsize=5, handlelength=0.8,
              handletextpad=0.3, columnspacing=0.7, borderaxespad=0.2)

    for ax in axes[len(C.PARAM_NAMES) + 1:]:
        ax.set_axis_off()
    axes[0].legend(fontsize=5.5, loc="best")
    axes[0].set_ylabel("$S_{T_i}$")
    for ax in axes[:len(C.PARAM_NAMES)]:
        ax.set_xlabel("$N$", fontsize=7, labelpad=1)
        ax.set_xticks(list(C.CONVERGENCE_SIZES))
        ax.set_xticklabels([str(n) for n in C.CONVERGENCE_SIZES],
                           fontsize=5, rotation=45)
    fig.tight_layout()
    return FS.save(fig, FIGDIR / "figure8_convergence.pdf")


# --------------------------------------------------------------------------
# Figure 9 - interaction shares, decoy-state model vs surrogate at N = 1024
# --------------------------------------------------------------------------
def figure9():
    """Interaction share of every output, full model and surrogate, N = 1024.

    Drawn from results/interaction_shares.csv (stored coding of the empty
    network), the rows on an equal footing: the same 1024 base rows and the
    same bootstrap resampling matrix for both models. Outputs are in the
    order of C.OUTPUTS_THRESHOLDED (the order of the manuscript's output
    table), preceded by the threshold-free link rate, which is therefore the
    same point in the three columns. No ordering is chosen from the values.
    """
    path = C.RESULTS / "interaction_shares.csv"
    if not path.exists():
        return None
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["coding"] != "stored" or int(r["n_base"]) != 1024:
                continue
            rows[(r["scenario"], r["model"], r["output"])] = (
                float(r["share_raw"]), float(r["share_ci_lo"]),
                float(r["share_ci_hi"]))

    tags = [f"{r:.0e}".replace("+0", "").replace("+", "") for r in C.R_MIN_SWEEP]
    metrics = [n for n, _, _ in C.OUTPUTS_THRESHOLDED]
    labels = [OUT_PRETTY["median_skr_log10"]] + [OUT_PRETTY[m] for m in metrics]
    x = np.arange(len(labels))
    models = [("full", "decoy-state model", FS.BLUE, "o", FS.BLUE),
              ("surrogate", "interaction-free surrogate", FS.ORANGE, "s",
               "none")]
    dx = 0.17
    scenarios = list(C.SCENARIOS)

    fig, axes = plt.subplots(len(scenarios), len(tags), figsize=(FS.W2, 4.2),
                             sharex=True, sharey=True)
    for i, scen in enumerate(scenarios):
        for j, (r_min, tag) in enumerate(zip(C.R_MIN_SWEEP, tags)):
            ax = axes[i, j]
            outs = ["median_skr_log10"] + [f"{m}@{tag}" for m in metrics]
            for k, (model, label, colour, marker, face) in enumerate(models):
                v = np.array([rows[(scen, model, o)] for o in outs])
                xo = x + (k - 0.5) * 2 * dx
                ax.errorbar(xo, v[:, 0],
                            yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]],
                            fmt=marker, color=colour, markerfacecolor=face,
                            markeredgecolor=colour, markersize=3.6,
                            elinewidth=0.9, capsize=1.5, linestyle="none",
                            label=label)
            ax.axhline(0, color=FS.AXIS, linewidth=0.6, zorder=0)
            # the link rate has no threshold: separate it from the rest
            ax.axvline(0.5, color=FS.GRID, linewidth=0.8, ls=":", zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=6.5)
            ax.set_xlim(-0.6, len(labels) - 0.4)
            ax.set_ylim(-0.15, 1.12)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(f"$R_{{\\min}}=$ {rate_label(r_min)} bit/s",
                             fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{FS.SCENARIO_STYLE[scen]['label']}\n"
                              "interaction share $I$", fontsize=7.5)
    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=2, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return FS.save(fig, FIGDIR / "figure9_interaction_shares.pdf")


# --------------------------------------------------------------------------
# Figure 10 - largest-component distributions by detector generation
# --------------------------------------------------------------------------
def figure10():
    """Empirical distribution of S_LCC per generation, scenario and threshold.

    Cumulative curves rather than histograms: most cells have a point mass at
    one end (every node connected, or none), which a histogram bin hides and a
    step in the cumulative curve shows at its true size.
    """
    path = C.RESULTS / "generation_samples.npz"
    if not path.exists():
        return None
    d = np.load(path)
    series = [("ingaas", "InGaAs/InP range", FS.BLUE, "-"),
              ("intermediate", "intermediate range", FS.ORANGE, "--"),
              ("snspd", "SNSPD range", FS.AQUA, ":")]
    tags = [f"{r:.0e}".replace("+0", "").replace("+", "") for r in C.R_MIN_SWEEP]
    scenarios = list(C.SCENARIOS)

    fig, axes = plt.subplots(len(scenarios), len(tags), figsize=(FS.W2, 3.7),
                             sharex=True, sharey=True)
    for i, scen in enumerate(scenarios):
        for j, (r_min, tag) in enumerate(zip(C.R_MIN_SWEEP, tags)):
            ax = axes[i, j]
            for key, label, colour, ls in series:
                x = np.sort(d[f"{key}_{scen}_{tag}"])
                # step from (0, 0) to (1, 1) through the sorted sample
                xs = np.concatenate([[0.0], x, [1.0]])
                ys = np.concatenate([[0.0], np.arange(1, len(x) + 1) / len(x),
                                     [1.0]])
                ax.step(xs, ys, where="post", color=colour, ls=ls,
                        linewidth=1.5, label=label)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
            if i == 0:
                ax.set_title(f"$R_{{\\min}}=$ {rate_label(r_min)} bit/s",
                             fontsize=8)
            if i == len(scenarios) - 1:
                ax.set_xlabel("$S_{\\mathrm{LCC}}$")
            if j == 0:
                ax.set_ylabel(f"{FS.SCENARIO_STYLE[scen]['label']}\n"
                              "cumulative fraction", fontsize=7.5)
    axes[0, 1].legend(loc="center", fontsize=7, handlelength=2.6,
                      title="detector-parameter range", title_fontsize=7)
    fig.tight_layout()
    return FS.save(fig, FIGDIR / "figure10_generations.pdf")


# --------------------------------------------------------------------------
def main(only=None):
    FS.apply()
    FIGDIR.mkdir(exist_ok=True)
    made, skipped = [], []

    want = lambda name: only is None or name in only

    for fn in (figure1, figure2):
        if want(fn.__name__):
            made.append(fn())

    rows = load_indices(C.RESULTS / "sobol_indices.csv")
    conv = load_indices(C.RESULTS / "convergence.csv")
    surf_path = C.RESULTS / "response_surfaces.npz"

    if rows:
        if want("figure3"):
            p3 = figure3()
            made.append(p3) if p3 else skipped.append("figure3")
        if want("figure4"):
            made.append(figure4(rows))
        if want("figure5"):
            made.append(figure5(rows))
        if want("figure8"):
            p8 = figure8(conv, rows)
            made.append(p8) if p8 else skipped.append("figure8")
    else:
        skipped += [f for f in ("figure3", "figure4", "figure5", "figure8")
                    if want(f)]

    if want("figure9"):
        p9 = figure9()
        made.append(p9) if p9 else skipped.append("figure9")

    if surf_path.exists():
        d = np.load(surf_path)
        if want("figure6"):
            made.append(figure6(d))
        if want("figure7"):
            made.append(figure7(d))
    else:
        skipped += [f for f in ("figure6", "figure7") if want(f)]

    if want("figure10"):
        p10 = figure10()
        made.append(p10) if p10 else skipped.append("figure10")

    for p in made:
        if p:
            print(f"  wrote {p.relative_to(C.REPO)}")
    if skipped:
        print(f"  skipped (inputs missing): {', '.join(skipped)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", metavar="figureN",
                    help="regenerate only these figures, e.g. --only figure8 "
                         "figure10")
    main(ap.parse_args().only)
