"""Facts needed to correct over-general statements (roadmap R3-9).

Run:  .venv/bin/python analysis/claim_support.py
Reads  results/sobol_indices.csv, convergence.csv, robustness_indices.csv,
       response_surfaces.npz, and (optional) raw_outputs_<scenario>.npz
Writes results/claim_support.json, results/rank_orders.csv

Sections of the output file:

  dark_counts            (i)   pointwise effect of the dark-count probability
                               on the key rate, at the worst corner of the
                               sampled domain and across the principal design
  rank_orders            (ii)  the order of all seven inputs for every output,
                               threshold and scenario, in the principal run
                               and under the 1.3 route factor, with ties
                               (overlapping intervals between adjacent ranks)
  rank_facts                   statements derived from (ii)
  convergence            (iii) outputs exceeding the 0.02 criterion, and where
                               rankings are and are not stable from N = 1024
  rate_bounds            (iv)  the rate below which every candidate link is
                               usable for every parameter combination
  zero_length_links      (v)   effect of the three zero-length candidate links
  single_scalar_reduction      check that the structural outputs are functions
                               of the usable-link count alone
  response_surface_facts       saturation of S_LCC and flatness in p_dc

All randomness is seeded from config.SEED.
"""

from __future__ import annotations

import csv
import itertools
import json

import numpy as np
from SALib.sample import sobol as sobol_sample

import config as C
import prefix_metrics as PM
import sobol_tools as ST
from generation_rankings import load_control_rows, ranked_table
from link_model import secret_key_rate
from summarize_results import read_rows
from topology import load_topology

TAGS = {r: f"{r:.0e}".replace("+0", "").replace("+", "") for r in C.R_MIN_SWEEP}
PRIMARY = TAGS[C.R_MIN_PRIMARY]
CRITERION = 0.02
ALL_OUTPUTS = ["median_skr_log10"] + [f"{n}@{t}" for t in TAGS.values()
                                      for n, _, _ in C.OUTPUTS_THRESHOLDED]
PRIMARY_OUTPUTS = ["median_skr_log10"] + [f"{n}@{PRIMARY}"
                                          for n, _, _ in C.OUTPUTS_THRESHOLDED]


def bounds():
    return {n: ((10.0 ** lo, 10.0 ** hi) if lg else (lo, hi))
            for n, (lo, hi), lg, _, _ in C.PARAMETERS}


def centre():
    return {n: (10.0 ** (0.5 * (lo + hi)) if lg else 0.5 * (lo + hi))
            for n, (lo, hi), lg, _, _ in C.PARAMETERS}


def rate(d_km, **p) -> float:
    return float(secret_key_rate(np.array([d_km]), **p)[0])


def design():
    X = sobol_sample.sample(C.SALIB_PROBLEM, C.N_BASE,
                            calc_second_order=C.CALC_SECOND_ORDER, seed=C.SEED)
    P = X.copy()
    for k, name in enumerate(C.PARAM_NAMES):
        if name in C.LOG_PARAMS:
            P[:, k] = 10.0 ** P[:, k]
    return P


# --------------------------------------------------------------------------
# (i) dark counts
# --------------------------------------------------------------------------
def dark_counts(P) -> dict:
    b = bounds()
    lo_dc, hi_dc = b["p_dc"]
    others = [n for n in C.PARAM_NAMES if n != "p_dc"]
    out = {"p_dc_low": lo_dc, "p_dc_high": hi_dc,
           "definition": "relative reduction = 1 - R(p_dc high) / R(p_dc low), "
                         "all other inputs held fixed",
           "by_length_km": {}}
    for d_km in (45.0, 28.0):
        worst = None
        for combo in itertools.product(*[b[n] for n in others]):
            p = dict(zip(others, combo))
            r_lo, r_hi = rate(d_km, p_dc=lo_dc, **p), rate(d_km, p_dc=hi_dc, **p)
            red = 1.0 - r_hi / r_lo if r_lo > 0 else float("nan")
            if worst is None or red > worst["relative_reduction"]:
                worst = dict(parameters=p, rate_p_dc_low_bps=r_lo,
                             rate_p_dc_high_bps=r_hi, relative_reduction=red)
        p = worst["parameters"]
        eta = p["eta_det"] * 10.0 ** (-(p["alpha"] * d_km + p["L0"]) / 10.0)
        worst["eta_times_mu"] = eta * p["mu"]
        worst["Y0_at_p_dc_high"] = 1.0 - (1.0 - hi_dc) ** 2
        c = {k: v for k, v in centre().items() if k != "p_dc"}
        c_lo, c_hi = rate(d_km, p_dc=lo_dc, **c), rate(d_km, p_dc=hi_dc, **c)
        eta_c = c["eta_det"] * 10.0 ** (-(c["alpha"] * d_km + c["L0"]) / 10.0)
        # Across the principal design: each row's other six inputs, p_dc swept
        # over its full range.
        cols = [C.PARAM_NAMES.index(n) for n in others]
        Q = P[:, cols]
        kw = {n: Q[:, i] for i, n in enumerate(others)}
        # secret_key_rate is written for an array of lengths and scalar
        # parameters; with a single length and array parameters it broadcasts.
        r_lo = secret_key_rate(np.array([d_km]), p_dc=lo_dc, **kw)
        r_hi = secret_key_rate(np.array([d_km]), p_dc=hi_dc, **kw)
        red = 1.0 - r_hi / r_lo
        out["by_length_km"][f"{d_km:g}"] = {
            "worst_corner": worst,
            "range_centre": {"rate_p_dc_low_bps": c_lo, "rate_p_dc_high_bps": c_hi,
                             "relative_reduction": 1.0 - c_hi / c_lo,
                             "eta_times_mu": eta_c * c["mu"]},
            "across_principal_design": {
                "n_samples": int(len(red)),
                "median": float(np.median(red)),
                "q95": float(np.quantile(red, 0.95)),
                "q99": float(np.quantile(red, 0.99)),
                "max": float(red.max()),
                "frac_samples_reduction_above_5pct": float((red > 0.05).mean()),
                "frac_samples_reduction_above_10pct": float((red > 0.10).mean()),
            },
        }
    return out


# --------------------------------------------------------------------------
# (ii) rank orders
# --------------------------------------------------------------------------
def rank_orders(rows):
    """Rank tables for the principal run and the route-factor control."""
    tables, flat = {"principal": {}, "route_x1.3": {}}, []
    for scen in C.SCENARIOS:
        tables["principal"][scen] = {}
        for o in ALL_OUTPUTS:
            sub = [r for r in rows if r["scenario"] == scen and r["output"] == o
                   and not r["degenerate"]]
            if sub:
                tables["principal"][scen][o] = ranked_table(sub)
    for (config, scen, _, o), sub in load_control_rows(("route",)).items():
        tables["route_x1.3"].setdefault(scen, {})[o] = ranked_table(sub)

    out = {}
    for design_name, by_scen in tables.items():
        out[design_name] = {}
        for scen, by_out in by_scen.items():
            out[design_name][scen] = {}
            for o in ALL_OUTPUTS:
                if o not in by_out:
                    continue
                t = by_out[o]
                out[design_name][scen][o] = {
                    "order": [r["parameter"] for r in t],
                    "ST": [r["ST"] for r in t],
                    "ST_conf": [r["ST_conf"] for r in t],
                    "tie_with_next_rank": [r["overlaps_next_rank"] for r in t],
                    "ranking_resolved": t[0]["ranking_resolved"],
                }
                for r in t:
                    flat.append(dict(
                        design=design_name, scenario=scen, output=o,
                        parameter=r["parameter"], rank=r["rank"], ST=r["ST"],
                        ST_conf=r["ST_conf"], ST_lo=r["ST_lo"], ST_hi=r["ST_hi"],
                        tie_with_next_rank=r["overlaps_next_rank"],
                        tie_with_previous_rank=r["overlaps_previous_rank"],
                        ranking_resolved=r["ranking_resolved"]))
    return out, flat


def rank_facts(ro) -> dict:
    facts = {}
    for design_name, by_scen in ro.items():
        f = {}
        for scen, by_out in by_scen.items():
            prim = {o: by_out[o] for o in PRIMARY_OUTPUTS if o in by_out}
            every = by_out

            def rank_of(tab, p):
                return tab["order"].index(p) + 1

            def tied(tab, p, q):
                """Do the intervals of p and q overlap?"""
                i, j = tab["order"].index(p), tab["order"].index(q)
                lo = max(tab["ST"][i] - tab["ST_conf"][i],
                         tab["ST"][j] - tab["ST_conf"][j])
                hi = min(tab["ST"][i] + tab["ST_conf"][i],
                         tab["ST"][j] + tab["ST_conf"][j])
                return bool(lo <= hi)

            f[scen] = {
                "eta_det_first_on_n_of_primary_outputs":
                    [sum(t["order"][0] == "eta_det" for t in prim.values()), len(prim)],
                "eta_det_first_on_n_of_all_outputs":
                    [sum(t["order"][0] == "eta_det" for t in every.values()), len(every)],
                "top3_is_eta_det_e_d_L0_on_n_of_primary_outputs":
                    [sum(t["order"][:3] == ["eta_det", "e_d", "L0"]
                         for t in prim.values()), len(prim)],
                "top2_is_eta_det_e_d_on_n_of_all_outputs":
                    [sum(t["order"][:2] == ["eta_det", "e_d"]
                         for t in every.values()), len(every)],
                "primary_outputs_where_top3_differs":
                    {o: t["order"][:3] for o, t in prim.items()
                     if t["order"][:3] != ["eta_det", "e_d", "L0"]},
                "alpha_rank_by_output": {o: rank_of(t, "alpha")
                                         for o, t in every.items()},
                "alpha_best_rank_primary": min(rank_of(t, "alpha")
                                               for t in prim.values()),
                "alpha_best_rank_all": min(rank_of(t, "alpha")
                                           for t in every.values()),
                "alpha_vs_L0_intervals_overlap_primary":
                    {o: tied(t, "alpha", "L0") for o, t in prim.items()},
                "alpha_minus_L0_ST_primary":
                    {o: t["ST"][t["order"].index("alpha")]
                        - t["ST"][t["order"].index("L0")] for o, t in prim.items()},
                "alpha_vs_mu_intervals_overlap_primary":
                    {o: tied(t, "alpha", "mu") for o, t in prim.items()},
                "ranks_4_to_7_primary": {o: t["order"][3:] for o, t in prim.items()},
                "f_EC_outranks_alpha_primary":
                    {o: {"f_EC_ST": t["ST"][t["order"].index("f_EC")],
                         "f_EC_conf": t["ST_conf"][t["order"].index("f_EC")],
                         "alpha_ST": t["ST"][t["order"].index("alpha")],
                         "alpha_conf": t["ST_conf"][t["order"].index("alpha")],
                         "intervals_overlap": tied(t, "f_EC", "alpha")}
                     for o, t in prim.items()
                     if rank_of(t, "f_EC") < rank_of(t, "alpha")},
                "unresolved_outputs": [o for o, t in every.items()
                                       if not t["ranking_resolved"]],
                "eta_det_ST_range_primary": [
                    min(t["ST"][t["order"].index("eta_det")] for t in prim.values()),
                    max(t["ST"][t["order"].index("eta_det")] for t in prim.values())],
                "eta_det_ST_range_all_resolved": [
                    min(t["ST"][t["order"].index("eta_det")]
                        for t in every.values() if t["ranking_resolved"]),
                    max(t["ST"][t["order"].index("eta_det")]
                        for t in every.values() if t["ranking_resolved"])],
                "max_ST_p_dc": max(
                    ((t["ST"][t["order"].index("p_dc")],
                      t["ST_conf"][t["order"].index("p_dc")], o)
                     for o, t in every.items()), key=lambda x: x[0]),
                "max_ST_p_dc_primary": max(
                    t["ST"][t["order"].index("p_dc")] for t in prim.values()),
                "max_ST_f_EC_primary": max(
                    ((t["ST"][t["order"].index("f_EC")], o)
                     for o, t in prim.items()), key=lambda x: x[0]),
                "max_ST_mu_primary": max(
                    ((t["ST"][t["order"].index("mu")], o)
                     for o, t in prim.items()), key=lambda x: x[0]),
            }
        # cross-scenario comparison at the primary threshold
        if set(by_scen) >= {"cyl", "spain"}:
            cross = {}
            for o in PRIMARY_OUTPUTS:
                a, b = by_scen["cyl"].get(o), by_scen["spain"].get(o)
                if not a or not b:
                    continue
                sa = dict(zip(a["order"], a["ST"]))
                sb = dict(zip(b["order"], b["ST"]))
                cross[o] = {
                    "same_full_order": a["order"] == b["order"],
                    "same_top3": a["order"][:3] == b["order"][:3],
                    "max_abs_ST_difference": max(abs(sa[p] - sb[p]) for p in sa),
                    "order_cyl": a["order"], "order_spain": b["order"]}
            f["cyl_vs_spain_primary"] = cross
        facts[design_name] = f
    return facts


# --------------------------------------------------------------------------
# (iii) convergence
# --------------------------------------------------------------------------
def convergence(conv, rows) -> dict:
    out = {"criterion_abs_change_ST": CRITERION,
           "note": "change between N = 2048 and N = 4096; the sizes are nested "
                   "subsets of one design, so the change understates "
                   "replicate-to-replicate variability",
           "scenarios": {}}
    for scen in C.SCENARIOS:
        exceed, unstable_top3, unstable_full, per_output = [], [], [], {}
        for o in ALL_OUTPUTS:
            by_n = {}
            for n in C.CONVERGENCE_SIZES:
                d = {r["parameter"]: r["ST"] for r in conv
                     if r["scenario"] == scen and r["output"] == o
                     and r["n_base"] == n and not r["degenerate"]}
                if d:
                    by_n[n] = d
            if 2048 not in by_n or 4096 not in by_n:
                continue
            delta = max(abs(by_n[2048][p] - by_n[4096][p]) for p in by_n[4096])
            order = {n: sorted(d, key=lambda p: -d[p]) for n, d in by_n.items()}
            final = order[4096]
            lead = max((r for r in rows if r["scenario"] == scen
                        and r["output"] == o), key=lambda r: r["ST"])
            resolved = not (lead["ST"] <= 0 or lead["ST_conf"] > 0.25 * lead["ST"])
            top3_from = min((n for n in sorted(order)
                             if all(order[m][:3] == final[:3]
                                    for m in order if m >= n)), default=None)
            full_from = min((n for n in sorted(order)
                             if all(order[m] == final for m in order if m >= n)),
                            default=None)
            per_output[o] = {
                "max_abs_dST_2048_4096": delta,
                "exceeds_criterion": bool(delta > CRITERION),
                "ranking_resolved_at_4096": resolved,
                "top3_unchanged_at_every_N": all(v[:3] == final[:3]
                                                 for v in order.values()),
                "top3_stable_from_N": top3_from,
                "full_order_stable_from_N": full_from,
                "leading_parameter_by_N": {str(n): v[0] for n, v in order.items()},
            }
            if delta > CRITERION:
                exceed.append(o)
            if top3_from is None or top3_from > 1024:
                unstable_top3.append(o)
            if full_from is None or full_from > 1024:
                unstable_full.append(o)
        out["scenarios"][scen] = {
            "outputs_exceeding_criterion": {
                o: {"max_abs_dST": per_output[o]["max_abs_dST_2048_4096"],
                    "ranking_resolved_at_4096":
                        per_output[o]["ranking_resolved_at_4096"]}
                for o in exceed},
            "outputs_exceeding_criterion_primary_threshold":
                [o for o in exceed if o in PRIMARY_OUTPUTS],
            "outputs_whose_top3_is_not_stable_from_N1024": unstable_top3,
            "outputs_whose_full_order_is_not_stable_from_N1024": unstable_full,
            "per_output": per_output,
        }
    return out


# --------------------------------------------------------------------------
# (iv) rate bounds
# --------------------------------------------------------------------------
def rate_bounds(P) -> dict:
    b = bounds()
    out = {"method": "the key rate is monotone in six inputs, so their extreme "
                     "is at a corner of the box; signal intensity is not "
                     "monotone and is searched on a grid of 601 values at "
                     "every corner of the other six. Cross-checked against "
                     "the minimum over the 36,864 rows of the principal "
                     "design.", "f_rep_hz": C.F_REP, "by_scenario": {}}
    mono = [n for n in C.PARAM_NAMES if n != "mu"]
    mus = np.linspace(*b["mu"], 601)

    def extreme(d_km, fn):
        best = None
        for combo in itertools.product(*[b[n] for n in mono]):
            p = dict(zip(mono, combo))
            r = np.array([rate(d_km, mu=m, **p) for m in mus])
            i = int(fn(r))
            if best is None or fn([best[0], r[i]]) == 1:
                best = (float(r[i]), dict(p, mu=float(mus[i])))
        return best

    for scen in C.SCENARIOS:
        topo = load_topology(scen)
        d_max, d_min = float(topo.edge_km.max()), float(topo.edge_km.min())
        lo, lo_p = extreme(d_max, np.argmin)
        hi, hi_p = extreme(d_min, np.argmax)
        hi_longest, _ = extreme(d_max, np.argmax)
        kw = {n: P[:, i] for i, n in enumerate(C.PARAM_NAMES)}
        r_design = secret_key_rate(np.array([d_max]), **kw)
        out["by_scenario"][scen] = {
            "longest_candidate_link_km": d_max,
            "shortest_candidate_link_km": d_min,
            "min_rate_on_longest_link_bps": lo,
            "min_rate_on_longest_link_parameters": lo_p,
            "max_rate_on_longest_link_bps": hi_longest,
            "R_pulse_range_on_longest_link": [lo / C.F_REP, hi_longest / C.F_REP],
            "every_link_usable_in_whole_box_below_bps": lo,
            "max_rate_on_shortest_link_bps": hi,
            "no_link_usable_anywhere_in_box_above_bps": hi,
            "min_rate_on_longest_link_over_principal_design_bps":
                float(r_design.min()),
            "frac_design_rows_with_every_link_usable": {
                tag: float((r_design >= r).mean()) for r, tag in TAGS.items()},
        }
    return out


# --------------------------------------------------------------------------
# (v) zero-length links
# --------------------------------------------------------------------------
def zero_length_links(P, rows) -> dict:
    out = {}
    for scen in C.SCENARIOS:
        topo = load_topology(scen)
        zero = np.flatnonzero(topo.edge_km == 0.0)
        cell = {"n_zero_length_links": int(len(zero)),
                "n_candidate_links": int(topo.n_edges)}
        if len(zero) == 0:
            out[scen] = cell
            continue
        cell["pairs"] = [[topo.names[topo.edge_u[i]], topo.names[topo.edge_v[i]]]
                         for i in zero]

        class Reduced:  # the same topology without the zero-length links
            n_nodes = topo.n_nodes
            edge_u = np.delete(topo.edge_u, zero)
            edge_v = np.delete(topo.edge_v, zero)
            edge_km = np.delete(topo.edge_km, zero)

        with_z, without = PM.build(topo), PM.build(Reduced)
        nz = len(zero)
        K = np.empty((len(P), len(TAGS)), dtype=np.int64)
        for i, row in enumerate(P):
            skr = secret_key_rate(with_z.km, **dict(zip(C.PARAM_NAMES, row)))
            for j, r_min in enumerate(TAGS):
                K[i, j] = PM.usable_count(skr, r_min)
        W = ST.bootstrap_weights(C.N_BASE, 200)
        cell["by_threshold"] = {}
        for j, (r_min, tag) in enumerate(TAGS.items()):
            k = K[:, j]
            # zero-length links are the first nz of the sorted list
            k_red = np.maximum(k - nz, 0)
            only_zero = (k > 0) & (k <= nz)
            block = {"frac_samples_where_only_zero_length_links_are_usable":
                     float(only_zero.mean()),
                     "frac_samples_no_usable_link_with": float((k == 0).mean()),
                     "frac_samples_no_usable_link_without":
                         float((k_red == 0).mean()),
                     "outputs": {}}
            for name in ("lcc_frac", "reachable_pairs", "isolated_frac"):
                a = getattr(with_z, name)[k]
                c = getattr(without, name)[k_red]
                ia, ic = ST.indices(a, W), ST.indices(c, W)
                stored = {r["parameter"]: r for r in rows
                          if r["scenario"] == scen and r["output"] == f"{name}@{tag}"}
                d_st = np.abs(ia["ST"] - ic["ST"])
                kmax = int(d_st.argmax())
                block["outputs"][name] = {
                    "max_abs_output_difference": float(np.abs(a - c).max()),
                    "max_abs_ST_difference": float(d_st.max()),
                    "parameter_of_max_ST_difference": C.PARAM_NAMES[kmax],
                    "stored_ST_conf_of_that_parameter":
                        stored[C.PARAM_NAMES[kmax]]["ST_conf"],
                    "max_abs_S1_difference": float(np.abs(ia["S1"] - ic["S1"]).max()),
                    "max_abs_interaction_share_difference":
                        float(abs(ia["share_raw"] - ic["share_raw"])),
                    "recomputed_ST_matches_stored_within": float(max(
                        abs(ia["ST"][i] - stored[p]["ST"])
                        for i, p in enumerate(C.PARAM_NAMES))),
                }
            cell["by_threshold"][tag] = block
        out[scen] = cell
    return out


# --------------------------------------------------------------------------
# single-scalar reduction, response surfaces
# --------------------------------------------------------------------------
def single_scalar_reduction() -> dict:
    out = {"statement": "with one parameter vector for all links, each "
                        "structural output is a function of the usable-link "
                        "count alone; checked on the stored principal outputs",
           "by_scenario": {}}
    for scen in C.SCENARIOS:
        path = C.RESULTS / f"raw_outputs_{scen}.npz"
        if not path.exists():
            out["by_scenario"][scen] = "raw outputs missing"
            continue
        d = np.load(path, allow_pickle=True)
        Y, cols = d["Y"], [str(c) for c in d["columns"]]
        cell = {}
        for tag in TAGS.values():
            u = Y[:, cols.index(f"usable_frac@{tag}")]
            res = {}
            for name in ("reach_km", "lcc_frac", "reachable_pairs",
                         "isolated_frac", "global_eff"):
                y = Y[:, cols.index(f"{name}@{tag}")]
                order = np.argsort(u, kind="stable")
                us, ys = u[order], y[order]
                same = us[1:] == us[:-1]
                res[name] = {
                    "single_valued_in_usable_fraction":
                        bool(np.all(ys[1:][same] == ys[:-1][same])),
                    "monotone_in_usable_fraction": bool(
                        np.all(np.diff(ys) >= -1e-12)
                        or np.all(np.diff(ys) <= 1e-12)),
                }
            cell[tag] = {"n_samples": int(len(u)),
                         "n_distinct_usable_fraction_values":
                             int(len(np.unique(u))), "outputs": res}
        out["by_scenario"][scen] = cell
    return out


def response_surface_facts() -> dict:
    path = C.RESULTS / "response_surfaces.npz"
    if not path.exists():
        return {}
    d = np.load(path)
    metrics = [str(m) for m in d["metrics"]]
    i_lcc = metrics.index("lcc_frac")
    out = {"r_min_bps": float(d["r_min"]), "grid": int(d["grid"]),
           "note": "other inputs at their range centres", "by_scenario": {}}
    for scen in C.SCENARIOS:
        Z = d[f"{scen}_alpha_eta_det_Z"][:, :, i_lcc]   # [eta index, alpha index]
        eta = d[f"{scen}_alpha_eta_det_y"]
        worst, best = Z[:, -1], Z[:, 0]
        i50 = int(np.argmin(np.abs(eta - 0.5)))
        Zp = d[f"{scen}_p_dc_eta_det_Z"][:, :, i_lcc]
        out["by_scenario"][scen] = {
            "eta_det_grid_step": float(eta[1] - eta[0]),
            "lcc_at_worst_alpha": {
                "eta_det": float(eta[i50]), "lcc": float(worst[i50]),
                "lcc_at_eta_det_max": float(worst[-1]),
                "gain_from_that_point_to_eta_det_max":
                    float(worst[-1] - worst[i50])},
            "lcc_at_best_alpha_eta_det_max": float(best[-1]),
            "max_lcc_for_eta_det_at_most_0.15": float(Z[eta <= 0.15].max()),
            "min_lcc_for_eta_det_at_least_0.5": float(Z[eta >= 0.5].min()),
            "max_change_in_lcc_across_p_dc_at_fixed_eta_det":
                float((Zp.max(axis=1) - Zp.min(axis=1)).max()),
        }
    return out


def input_spans_db() -> dict:
    """Range of each input expressed as a change in the link budget, in dB.

    alpha acts over the median candidate-link length; eta_det and L0 act
    directly. The ordering of the Sobol indices follows the ordering of these
    spans, which is the simplest account of the ranking.
    """
    b = bounds()
    out = {"eta_det_db": 10.0 * np.log10(b["eta_det"][1] / b["eta_det"][0]),
           "L0_db": b["L0"][1] - b["L0"][0], "alpha_db_by_scenario": {}}
    for scen in C.SCENARIOS:
        med = float(np.median(load_topology(scen).edge_km))
        out["alpha_db_by_scenario"][scen] = {
            "median_link_km": med,
            "at_median_link": (b["alpha"][1] - b["alpha"][0]) * med,
            "at_28_km": (b["alpha"][1] - b["alpha"][0]) * 28.0,
            "at_45_km": (b["alpha"][1] - b["alpha"][0]) * C.MAX_EDGE_KM}
    return out


def main() -> int:
    rows = read_rows(C.RESULTS / "sobol_indices.csv")
    conv = read_rows(C.RESULTS / "convergence.csv")
    P = design()
    out = {"generated_by": "analysis/claim_support.py",
           "seed": C.SEED, "n_base": C.N_BASE}
    out["dark_counts"] = dark_counts(P)
    print("  (i) dark counts done", flush=True)
    ro, flat = rank_orders(rows)
    out["rank_orders"] = ro
    out["rank_facts"] = rank_facts(ro)
    print("  (ii) rank orders done", flush=True)
    out["convergence"] = convergence(conv, rows)
    print("  (iii) convergence done", flush=True)
    out["rate_bounds"] = rate_bounds(P)
    print("  (iv) rate bounds done", flush=True)
    out["zero_length_links"] = zero_length_links(P, rows)
    print("  (v) zero-length links done", flush=True)
    out["single_scalar_reduction"] = single_scalar_reduction()
    out["response_surface_facts"] = response_surface_facts()
    out["input_spans_db"] = input_spans_db()

    (C.RESULTS / "claim_support.json").write_text(json.dumps(out, indent=2))
    with open(C.RESULTS / "rank_orders.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    print(f"Written to {C.RESULTS / 'claim_support.json'} and rank_orders.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
