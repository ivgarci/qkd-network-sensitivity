"""Interaction shares for every output, threshold, scenario and link model.

Run:  .venv/bin/python analysis/interaction_shares.py [--processes 5]
Reads  results/raw_outputs_<scenario>.npz  (run_sobol.py; git-ignored)
Writes results/interaction_shares.csv
       results/interaction_shares_summary.json
       results/raw_outputs_surrogate_<scenario>.npz  (git-ignored cache)

Roadmap R3-2. The interaction share is 1 - sum_i S_i, clipped at zero, exactly
as summarize_results.py and robustness.py compute it; the unclipped value is
stored next to it and the interval is of the unclipped value.

Equal footing. The manuscript compared the decoy-state model at N = 4096
(1000 resamples) with the surrogate at N = 1024 (200 resamples, no interval
stored). Here three rows are produced per output:

  full       N = 4096   the principal run; reproduces summary.json
  full       N = 1024   the first 1024 base rows of the same design
  surrogate  N = 1024   the surrogate re-evaluated on the same 1024 base rows

all with the same number of bootstrap resamples. The two N = 1024 rows use the
same base points and the same resampling matrix, so the difference "full minus
surrogate" is a paired comparison and carries its own interval. The surrogate
is not run at N = 4096; that asymmetry is stated in the output file.

The surrogate outputs are regenerated here because robustness.py did not keep
them. The point estimates are checked against robustness_summary.json.

For bottleneck capacity the alternative codings of
bottleneck_decomposition.py are listed as extra rows (column `coding`).
"""

from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool, cpu_count

import numpy as np
from SALib.sample import sobol as sobol_sample

import config as C
import robustness as RB
import sobol_tools as ST
from bottleneck_decomposition import TAGS, load_raw, recode
from topology import load_topology

N_BOOT = C.N_BOOTSTRAP
N_CONTROL = 1024

# Two output orders. The first is the one the manuscript and Figure 9 used,
# which was chosen after seeing the shares at the primary threshold. The second
# is the order of manuscript Table 3 / config.OUTPUTS_THRESHOLDED, fixed before
# any result existed.
ORDER_MANUSCRIPT = ("reach_km", "usable_frac", "global_eff", "lcc_frac",
                    "reachable_pairs", "isolated_frac", "bottleneck_log10")
ORDER_TABLE3 = tuple(n for n, _, _ in C.OUTPUTS_THRESHOLDED)


def surrogate_raw(scenario: str, procs: int):
    """Surrogate outputs on the first N_CONTROL base rows (cached)."""
    path = C.RESULTS / f"raw_outputs_surrogate_{scenario}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        return d["Y"], [str(c) for c in d["columns"]]
    topo = load_topology(scenario)
    scale = RB._calibrate_surrogate(topo)
    problem = RB.problem_for("surrogate")
    X = sobol_sample.sample(problem, N_CONTROL, calc_second_order=False,
                            seed=C.SEED)
    chunks = [(c, problem["names"], C.LOG_PARAMS)
              for c in np.array_split(X, max(procs * 3, 1))]
    with Pool(procs, initializer=RB._init,
              initargs=(scenario, "surrogate", scale, None)) as pool:
        Y = np.vstack(pool.map(RB._eval_chunk, chunks))
    cols = RB.columns()
    np.savez_compressed(path, X=X, Y=Y, columns=np.array(cols))
    return Y, cols


def analyse(Y, cols, n_base, W):
    """{(output, coding): indices dict or None} for one model and base size."""
    res = {}
    Y = Y[: n_base * ST.STRIDE]
    for j, col in enumerate(cols):
        res[(col, "stored")] = ST.indices(Y[:, j], W)
    for r_min, tag in TAGS.items():
        col = f"bottleneck_log10@{tag}"
        for coding, y in recode(Y[:, cols.index(col)], r_min).items():
            if coding != "stored":
                res[(col, coding)] = ST.indices(y, W)
    return res


def paired(a, b):
    """Difference of two shares over shared resamples: value and interval."""
    if a is None or b is None:
        return None
    lo, hi, _ = ST.interval(a["share_boot"] - b["share_boot"])
    return {"difference": a["share_raw"] - b["share_raw"], "ci": [lo, hi],
            "resolved": bool(lo > 0 or hi < 0)}


def ordering(res, tag, order):
    """Is the share non-decreasing along `order` (link rate first)?"""
    keys = [("median_skr_log10", "stored")] + [(f"{n}@{tag}", "stored")
                                               for n in order]
    if any(res.get(k) is None for k in keys):
        missing = [k[0] for k in keys if res.get(k) is None]
        return {"outputs": [k[0] for k in keys], "evaluable": False,
                "degenerate_outputs": missing}
    shares = [res[k]["share_raw"] for k in keys]
    steps = []
    for (ka, kb) in zip(keys[:-1], keys[1:]):
        p = paired(res[kb], res[ka])
        steps.append({"from": ka[0], "to": kb[0], **p})
    falls = [s for s in steps if s["difference"] < 0]
    return {
        "outputs": [k[0] for k in keys], "evaluable": True,
        "shares": shares,
        "monotone_nondecreasing_in_point_estimates": not falls,
        "n_steps": len(steps),
        "n_steps_that_fall": len(falls),
        "n_falls_resolved": sum(1 for s in falls if s["resolved"]),
        "n_rises_resolved": sum(1 for s in steps
                                if s["difference"] > 0 and s["resolved"]),
        "steps": steps,
        "outputs_sorted_by_share": [k[0] for k, _ in sorted(
            zip(keys, shares), key=lambda t: t[1])],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()

    stored = json.loads((C.RESULTS / "summary.json").read_text())["rankings"]
    rb = json.loads((C.RESULTS / "robustness_summary.json").read_text())
    rows, summary = [], {
        "definition": "interaction share = max(0, 1 - sum of first-order "
                      "indices); *_raw and all intervals are unclipped",
        "bootstrap": {"resamples": N_BOOT, "conf_level": C.CONF_LEVEL,
                      "scheme": "base rows resampled with replacement "
                                "(as SALib); percentile interval; seed from "
                                "config.SEED; the same resampling matrix for "
                                "every output and model at a given N",
                      "caveat": "rows of a scrambled Sobol design are treated "
                                "as i.i.d."},
        "footing": "full model at N = 4096 and at its first 1024 base rows; "
                   "surrogate at the same 1024 base rows; equal resamples. "
                   "The surrogate was NOT run at N = 4096: compare models "
                   "only between the two N = 1024 rows.",
        "orders": {"manuscript_fig9": ["median_skr_log10", *ORDER_MANUSCRIPT],
                   "table3": ["median_skr_log10", *ORDER_TABLE3]},
        "checks": {}, "ordering": {}, "full_minus_surrogate_N1024": {},
        "network_minus_link_full_N4096": {},
    }

    for scen in C.SCENARIOS:
        Yf, cols = load_raw(scen, "full")
        if Yf is None:
            print(f"  {scen}: raw_outputs_{scen}.npz missing - run run_sobol.py")
            continue
        Ys, cols_s = surrogate_raw(scen, args.processes)
        assert cols_s == cols
        W4, W1 = (ST.bootstrap_weights(C.N_BASE, N_BOOT),
                  ST.bootstrap_weights(N_CONTROL, N_BOOT))
        models = {
            ("full", C.N_BASE): analyse(Yf, cols, C.N_BASE, W4),
            ("full", N_CONTROL): analyse(Yf, cols, N_CONTROL, W1),
            ("surrogate", N_CONTROL): analyse(Ys, cols, N_CONTROL, W1),
        }

        # consistency with what is already stored
        d_full = max(abs(models[("full", C.N_BASE)][(o, "stored")]["share"]
                         - v["interaction_share"])
                     for o, v in stored[scen].items())
        d_sur = max(abs(models[("surrogate", N_CONTROL)][(o, "stored")]["share"]
                        - v)
                    for o, v in rb["interaction_shares"][f"surrogate/{scen}"].items())
        summary["checks"][scen] = {
            "max_abs_diff_vs_summary_json_full_N4096": d_full,
            "max_abs_diff_vs_robustness_summary_surrogate": d_sur}
        print(f"  {scen}: recomputed shares differ from stored by at most "
              f"{d_full:.1e} (full) and {d_sur:.1e} (surrogate)", flush=True)

        for (model, n_base), res in models.items():
            for (output, coding), r in res.items():
                metric, _, thr = output.partition("@")
                row = dict(scenario=scen, model=model, n_base=n_base,
                           n_bootstrap=N_BOOT, output=output, metric=metric,
                           threshold=thr or "none", coding=coding)
                if r is None:
                    row.update(degenerate=True)
                else:
                    lo, hi, half = ST.interval(r["share_boot"])
                    row.update(degenerate=False, sum_S1=r["sum_S1"],
                               share=r["share"], share_raw=r["share_raw"],
                               share_ci_lo=lo, share_ci_hi=hi,
                               share_half_width=half,
                               share_distinct_from_zero=bool(lo > 0),
                               n_boot_undefined=r["n_boot_undefined"])
                rows.append(row)

            key = f"{model}_N{n_base}"
            for tag in TAGS.values():
                summary["ordering"].setdefault(scen, {}).setdefault(tag, {})[key] = {
                    "manuscript_fig9": ordering(res, tag, ORDER_MANUSCRIPT),
                    "table3": ordering(res, tag, ORDER_TABLE3)}

        f1, s1 = models[("full", N_CONTROL)], models[("surrogate", N_CONTROL)]
        summary["full_minus_surrogate_N1024"][scen] = {
            f"{o}|{c}" if c != "stored" else o: paired(f1[(o, c)], s1[(o, c)])
            for (o, c) in f1}
        f4 = models[("full", C.N_BASE)]
        link = f4[("median_skr_log10", "stored")]
        summary["network_minus_link_full_N4096"][scen] = {
            f"{o}|{c}" if c != "stored" else o: paired(f4[(o, c)], link)
            for (o, c) in f4 if o != "median_skr_log10"}

    fields = ["scenario", "model", "n_base", "n_bootstrap", "output", "metric",
              "threshold", "coding", "degenerate", "sum_S1", "share",
              "share_raw", "share_ci_lo", "share_ci_hi", "share_half_width",
              "share_distinct_from_zero", "n_boot_undefined"]
    with open(C.RESULTS / "interaction_shares.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    (C.RESULTS / "interaction_shares_summary.json").write_text(
        json.dumps(summary, indent=2))

    for scen, by_tag in summary["ordering"].items():
        for tag, by_model in by_tag.items():
            for key, o in by_model.items():
                for name in ("manuscript_fig9", "table3"):
                    v = o[name]
                    if not v["evaluable"]:
                        continue
                    print(f"  {scen:<6} @{tag:<4} {key:<15} {name:<16} "
                          f"monotone={v['monotone_nondecreasing_in_point_estimates']} "
                          f"falls={v['n_steps_that_fall']}/{v['n_steps']} "
                          f"(resolved {v['n_falls_resolved']})  "
                          + " ".join(f"{s:.3f}" for s in v["shares"]))
    print(f"Written to {C.RESULTS / 'interaction_shares.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
