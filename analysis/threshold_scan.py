"""Threshold scan: how the network outputs spread as R_min is varied.

Run:  .venv/bin/python analysis/threshold_scan.py
Writes results/threshold_scan.csv and results/threshold_scan.json

Section 3.5 of the manuscript says the primary threshold was located by a scan
of the spread of the largest-component fraction over R_min. Until round 3 that
scan existed only as a comment in config.py, with no script and no stored
output. This script is the artifact.

Design: the principal Saltelli design itself (same sampler, seed and base size
as run_sobol.py, regenerated here rather than read from the git-ignored raw
outputs), both scenarios. For each sample the link model is evaluated once and
the structural outputs are read from the prefix table of prefix_metrics.py at
every threshold of the grid.

What the scan is and is not: it records where the outputs have variance. The
primary threshold was chosen where that variance is largest, which is a choice
made after seeing model output; the deviation is recorded in EXPERIMENT_DESIGN.md.
"""

from __future__ import annotations

import csv
import json

import numpy as np
from SALib.sample import sobol as sobol_sample

import config as C
import prefix_metrics as PM
from link_model import secret_key_rate
from topology import load_topology

# The eight values quoted in the config.py comment, and the scan range the
# manuscript states (1e6 .. 3e7), on a finer grid; then an extension downwards
# to show where the outputs become constant.
COMMENT_GRID = (1e6, 2e6, 3e6, 5e6, 7e6, 1e7, 2e7, 3e7)
COMMENT_SD = (.019, .146, .260, .378, .387, .339, .135, .038)
GRID = (1e4, 2e4, 5e4, 1e5, 2e5, 5e5,
        1e6, 1.5e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 1e7, 1.5e7, 2e7, 3e7,
        5e7)


def unpack(row):
    d = dict(zip(C.PARAM_NAMES, row))
    for k in C.LOG_PARAMS:
        d[k] = 10.0 ** d[k]
    return d


def scan(scenario: str, X: np.ndarray) -> list[dict]:
    topo = load_topology(scenario)
    curves = PM.build(topo)
    K = np.empty((len(X), len(GRID)), dtype=np.int64)
    for i, row in enumerate(X):
        skr = secret_key_rate(curves.km, **unpack(row))
        for j, r_min in enumerate(GRID):
            K[i, j] = PM.usable_count(skr, r_min)
    rows = []
    for j, r_min in enumerate(GRID):
        k = K[:, j]
        lcc = curves.lcc_frac[k]
        usable = k / curves.n_edges
        reach = np.where(k > 0, curves.km[np.maximum(k, 1) - 1], 0.0)
        q = np.quantile(lcc, [0.05, 0.25, 0.75, 0.95])
        rows.append(dict(
            scenario=scenario, r_min_bps=r_min, n_samples=len(X),
            in_stated_scan_range=bool(1e6 <= r_min <= 3e7),
            lcc_mean=float(lcc.mean()), lcc_sd=float(lcc.std()),
            lcc_median=float(np.median(lcc)),
            lcc_q05=float(q[0]), lcc_q25=float(q[1]),
            lcc_q75=float(q[2]), lcc_q95=float(q[3]),
            frac_lcc_above_095=float((lcc > 0.95).mean()),
            frac_lcc_below_005=float((lcc < 0.05).mean()),
            frac_no_usable_link=float((k == 0).mean()),
            frac_all_links_usable=float((k == curves.n_edges).mean()),
            usable_frac_mean=float(usable.mean()),
            usable_frac_sd=float(usable.std()),
            reach_km_sd=float(reach.std()),
            reachable_pairs_sd=float(curves.reachable_pairs[k].std()),
            isolated_frac_sd=float(curves.isolated_frac[k].std()),
        ))
    return rows


def main() -> int:
    X = sobol_sample.sample(C.SALIB_PROBLEM, C.N_BASE,
                            calc_second_order=C.CALC_SECOND_ORDER, seed=C.SEED)
    rows = []
    for scen in C.SCENARIOS:
        rows += scan(scen, X)
        print(f"  {scen}: scanned {len(GRID)} thresholds over {len(X):,} samples",
              flush=True)

    with open(C.RESULTS / "threshold_scan.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Summary facts the manuscript may quote.
    out = {"design": {"sampler": "SALib Saltelli/Sobol, same as run_sobol.py",
                      "n_base": C.N_BASE, "n_samples": int(len(X)),
                      "seed": C.SEED, "grid_bps": list(GRID)},
           "config_comment": {"grid_bps": list(COMMENT_GRID),
                              "lcc_sd": list(COMMENT_SD)},
           "scenarios": {}}
    for scen in C.SCENARIOS:
        sub = [r for r in rows if r["scenario"] == scen]
        stated = [r for r in sub if r["in_stated_scan_range"]]
        peak = max(stated, key=lambda r: r["lcc_sd"])
        at = {r["r_min_bps"]: r for r in sub}
        # Thresholds at which S_LCC sd is within 10% of its peak.
        plateau = [r["r_min_bps"] for r in stated
                   if r["lcc_sd"] >= 0.9 * peak["lcc_sd"]]
        const_below = max((r["r_min_bps"] for r in sub
                           if r["frac_all_links_usable"] == 1.0), default=None)
        out["scenarios"][scen] = {
            "lcc_sd_peak_r_min_bps": peak["r_min_bps"],
            "lcc_sd_peak": peak["lcc_sd"],
            "lcc_sd_within_10pct_of_peak_bps": [min(plateau), max(plateau)],
            "lcc_sd_at_comment_grid": [at[r]["lcc_sd"] for r in COMMENT_GRID],
            "max_abs_diff_from_config_comment": float(max(
                abs(at[r]["lcc_sd"] - s)
                for r, s in zip(COMMENT_GRID, COMMENT_SD))),
            "largest_grid_threshold_with_every_link_usable_in_every_sample":
                const_below,
            "sweep": {f"{r:.0e}": {k: at[r][k] for k in (
                "lcc_mean", "lcc_sd", "lcc_median", "frac_lcc_above_095",
                "frac_lcc_below_005", "frac_no_usable_link")}
                for r in C.R_MIN_SWEEP},
        }
        s = out["scenarios"][scen]
        print(f"  {scen}: S_LCC sd peaks at {s['lcc_sd_peak_r_min_bps']:.1e} "
              f"({s['lcc_sd_peak']:.3f}); within 10% of peak over "
              f"{s['lcc_sd_within_10pct_of_peak_bps']}; "
              f"max diff from config comment "
              f"{s['max_abs_diff_from_config_comment']:.3f}")
    (C.RESULTS / "threshold_scan.json").write_text(json.dumps(out, indent=2))
    print(f"Written to {C.RESULTS / 'threshold_scan.csv'} and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
