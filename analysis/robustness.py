"""Robustness analyses requested by the round-1 review panel.

Run:  .venv/bin/python analysis/robustness.py [--n-base 1024]

Writes results/robustness_indices.csv and results/robustness_summary.json.

Five configurations, each a variation on the primary design with everything
else held fixed:

  surrogate    Roadmap R-1. The decoy-state link model is replaced by a smooth
               multiplicative surrogate over the same seven inputs. The network
               layer, topologies, thresholds and estimators are unchanged. If
               the interaction-share ordering survives this swap, that ordering
               is a property of the output metrics rather than of QKD physics,
               and hypothesis H2 does not hold.
  ingaas       Roadmap R-2. Ranges narrowed to an InGaAs/InP detector
               generation.
  snspd        Roadmap R-2. Ranges narrowed to a superconducting-nanowire
               generation. Together with `ingaas` this is the correlated
               device-profile analysis: sampling within one technology
               generation removes the pairings that independent priors create
               but that no real device exhibits.
  route        Roadmap R-2. Link length inflated by a route factor, so distance
               is an installed-fibre estimate rather than great-circle.
  sources      Roadmap R-8. Estimator source-set size varied, to show the
               reported indices do not depend on it.

Robustness checks are run at a smaller base sample than the headline analysis.
The convergence study shows rankings are stable from N = 1024, which is the
property these checks interrogate; they are not used for headline index values.
"""

from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool, cpu_count

import numpy as np
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

import config as C
from link_model import binary_entropy, channel_transmittance, secret_key_rate
from network_metrics import evaluate
from topology import load_topology

ROUTE_FACTOR = 1.3  # installed fibre route length vs great-circle distance

# Technology generations. Parameters not listed keep their primary range.
GENERATIONS = {
    "ingaas": {"eta_det": (0.10, 0.30), "p_dc": (-6.0, -5.0),
               "e_d": (0.020, 0.040)},
    "snspd":  {"eta_det": (0.60, 0.85), "p_dc": (-8.0, -7.0),
               "e_d": (0.005, 0.020)},
}

_STATE: dict = {}


# ---------------------------------------------------------------------------
# Surrogate link model
# ---------------------------------------------------------------------------
def surrogate_key_rate(d_km, alpha, eta_det, p_dc, e_d, mu, f_EC, L0):
    """A smooth, strictly multiplicative stand-in for the decoy-state model.

    Every factor is monotone in its input and the whole expression is a product,
    so in logarithms it is exactly additive: the model contributes no
    interaction of its own. It has no binary entropy, no minimum, no threshold
    and no regime where a term switches off. Loss enters through the same
    channel expression so distance still matters in the same way.

    The constant is calibrated in `_calibrate_surrogate` so the surrogate spans
    a rate range comparable to the real model, which keeps the downstream
    threshold meaningful.
    """
    eta = eta_det * channel_transmittance(d_km, alpha, L0)
    return (_STATE.get("surrogate_scale", 1.0)
            * eta
            * mu
            * (1.0 - 2.0 * e_d)
            * (1.0 / f_EC)
            * (1.0 - p_dc))


def _calibrate_surrogate(topo) -> float:
    """Match the surrogate's median rate to the real model's at range centre."""
    mid = {}
    for name, (lo, hi), is_log, _, _ in C.PARAMETERS:
        m = 0.5 * (lo + hi)
        mid[name] = 10.0 ** m if is_log else m
    real = float(np.median(secret_key_rate(topo.edge_km, **mid)))
    _STATE["surrogate_scale"] = 1.0
    raw = float(np.median(surrogate_key_rate(topo.edge_km, **mid)))
    return real / raw if raw > 0 else 1.0


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------
def _init(scenario, kind, scale, n_sources):
    global _STATE
    topo = load_topology(scenario)
    if kind == "route":
        topo.edge_km = topo.edge_km * ROUTE_FACTOR
    if n_sources is not None:
        rng = np.random.default_rng(C.SEED + C.SCENARIOS[scenario]["seed_offset"])
        k = min(n_sources, topo.n_nodes)
        topo.eff_sources = np.sort(rng.choice(topo.n_nodes, size=k, replace=False))
        topo.bott_sources = np.sort(rng.choice(topo.n_nodes, size=k, replace=False))
    _STATE = {"topo": topo, "kind": kind, "surrogate_scale": scale}


def _unpack(row, problem_names, log_names):
    d = dict(zip(problem_names, row))
    for k in log_names:
        d[k] = 10.0 ** d[k]
    return d


def _eval_chunk(args):
    rows, names, logs = args
    topo, kind = _STATE["topo"], _STATE["kind"]
    model = surrogate_key_rate if kind == "surrogate" else secret_key_rate
    out = np.empty((len(rows), 1 + len(C.R_MIN_SWEEP) * len(C.OUTPUTS_THRESHOLDED)))
    for i, row in enumerate(rows):
        skr = model(topo.edge_km, **_unpack(row, names, logs))
        col = 0
        out[i, col] = np.log10(1.0 + float(np.median(skr))); col += 1
        for r_min in C.R_MIN_SWEEP:
            res = evaluate(topo, skr, r_min)
            for n, _, _ in C.OUTPUTS_THRESHOLDED:
                out[i, col] = res[n]; col += 1
    return out


def columns():
    cols = ["median_skr_log10"]
    for r in C.R_MIN_SWEEP:
        tag = f"{r:.0e}".replace("+0", "").replace("+", "")
        cols += [f"{n}@{tag}" for n, _, _ in C.OUTPUTS_THRESHOLDED]
    return cols


def problem_for(kind):
    """Parameter bounds, narrowed when a technology generation is selected."""
    names = C.PARAM_NAMES
    bounds = []
    for name, b, _, _, _ in C.PARAMETERS:
        bounds.append(list(GENERATIONS.get(kind, {}).get(name, b)))
    return {"num_vars": len(names), "names": names, "bounds": bounds}


def run_config(kind, scenario, n_base, procs, n_sources=None):
    topo = load_topology(scenario)
    scale = _calibrate_surrogate(topo) if kind == "surrogate" else 1.0
    problem = problem_for(kind)
    X = sobol_sample.sample(problem, n_base, calc_second_order=False, seed=C.SEED)
    chunks = [(c, problem["names"], C.LOG_PARAMS)
              for c in np.array_split(X, max(procs * 3, 1))]
    with Pool(procs, initializer=_init,
              initargs=(scenario, kind, scale, n_sources)) as pool:
        Y = np.vstack(pool.map(_eval_chunk, chunks))

    cols = columns()
    rows, shares, skipped = [], {}, []
    stride = problem["num_vars"] + 2
    for j, col in enumerate(cols):
        y = Y[:, j]
        # Narrowing the ranges can drive an output constant, or constant within
        # one of the A / B / AB sub-designs. SALib raises rather than returning
        # NaN in that case, so the condition is checked here and the output is
        # recorded as degenerate instead of aborting the whole suite.
        A, B = y[0::stride], y[stride - 1::stride]
        sub_const = (np.ptp(A) <= 1e-12 or np.ptp(B) <= 1e-12)
        if not np.isfinite(y).all() or np.ptp(y) <= 1e-12 or sub_const:
            skipped.append(col)
            continue
        try:
            si = sobol_analyze.analyze(problem, y, calc_second_order=False,
                                       num_resamples=200,
                                       conf_level=C.CONF_LEVEL,
                                       seed=C.SEED, print_to_console=False)
        except Exception as exc:
            skipped.append(f"{col} ({type(exc).__name__})")
            continue
        shares[col] = float(max(0.0, 1.0 - np.nansum(si["S1"])))
        for k, p in enumerate(problem["names"]):
            rows.append(dict(config=kind, scenario=scenario, n_base=n_base,
                             n_sources=n_sources or C.N_SOURCES_EFFICIENCY,
                             output=col, parameter=p,
                             S1=float(si["S1"][k]), ST=float(si["ST"][k]),
                             ST_conf=float(si["ST_conf"][k])))
    return rows, shares, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=1024)
    ap.add_argument("--processes", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()
    C.RESULTS.mkdir(exist_ok=True)

    plan = [("surrogate", "cyl", None), ("surrogate", "spain", None),
            ("ingaas", "cyl", None), ("ingaas", "spain", None),
            ("snspd", "cyl", None), ("snspd", "spain", None),
            ("route", "cyl", None), ("route", "spain", None),
            ("sources", "spain", 48), ("sources", "spain", 192)]

    all_rows, all_shares, all_skipped = [], {}, {}
    for kind, scen, nsrc in plan:
        label = f"{kind}/{scen}" + (f"/{nsrc}src" if nsrc else "")
        eff_kind = "primary" if kind == "sources" else kind
        rows, shares, skipped = run_config(eff_kind, scen, args.n_base,
                                           args.processes, nsrc)
        for r in rows:
            r["config"] = kind
        all_rows += rows
        all_shares[label] = shares
        all_skipped[label] = skipped
        tag = f"{C.R_MIN_PRIMARY:.0e}".replace("+0", "").replace("+", "")
        lk, bn = shares.get("median_skr_log10"), shares.get(f"bottleneck_log10@{tag}")
        msg = f"  {label:<24} link={lk:.3f} " if lk is not None else f"  {label:<24} link=n/a  "
        msg += f"bottleneck={bn:.3f}" if bn is not None else "bottleneck=n/a"
        if skipped:
            msg += f"   [{len(skipped)} degenerate output(s)]"
        print(msg, flush=True)
        # write after every configuration so a late failure cannot discard work
        if all_rows:
            with open(C.RESULTS / "robustness_indices.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
                w.writeheader(); w.writerows(all_rows)

    (C.RESULTS / "robustness_summary.json").write_text(json.dumps(
        {"n_base": args.n_base, "route_factor": ROUTE_FACTOR,
         "degenerate_outputs": all_skipped,
         "generations": {k: {p: list(v) for p, v in g.items()}
                         for k, g in GENERATIONS.items()},
         "interaction_shares": all_shares}, indent=2))
    print(f"\nWritten to {C.RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
