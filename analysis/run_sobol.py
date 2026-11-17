"""Driver: Sobol sampling, model evaluation, and variance decomposition.

Run:  .venv/bin/python analysis/run_sobol.py [--n-base 4096] [--scenarios cyl spain]

Writes to results/:
  raw_outputs_<scenario>.npz   every model output for every sample
  sobol_indices.csv            S1 / ST with bootstrap confidence intervals
  convergence.csv              indices recomputed at increasing base sample size
  topology_summary.json        node/edge statistics per scenario
  run_metadata.json            seeds, versions, timings, configuration echo

Nothing here writes into the manuscript. Per analysis/EXPERIMENT_DESIGN.md, a
RESULT marker is replaced only from the tables produced by this script.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from multiprocessing import Pool, cpu_count

import numpy as np
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

import config as C
from link_model import secret_key_rate
from network_metrics import evaluate
from topology import load_topology

_TOPO = None  # per-worker topology, set by _init_worker


def unpack(row: np.ndarray) -> dict:
    """Map a raw sampler row to model keyword arguments."""
    d = dict(zip(C.PARAM_NAMES, row))
    for k in C.LOG_PARAMS:
        d[k] = 10.0 ** d[k]
    return d


def _init_worker(scenario: str):
    global _TOPO
    _TOPO = load_topology(scenario)


def _eval_chunk(rows: np.ndarray) -> np.ndarray:
    """Evaluate a block of samples. Returns shape (len(rows), n_outputs)."""
    topo = _TOPO
    n_thr = len(C.R_MIN_SWEEP)
    n_out = 1 + n_thr * len(C.OUTPUTS_THRESHOLDED)
    out = np.empty((len(rows), n_out), dtype=float)

    for i, row in enumerate(rows):
        skr = secret_key_rate(topo.edge_km, **unpack(row))
        col = 0
        # Threshold-free: median link SKR, compressed as log10(1+R).
        out[i, col] = np.log10(1.0 + float(np.median(skr)))
        col += 1
        for r_min in C.R_MIN_SWEEP:
            res = evaluate(topo, skr, r_min)
            for name, _, _ in C.OUTPUTS_THRESHOLDED:
                out[i, col] = res[name]
                col += 1
    return out


def output_columns() -> list[str]:
    cols = [C.OUTPUTS_THRESHOLD_FREE[0][0]]
    for r_min in C.R_MIN_SWEEP:
        tag = f"{r_min:.0e}".replace("+0", "").replace("+", "")
        for name, _, _ in C.OUTPUTS_THRESHOLDED:
            cols.append(f"{name}@{tag}")
    return cols


def run_scenario(scenario: str, X: np.ndarray, n_proc: int) -> tuple[np.ndarray, float]:
    chunks = np.array_split(X, max(n_proc * 4, 1))
    t0 = time.perf_counter()
    with Pool(n_proc, initializer=_init_worker, initargs=(scenario,)) as pool:
        parts = pool.map(_eval_chunk, chunks)
    return np.vstack(parts), time.perf_counter() - t0


def analyse(Y: np.ndarray, cols: list[str], scenario: str, n_base: int) -> list[dict]:
    """First- and total-order indices with bootstrap confidence intervals."""
    rows = []
    for j, col in enumerate(cols):
        y = Y[:, j]
        if not np.isfinite(y).all():
            continue
        spread = float(y.max() - y.min())
        if spread <= 1e-12 or float(y.std()) <= 1e-12:
            # Constant output: the variance decomposition is undefined (0/0).
            # Record it explicitly rather than emitting meaningless indices.
            for p in C.PARAM_NAMES:
                rows.append(dict(scenario=scenario, output=col, parameter=p,
                                 S1=np.nan, S1_conf=np.nan, ST=np.nan,
                                 ST_conf=np.nan, degenerate=True,
                                 output_std=float(y.std()), n_base=n_base))
            continue
        si = sobol_analyze.analyze(
            C.SALIB_PROBLEM, y, calc_second_order=C.CALC_SECOND_ORDER,
            num_resamples=C.N_BOOTSTRAP, conf_level=C.CONF_LEVEL,
            seed=C.SEED, print_to_console=False,
        )
        for k, p in enumerate(C.PARAM_NAMES):
            rows.append(dict(
                scenario=scenario, output=col, parameter=p,
                S1=float(si["S1"][k]), S1_conf=float(si["S1_conf"][k]),
                ST=float(si["ST"][k]), ST_conf=float(si["ST_conf"][k]),
                degenerate=False, output_std=float(y.std()), n_base=n_base,
            ))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=C.N_BASE)
    ap.add_argument("--scenarios", nargs="+", default=list(C.SCENARIOS))
    ap.add_argument("--processes", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--skip-convergence", action="store_true")
    args = ap.parse_args()

    C.RESULTS.mkdir(exist_ok=True)
    D = C.SALIB_PROBLEM["num_vars"]
    stride = D + 2

    X = sobol_sample.sample(C.SALIB_PROBLEM, args.n_base,
                            calc_second_order=C.CALC_SECOND_ORDER, seed=C.SEED)
    assert len(X) == args.n_base * stride, (len(X), args.n_base * stride)

    cols = output_columns()
    print(f"Sobol design: N={args.n_base}, D={D}, "
          f"{len(X):,} evaluations per scenario, {len(cols)} outputs")
    print(f"Processes: {args.processes}\n")

    topo_summary, index_rows, conv_rows, timings = {}, [], [], {}

    for scenario in args.scenarios:
        topo = load_topology(scenario)
        topo_summary[scenario] = topo.summary()
        print(f"[{topo.label}] {topo.n_nodes} nodes, {topo.n_edges:,} candidate edges")

        Y, dt = run_scenario(scenario, X, args.processes)
        timings[scenario] = dt
        print(f"  evaluated in {dt/60:.1f} min ({1000*dt/len(X):.1f} ms/sample)")

        np.savez_compressed(C.RESULTS / f"raw_outputs_{scenario}.npz",
                            X=X, Y=Y, columns=np.array(cols))

        index_rows += analyse(Y, cols, scenario, args.n_base)

        if not args.skip_convergence:
            for n_sub in C.CONVERGENCE_SIZES:
                if n_sub > args.n_base:
                    continue
                conv_rows += analyse(Y[: n_sub * stride], cols, scenario, n_sub)
            print(f"  convergence computed at N={C.CONVERGENCE_SIZES}")

        n_deg = sum(1 for r in index_rows
                    if r["scenario"] == scenario and r["degenerate"])
        if n_deg:
            print(f"  WARNING: {n_deg // D} output(s) constant -> indices undefined")
        print()

    _write_csv(C.RESULTS / "sobol_indices.csv", index_rows)
    if conv_rows:
        _write_csv(C.RESULTS / "convergence.csv", conv_rows)
    (C.RESULTS / "topology_summary.json").write_text(
        json.dumps(topo_summary, indent=2))

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": C.SEED,
        "n_base": args.n_base,
        "n_evaluations_per_scenario": len(X),
        "calc_second_order": C.CALC_SECOND_ORDER,
        "n_bootstrap": C.N_BOOTSTRAP,
        "conf_level": C.CONF_LEVEL,
        "parameters": {n: {"bounds": list(b), "log10": lg, "unit": u}
                       for n, b, lg, u, _ in C.PARAMETERS},
        "f_rep_hz": C.F_REP,
        "q_sift": C.Q_SIFT,
        "max_edge_km": C.MAX_EDGE_KM,
        "r_min_primary": C.R_MIN_PRIMARY,
        "r_min_sweep": list(C.R_MIN_SWEEP),
        "n_sources_efficiency": C.N_SOURCES_EFFICIENCY,
        "n_sources_bottleneck": C.N_SOURCES_BOTTLENECK,
        "eval_seconds": timings,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("numpy", "scipy", "SALib")},
    }
    (C.RESULTS / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Written to {C.RESULTS}")
    return 0


def _write_csv(path, rows):
    import csv as _csv
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
