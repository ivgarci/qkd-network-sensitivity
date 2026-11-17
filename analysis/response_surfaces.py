"""Conditional response surfaces for figures 6 and 7.

Sweeps two parameters on a grid with the remaining five held at the centre of
their sampled range, and records every network output. This is a conditional
slice, not a marginal effect: it shows the shape of the response surface, while
the Sobol indices carry the variance attribution.

Run:  .venv/bin/python analysis/response_surfaces.py [--grid 48]
Writes results/response_surfaces.npz
"""

from __future__ import annotations

import argparse
from itertools import product
from multiprocessing import Pool, cpu_count

import numpy as np

import config as C
from link_model import secret_key_rate
from network_metrics import evaluate
from topology import load_topology

# Parameter pairs to sweep. The first is the hardware-vs-loss trade-off of
# manuscript figure 6; the second is the noise-vs-efficiency trade-off.
PAIRS = [("alpha", "eta_det"), ("p_dc", "eta_det")]

METRICS = ["lcc_frac", "usable_frac", "reachable_pairs", "global_eff",
           "isolated_frac", "reach_km", "bottleneck_log10"]

_TOPO = None


def nominal() -> dict:
    """Centre of each sampled range, in model units."""
    out = {}
    for name, (lo, hi), is_log, _, _ in C.PARAMETERS:
        mid = 0.5 * (lo + hi)
        out[name] = 10.0 ** mid if is_log else mid
    return out


def axis_values(name: str, n: int) -> np.ndarray:
    """Grid values for one parameter, in model units."""
    spec = next(p for p in C.PARAMETERS if p[0] == name)
    lo, hi, is_log = spec[1][0], spec[1][1], spec[2]
    grid = np.linspace(lo, hi, n)
    return 10.0 ** grid if is_log else grid


def _init(scenario):
    global _TOPO
    _TOPO = load_topology(scenario)


def _cell(task):
    xname, yname, xval, yval, r_min = task
    p = nominal()
    p[xname], p[yname] = xval, yval
    skr = secret_key_rate(_TOPO.edge_km, **p)
    res = evaluate(_TOPO, skr, r_min)
    return [res[m] for m in METRICS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--processes", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()

    C.RESULTS.mkdir(exist_ok=True)
    store = {"metrics": np.array(METRICS), "r_min": C.R_MIN_PRIMARY,
             "grid": args.grid}

    for scenario in C.SCENARIOS:
        for xname, yname in PAIRS:
            xs = axis_values(xname, args.grid)
            ys = axis_values(yname, args.grid)
            tasks = [(xname, yname, xv, yv, C.R_MIN_PRIMARY)
                     for yv in ys for xv in xs]
            with Pool(args.processes, initializer=_init,
                      initargs=(scenario,)) as pool:
                flat = pool.map(_cell, tasks, chunksize=8)
            arr = np.array(flat).reshape(len(ys), len(xs), len(METRICS))
            tag = f"{scenario}_{xname}_{yname}"
            store[f"{tag}_Z"] = arr
            store[f"{tag}_x"] = xs
            store[f"{tag}_y"] = ys
            print(f"  {tag}: {arr.shape}  "
                  f"S_LCC {arr[:, :, 0].min():.3f}..{arr[:, :, 0].max():.3f}")

    np.savez_compressed(C.RESULTS / "response_surfaces.npz", **store)
    print(f"\nWritten to {C.RESULTS / 'response_surfaces.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
