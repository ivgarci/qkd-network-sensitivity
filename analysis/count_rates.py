"""Detector count rates implied by the link model (round 3, R3-11).

Run:  .venv/bin/python analysis/count_rates.py
Writes results/count_rates.json

Section 5.8 of the manuscript bounds the detector count rate a link needs in
order to meet a key-rate threshold. This script is the artifact for those
numbers.

Definition. The detection rate of a link is D = f_rep * Q_mu: the repetition
rate times the overall gain of eq. (3) of the link model, i.e. the probability
per pulse that at least one of the receiver's detectors clicks, dark counts
included. It is therefore a rate summed over the receiver's detectors. The
asymptotic model sends every pulse at the signal intensity mu (decoy pulses are
not counted separately), as `link_model.secret_key_rate` does. Q_mu is computed
here with the same expression as in `link_model.py`, and the script checks
that the key rate it reads from `link_model.py` never exceeds D.

What is computed.
1. The largest ratio R / D of key rate to detection rate over the sampled box
   (the seven ranges of config.PARAMETERS) and link lengths 0 to 45 km
   (config.MAX_EDGE_KM; zero-length candidate links exist in mainland Spain).
   f_rep cancels in the ratio. Three searches, the bound being their maximum:
   (a) a grid: alpha, L0, p_dc, e_d and f_EC at both ends of their ranges,
       eta_det and mu on fine grids, d on a 0.25 km grid;
   (b) the principal Saltelli design (same sampler, seed and N as run_sobol.py);
   (c) 200,000 seeded uniform draws in the box.
   Because R >= R_min requires D >= R_min / max(R/D), the minimum detection
   rate of a usable link at each threshold follows.
2. D, R and R/D at the centre of every range (p_dc at the centre of its log10
   range) on 28 km and 45 km links, for the principal box and for the
   InGaAs/InP and SNSPD ranges of robustness.GENERATIONS.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
from SALib.sample import sobol as sobol_sample

import config as C
from link_model import channel_transmittance, secret_key_rate
from robustness import GENERATIONS

OUT = C.RESULTS / "count_rates.json"
D_GRID = np.arange(0.0, C.MAX_EDGE_KM + 1e-9, 0.25)
CENTRE_LENGTHS_KM = (28.0, 45.0)
N_RANDOM = 200_000
BOUNDS = {p[0]: p[1] for p in C.PARAMETERS}


def gain(d_km, alpha, eta_det, p_dc, mu, L0):
    """Overall gain Q_mu, as in link_model.secret_key_rate."""
    eta = eta_det * channel_transmittance(d_km, alpha, L0)
    Y0 = 1.0 - (1.0 - p_dc) ** 2
    one_minus_exp = -np.expm1(-eta * mu)
    return Y0 + one_minus_exp - Y0 * one_minus_exp


def ratio_over_d(alpha, eta_det, p_dc, e_d, mu, f_EC, L0):
    """(max over D_GRID of R/D, length at which it occurs)."""
    r = secret_key_rate(D_GRID, alpha, eta_det, p_dc, e_d, mu, f_EC, L0)
    q = C.F_REP * gain(D_GRID, alpha, eta_det, p_dc, mu, L0)
    rr = r / q
    i = int(np.argmax(rr))
    return float(rr[i]), float(D_GRID[i])


def physical(row):
    """Design row (log10 for log-sampled inputs) -> physical parameter dict."""
    return {n: (10.0 ** v if n in C.LOG_PARAMS else float(v))
            for n, v in zip(C.PARAM_NAMES, row)}


def search_grid():
    ends = {n: BOUNDS[n] for n in ("alpha", "L0", "e_d", "f_EC")}
    ends["p_dc"] = tuple(10.0 ** v for v in BOUNDS["p_dc"])
    etas = np.linspace(*BOUNDS["eta_det"], 76)
    mus = np.linspace(*BOUNDS["mu"], 121)
    best = (-1.0, None)
    n = 0
    for a, L0, p, e, f in itertools.product(ends["alpha"], ends["L0"],
                                           ends["p_dc"], ends["e_d"],
                                           ends["f_EC"]):
        for eta in etas:
            for mu in mus:
                v, d = ratio_over_d(a, eta, p, e, mu, f, L0)
                n += 1
                if v > best[0]:
                    best = (v, dict(alpha=a, eta_det=float(eta), p_dc=p, e_d=e,
                                    mu=float(mu), f_EC=f, L0=L0, d_km=d))
    return best, n


def search_rows(rows):
    best = (-1.0, None)
    for row in rows:
        par = physical(row)
        v, d = ratio_over_d(**par)
        if v > best[0]:
            best = (v, dict(par, d_km=d))
    return best


def centre_point(bounds, d_km):
    par = {n: 0.5 * (lo + hi) for n, (lo, hi) in bounds.items()}
    par["p_dc"] = 10.0 ** par["p_dc"]
    r = float(secret_key_rate(np.array([d_km]), **par)[0])
    D = float(C.F_REP * gain(np.array([d_km]), par["alpha"], par["eta_det"],
                             par["p_dc"], par["mu"], par["L0"])[0])
    return {"parameters": par, "d_km": d_km, "detection_rate_per_s": D,
            "key_rate_bps": r, "key_over_detection": r / D}


def main() -> int:
    grid_best, n_grid = search_grid()

    design = sobol_sample.sample(C.SALIB_PROBLEM, C.N_BASE,
                                 calc_second_order=C.CALC_SECOND_ORDER,
                                 seed=C.SEED)
    design_best = search_rows(design)

    rng = np.random.default_rng(C.SEED)
    lo = np.array([BOUNDS[n][0] for n in C.PARAM_NAMES])
    hi = np.array([BOUNDS[n][1] for n in C.PARAM_NAMES])
    random_best = search_rows(rng.uniform(lo, hi, size=(N_RANDOM, len(lo))))

    searches = {"grid": grid_best, "principal_design": design_best,
                "uniform_random": random_best}
    max_ratio = max(v[0] for v in searches.values())
    assert 0.0 < max_ratio < 1.0, "key rate must stay below the detection rate"

    boxes = {"principal": dict(BOUNDS)}
    for g, spec in GENERATIONS.items():
        boxes[g] = dict(BOUNDS, **spec)
    centres = {name: {f"{d:g}": centre_point(b, d) for d in CENTRE_LENGTHS_KM}
               for name, b in boxes.items()}

    res = {
        "definition": ("detection rate D = f_rep * Q_mu, Q_mu the overall gain "
                       "of the link model (probability per pulse that at least "
                       "one receiver detector clicks, dark counts included; "
                       "summed over the receiver's detectors; every pulse at "
                       "intensity mu). key rate R = link_model.secret_key_rate."),
        "f_rep_hz": C.F_REP,
        "length_range_km": [float(D_GRID[0]), float(D_GRID[-1])],
        "length_step_km": 0.25,
        "max_key_over_detection": max_ratio,
        "searches": {
            k: {"max_key_over_detection": v[0], "at": v[1]}
            for k, v in searches.items()},
        "search_sizes": {
            "grid_parameter_points": n_grid,
            "grid_description": ("alpha, L0, p_dc, e_d, f_EC at both ends of "
                                 "their ranges; eta_det 76 values; mu 121 "
                                 "values"),
            "principal_design_rows": int(len(design)),
            "uniform_random_points": N_RANDOM,
            "random_seed": C.SEED},
        "min_detection_rate_at_threshold_per_s": {
            f"{t:.0e}".replace("+0", "").replace("+", ""): t / max_ratio
            for t in (C.R_MIN_SWEEP)},
        "range_centre": centres,
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    print(f"max R/D over the box: {max_ratio:.4f}")
    for k, v in searches.items():
        print(f"  {k}: {v[0]:.4f} at {v[1]}")
    for t, v in res["min_detection_rate_at_threshold_per_s"].items():
        print(f"  R_min {t}: D >= {v:.3e} /s")
    for name, by_d in centres.items():
        for d, c in by_d.items():
            print(f"  centre {name} {d} km: D {c['detection_rate_per_s']:.3e} /s,"
                  f" R {c['key_rate_bps']:.3e} bit/s, R/D "
                  f"{c['key_over_detection']:.3f}")
    print(f"wrote {OUT.relative_to(C.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
