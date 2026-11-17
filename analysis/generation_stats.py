"""Output distributions within each detector generation.

Run:  .venv/bin/python analysis/generation_stats.py
Writes results/generation_stats.json and results/generation_samples.npz

The manuscript quotes the mean and median largest-component fraction within
each generation, and the fraction of samples falling above or below fixed
levels. Those figures must come from a stored artifact like every other number
in the paper, not from an interactive check, so they are computed here.

Round-3 extension (roadmap R3-4, R3-10, R3-15)
----------------------------------------------
* All three thresholds of config.R_MIN_SWEEP, not only the primary one.
* A gap-filling "intermediate" generation (INTERMEDIATE below). The two
  generations of robustness.GENERATIONS leave receiver efficiency 0.30-0.60
  unsampled, which is where the connectivity transition sits. The intermediate
  case shows what a design placed in that gap does. It is a sensitivity case
  that interpolates between the two generations; it is not a claim about a
  device class on the market.
* Two variants of the whole analysis: link lengths inflated by the 1.3 route
  factor of robustness.py, and misalignment error decoupled from the detector
  generation (e_d at its full primary range in every generation).
* Bootstrap intervals for every summary statistic.

The sample
----------
Each (generation, scenario) cell is the 2304 rows of a Saltelli cross design
with base size 256 and seven inputs: 256 blocks of nine rows (A_i, the seven
rows AB_i^(j) that differ from A_i in one coordinate taken from B_i, and B_i).
Every row is marginally uniform over the generation's box, so means and
fractions are unbiased, but the rows are NOT 2304 independent draws: rows of a
block share coordinates. The intervals therefore come from a cluster bootstrap
that resamples the 256 blocks with replacement (N_BOOT resamples, percentile
interval), seeded from config.SEED.

The structural outputs are read from the prefix table of prefix_metrics.py,
which test_metrics.py checks against network_metrics.evaluate. The values at
the primary threshold reproduce the pre-round-3 file, which called evaluate
directly.
"""

from __future__ import annotations

import json

import numpy as np
from SALib.sample import sobol as sobol_sample

import config as C
import prefix_metrics as PM
from link_model import secret_key_rate
from robustness import GENERATIONS, ROUTE_FACTOR
from topology import load_topology

N_BASE = 256  # distribution shape, not index estimation; 2304 samples per cell
N_BOOT = 2000
STRIDE = len(C.PARAMETERS) + 2

# Gap-filling case. eta_det covers exactly the interval the two generations
# leave out. The two other generation-specific inputs are interpolated:
#   p_dc  InGaAs [1e-6, 1e-5], SNSPD [1e-8, 1e-7]  ->  the decade between them
#   e_d   lower bound = mean of the two lower bounds (0.020, 0.005) = 0.0125
#         upper bound = mean of the two upper bounds (0.040, 0.020) = 0.030
# The remaining four inputs keep their primary ranges, as in both generations.
INTERMEDIATE = {"eta_det": (0.30, 0.60), "p_dc": (-7.0, -6.0),
                "e_d": (0.0125, 0.030)}

ALL_GENERATIONS = {"ingaas": GENERATIONS["ingaas"],
                   "intermediate": INTERMEDIATE,
                   "snspd": GENERATIONS["snspd"]}
ORDER = ("ingaas", "intermediate", "snspd")

VARIANTS = {
    "baseline": dict(route=1.0, free=()),
    "route_x1.3": dict(route=ROUTE_FACTOR, free=()),
    "e_d_decoupled": dict(route=1.0, free=("e_d",)),
}

TAGS = {r: f"{r:.0e}".replace("+0", "").replace("+", "") for r in C.R_MIN_SWEEP}


def problem(gen: str, free=()) -> dict:
    """Bounds for one generation; inputs in `free` revert to the primary range."""
    spec = {k: v for k, v in ALL_GENERATIONS[gen].items() if k not in free}
    return {"num_vars": len(C.PARAM_NAMES), "names": C.PARAM_NAMES,
            "bounds": [list(spec.get(name, b)) for name, b, _, _, _ in C.PARAMETERS]}


def _stats(lcc: np.ndarray) -> dict:
    q = np.quantile(lcc, [0.05, 0.25, 0.75, 0.95])
    return dict(lcc_mean=lcc.mean(), lcc_median=np.median(lcc),
                lcc_q05=q[0], lcc_q25=q[1], lcc_q75=q[2], lcc_q95=q[3],
                lcc_frac_above_095=(lcc > 0.95).mean(),
                lcc_frac_below_005=(lcc < 0.05).mean())


def summarise(lcc, usable, pairs, k, rng) -> dict:
    """Point estimates plus cluster-bootstrap intervals over the base blocks."""
    n_blocks = len(lcc) // STRIDE
    blocks = lcc.reshape(n_blocks, STRIDE)
    point = {name: float(v) for name, v in _stats(lcc).items()}
    boot = {name: np.empty(N_BOOT) for name in point}
    for b in range(N_BOOT):
        sel = blocks[rng.integers(0, n_blocks, n_blocks)].ravel()
        for name, v in _stats(sel).items():
            boot[name][b] = v
    out = {"n_samples": int(len(lcc)), "n_blocks": int(n_blocks)}
    for name, v in point.items():
        lo, hi = np.quantile(boot[name], [0.025, 0.975])
        out[name] = v
        out[name + "_ci"] = [float(lo), float(hi)]
    out.update(lcc_max=float(lcc.max()), lcc_min=float(lcc.min()),
               usable_mean=float(usable.mean()),
               reachable_pairs_mean=float(pairs.mean()),
               frac_no_usable_link=float((k == 0).mean()))
    return out


def run_cell(gen, scen, curves, variant, rng):
    """All thresholds for one (variant, generation, scenario)."""
    prob = problem(gen, VARIANTS[variant]["free"])
    X = sobol_sample.sample(prob, N_BASE, calc_second_order=False, seed=C.SEED)
    K = np.empty((len(X), len(C.R_MIN_SWEEP)), dtype=np.int64)
    for i, row in enumerate(X):
        d = dict(zip(prob["names"], row))
        for name in C.LOG_PARAMS:
            d[name] = 10.0 ** d[name]
        skr = secret_key_rate(curves.km, **d)
        for j, r_min in enumerate(C.R_MIN_SWEEP):
            K[i, j] = PM.usable_count(skr, r_min)
    res, samples = {}, {}
    for j, r_min in enumerate(C.R_MIN_SWEEP):
        k = K[:, j]
        lcc = curves.lcc_frac[k]
        res[TAGS[r_min]] = summarise(lcc, k / curves.n_edges,
                                     curves.reachable_pairs[k], k, rng)
        samples[TAGS[r_min]] = lcc
    return prob, res, samples


def main() -> int:
    primary = TAGS[C.R_MIN_PRIMARY]
    rng = np.random.default_rng(C.SEED)
    out: dict = {
        "n_base": N_BASE, "r_min": C.R_MIN_PRIMARY,
        "design": {
            "sampler": "SALib Saltelli/Sobol cross design, calc_second_order=False",
            "n_base": N_BASE, "rows_per_block": STRIDE,
            "n_samples": N_BASE * STRIDE,
            "independent_draws": False,
            "note": ("256 base blocks x 9 rows; rows within a block share "
                     "coordinates. Intervals are a cluster bootstrap over the "
                     "256 blocks."),
            "bootstrap_resamples": N_BOOT, "conf_level": 0.95,
            "seed": C.SEED,
        },
        "bounds": {}, "generations": {}, "by_threshold": {},
    }
    keep = {}

    for variant, spec in VARIANTS.items():
        out["by_threshold"][variant] = {}
        for scen in C.SCENARIOS:
            topo = load_topology(scen)
            curves = PM.build(topo, topo.edge_km * spec["route"])
            for gen in ORDER:
                prob, res, samples = run_cell(gen, scen, curves, variant, rng)
                out["bounds"].setdefault(variant, {})[gen] = dict(
                    zip(prob["names"], prob["bounds"]))
                out["by_threshold"][variant].setdefault(gen, {})[scen] = res
                if variant == "baseline":
                    for tag, lcc in samples.items():
                        keep[f"{gen}_{scen}_{tag}"] = lcc
                for tag, st in res.items():
                    print(f"  {variant:<14} {gen:<12} {scen:<6} @{tag:<4} "
                          f"mean={st['lcc_mean']:.3f} "
                          f"[{st['lcc_mean_ci'][0]:.3f},{st['lcc_mean_ci'][1]:.3f}] "
                          f"median={st['lcc_median']:.3f} "
                          f">0.95:{100*st['lcc_frac_above_095']:5.1f}% "
                          f"<0.05:{100*st['lcc_frac_below_005']:5.1f}%",
                          flush=True)

    # Pre-round-3 layout, kept so existing readers of this file still work:
    # the two original generations at the primary threshold.
    base = out["by_threshold"]["baseline"]
    legacy = ("n_samples", "lcc_mean", "lcc_median", "lcc_frac_above_095",
              "lcc_frac_below_005", "lcc_max", "lcc_min", "usable_mean",
              "reachable_pairs_mean")
    for gen in GENERATIONS:
        out["generations"][gen] = {
            scen: {k: base[gen][scen][primary][k] for k in legacy}
            for scen in C.SCENARIOS}

    # Do the two generations overlap anywhere? At every threshold and variant.
    out["overlap"] = {}
    out["overlap_by_threshold"] = {}
    for variant in VARIANTS:
        bt = out["by_threshold"][variant]
        out["overlap_by_threshold"][variant] = {}
        for scen in C.SCENARIOS:
            cell = {}
            for tag in TAGS.values():
                a, b = bt["ingaas"][scen][tag], bt["snspd"][scen][tag]
                cell[tag] = {
                    "ingaas_max_lcc": a["lcc_max"], "snspd_min_lcc": b["lcc_min"],
                    "disjoint": bool(a["lcc_max"] < b["lcc_min"]),
                    "mean_gap_snspd_minus_ingaas": b["lcc_mean"] - a["lcc_mean"],
                }
            out["overlap_by_threshold"][variant][scen] = cell
            if variant == "baseline":
                out["overlap"][scen] = {k: cell[primary][k] for k in
                                        ("ingaas_max_lcc", "snspd_min_lcc",
                                         "disjoint")}
    for scen in C.SCENARIOS:
        o = out["overlap"][scen]
        print(f"\n  {scen}: InGaAs max S_LCC = {o['ingaas_max_lcc']:.3f}, "
              f"SNSPD min S_LCC = {o['snspd_min_lcc']:.3f} -> "
              f"{'disjoint' if o['disjoint'] else 'OVERLAPPING'}")

    (C.RESULTS / "generation_stats.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(C.RESULTS / "generation_samples.npz", **keep)
    print(f"\nWritten to {C.RESULTS / 'generation_stats.json'} "
          f"and generation_samples.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
