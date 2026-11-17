"""What the bottleneck-capacity indices measure (roadmap R3-1).

Run:  .venv/bin/python analysis/bottleneck_decomposition.py
Reads  results/raw_outputs_<scenario>.npz           (run_sobol.py, N = 4096)
       results/raw_outputs_surrogate_<scenario>.npz (interaction_shares.py,
                                                     N = 1024; optional)
Writes results/bottleneck_decomposition.json

`network_metrics.evaluate` stores the median bottleneck capacity as
log10(1 + B). When no tracked source node has a usable link there is no pair
to take a median over and the function returns B = 0, i.e. a stored value of
exactly 0.0. Every non-empty sample has B >= R_min, i.e. a stored value of at
least log10(1 + R_min) = 6.0 to 7.3. The stored output is therefore a mixture
of a point mass at 0 and a narrow continuous part more than six decades away,
and its variance is dominated by which of the two a sample falls in.

This script quantifies that and re-runs the variance decomposition under
alternative codings of the empty case, on the stored design. The stored
principal results are not modified.

Codings (all computed alongside, none replaces the stored one):

  stored     log10(1 + B) with B = 0 when empty. What sobol_indices.csv holds.
  indicator  1 if the stored value is non-zero, else 0. The two-valued part.
  floor      the empty case is assigned log10(1 + R_min), the infimum of the
             non-empty values: "no connection" is treated as "a connection at
             exactly the required rate". Removes the six-decade jump; a point
             mass at the floor remains.
  linear     B itself in bit/s, with B = 0 when empty. Zero is then a natural
             value (no capacity) rather than a sentinel, at the price of an
             output that is no longer log-compressed.

Every one of these is a choice. There is no coding-free answer to "what is the
interaction share of bottleneck capacity" once some samples have no network to
measure. Conditioning on the non-empty samples is not an option for a Sobol
decomposition: dropping rows breaks the A / B / AB structure of the Saltelli
design and makes the remaining inputs dependent. The conditional (within
non-empty) variance is reported as a descriptive statistic only.
"""

from __future__ import annotations

import json

import numpy as np

import config as C
import sobol_tools as ST

TAGS = {r: f"{r:.0e}".replace("+0", "").replace("+", "") for r in C.R_MIN_SWEEP}
CODINGS = ("stored", "indicator", "floor", "linear")
N_BOOT = C.N_BOOTSTRAP


def load_raw(scenario: str, model: str = "full"):
    name = (f"raw_outputs_{scenario}.npz" if model == "full"
            else f"raw_outputs_{model}_{scenario}.npz")
    path = C.RESULTS / name
    if not path.exists():
        return None, None
    d = np.load(path, allow_pickle=True)
    return d["Y"], [str(c) for c in d["columns"]]


def recode(y: np.ndarray, r_min: float) -> dict:
    """The four codings of one stored bottleneck column."""
    empty = y == 0.0
    return {
        "stored": y,
        "indicator": (~empty).astype(float),
        "floor": np.where(empty, np.log10(1.0 + r_min), y),
        "linear": np.where(empty, 0.0, 10.0 ** y - 1.0),
    }


def sobol_block(y: np.ndarray, W: np.ndarray) -> dict:
    res = ST.indices(y, W)
    if res is None:
        return {"degenerate": True,
                "reason": "constant in the A or B sub-design"}
    lo, hi, half = ST.interval(res["share_boot"])
    block = {"degenerate": False, "output_sd": float(np.std(y)),
             "sum_S1": res["sum_S1"],
             "interaction_share": res["share"],
             "interaction_share_unclipped": res["share_raw"],
             "interaction_share_ci": [lo, hi],
             "interaction_share_half_width": half,
             "parameters": {}}
    for k, p in enumerate(C.PARAM_NAMES):
        block["parameters"][p] = {
            "S1": float(res["S1"][k]),
            "S1_conf": ST.interval(res["S1_boot"][:, k])[2],
            "ST": float(res["ST"][k]),
            "ST_conf": ST.interval(res["ST_boot"][:, k])[2]}
    order = sorted(C.PARAM_NAMES, key=lambda p: -block["parameters"][p]["ST"])
    block["order_by_ST"] = order
    return block


def decompose(Y, cols, r_min: float, W: np.ndarray, n_base: int) -> dict:
    tag = TAGS[r_min]
    n_rows = n_base * ST.STRIDE
    y = Y[:n_rows, cols.index(f"bottleneck_log10@{tag}")]
    usable = Y[:n_rows, cols.index(f"usable_frac@{tag}")]
    empty = y == 0.0
    A, B, _ = ST.split(empty.astype(float))
    p = float(empty.mean())
    out = {
        "r_min_bps": r_min, "n_base": n_base, "n_samples": int(len(y)),
        "frac_sentinel": p,
        "frac_sentinel_A_and_B_rows_only": float(np.r_[A, B].mean()),
        "frac_no_usable_link": float((usable == 0.0).mean()),
        "frac_usable_links_but_no_source_joined":
            float((empty & (usable > 0.0)).mean()),
        "floor_log10": float(np.log10(1.0 + r_min)),
        "stored_variance": float(y.var()),
    }
    if empty.any() and not empty.all():
        nz = y[~empty]
        between = p * (1.0 - p) * (nz.mean() - 0.0) ** 2
        out.update(
            nonempty_min=float(nz.min()), nonempty_max=float(nz.max()),
            nonempty_mean=float(nz.mean()),
            nonempty_variance=float(nz.var()),
            variance_share_of_indicator=float(between / y.var()),
            variance_share_within_nonempty=float((1 - p) * nz.var() / y.var()),
            gap_decades_between_sentinel_and_smallest_value=float(nz.min()))
    elif not empty.any():
        out.update(nonempty_min=float(y.min()), nonempty_max=float(y.max()),
                   nonempty_variance=float(y.var()),
                   variance_share_of_indicator=0.0,
                   variance_share_within_nonempty=1.0)
    out["codings"] = {name: sobol_block(v, W)
                      for name, v in recode(y, r_min).items()}
    return out


def main() -> int:
    out = {
        "definition": {
            "stored": "log10(1 + B), B = 0 when no tracked source has a usable link",
            "indicator": "1 if the stored value is non-zero, else 0",
            "floor": "empty case assigned log10(1 + R_min)",
            "linear": "B in bit/s, 0 when empty",
            "variance_share_of_indicator":
                "between-group variance p(1-p)(mean of non-empty)^2 divided "
                "by the total variance of the stored output (law of total "
                "variance; p = sentinel fraction)",
            "interaction_share": "max(0, 1 - sum of first-order indices), as "
                                 "summarize_results.py; *_unclipped and the "
                                 "interval are not clipped",
            "interval": f"percentile bootstrap over base rows, {N_BOOT} "
                        f"resamples, {C.CONF_LEVEL:.0%}",
            "caveat": "every coding of the empty case is a choice; see the "
                      "module docstring of analysis/bottleneck_decomposition.py",
        },
        "models": {},
    }
    for model, n_base in (("full", C.N_BASE), ("full", 1024), ("surrogate", 1024)):
        key = f"{model}_N{n_base}"
        for scen in C.SCENARIOS:
            Y, cols = load_raw(scen, model)
            if Y is None:
                print(f"  {key}/{scen}: raw outputs missing, skipped")
                continue
            W = ST.bootstrap_weights(n_base, N_BOOT)
            for r_min in C.R_MIN_SWEEP:
                d = decompose(Y, cols, r_min, W, n_base)
                out["models"].setdefault(key, {}).setdefault(scen, {})[
                    TAGS[r_min]] = d
                c = d["codings"]
                msg = (f"  {key:<15} {scen:<6} @{TAGS[r_min]:<4} "
                       f"sentinel {100*d['frac_sentinel']:5.2f}%  "
                       f"indicator variance share "
                       f"{d['variance_share_of_indicator']:.3f}  shares:")
                for name in CODINGS:
                    b = c[name]
                    msg += (f" {name}=n/a" if b["degenerate"] else
                            f" {name}={b['interaction_share_unclipped']:.3f}"
                            f"[{b['interaction_share_ci'][0]:.2f},"
                            f"{b['interaction_share_ci'][1]:.2f}]")
                print(msg, flush=True)
    (C.RESULTS / "bottleneck_decomposition.json").write_text(
        json.dumps(out, indent=2))
    print(f"Written to {C.RESULTS / 'bottleneck_decomposition.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
