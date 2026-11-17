"""Sobol estimators with a bootstrap that keeps the resamples.

SALib returns each index with a confidence half-width but discards the
resamples, so it cannot give an interval for a function of several indices.
The interaction share 1 - sum_i S_i is such a function, and so is the
difference between the shares of two outputs or two models.

This module re-implements exactly the estimators SALib 1.5.2 uses in
`SALib.analyze.sobol` (first order: Saltelli et al. 2010; total order: Jansen
1999) and the same bootstrap scheme (resample the N base rows with
replacement, keeping each row's A, B and AB evaluations together). The point
estimates agree with SALib to rounding; `check_against_salib` verifies that.

The bootstrap is drawn as a multinomial count matrix, which is the same thing
as resampling row indices and lets every output and model analysed with the
same matrix share its resamples. Shares computed from the same matrix are
therefore paired, and their differences have honest intervals.

Caveat that applies to SALib's intervals as well: the base rows come from a
scrambled Sobol sequence, not an i.i.d. sample, and the bootstrap treats them
as i.i.d. That is the usual practice and is generally conservative.
"""

from __future__ import annotations

import numpy as np

import config as C

D = len(C.PARAM_NAMES)
STRIDE = D + 2


def split(y: np.ndarray):
    """A, B (length N) and AB (N x D) from a Saltelli-ordered output vector."""
    A = y[0::STRIDE]
    B = y[STRIDE - 1::STRIDE]
    AB = np.column_stack([y[j + 1::STRIDE] for j in range(D)])
    return A, B, AB


def bootstrap_weights(n_base: int, n_resamples: int, seed: int = C.SEED):
    """Multinomial resampling counts, shape (n_resamples, n_base)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_base, np.full(n_base, 1.0 / n_base),
                           size=n_resamples).astype(float)


def indices(y: np.ndarray, W: np.ndarray | None = None) -> dict:
    """S1, ST and interaction share, with bootstrap resamples if W is given.

    Returns None if the output is constant in the A or B sub-design, where
    the decomposition is undefined.
    """
    y = np.asarray(y, dtype=float)
    A, B, AB = split(y)
    if np.ptp(A) <= 1e-12 or np.ptp(B) <= 1e-12:
        return None
    y = (y - y.mean()) / y.std()          # SALib normalises first
    A, B, AB = split(y)
    n = len(A)

    p1 = B[:, None] * (AB - A[:, None])            # first-order summands
    pt = 0.5 * (A[:, None] - AB) ** 2              # total-order summands
    var = np.var(np.r_[A, B])
    out = {"S1": p1.mean(axis=0) / var, "ST": pt.mean(axis=0) / var}
    out["sum_S1"] = float(out["S1"].sum())
    out["share_raw"] = 1.0 - out["sum_S1"]
    out["share"] = max(0.0, out["share_raw"])      # as summarize_results.py

    if W is not None:
        assert W.shape[1] == n
        s_a, s_b = W @ A, W @ B
        mean_ab = (s_a + s_b) / (2 * n)
        var_r = (W @ A ** 2 + W @ B ** 2) / (2 * n) - mean_ab ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            S1_r = (W @ p1) / n / var_r[:, None]
            ST_r = (W @ pt) / n / var_r[:, None]
        ok = var_r > 1e-12
        S1_r[~ok], ST_r[~ok] = np.nan, np.nan
        out["S1_boot"], out["ST_boot"] = S1_r, ST_r
        out["share_boot"] = 1.0 - S1_r.sum(axis=1)
        out["n_boot_undefined"] = int((~ok).sum())
    return out


def interval(boot: np.ndarray, level: float = C.CONF_LEVEL):
    """Percentile interval and SALib-style normal half-width of resamples."""
    b = boot[np.isfinite(boot)]
    if len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.quantile(b, [(1 - level) / 2, 1 - (1 - level) / 2])
    from scipy.stats import norm
    half = float(norm.ppf(0.5 + level / 2) * b.std(ddof=1))
    return float(lo), float(hi), half


def check_against_salib(y: np.ndarray) -> float:
    """Largest |difference| between these point estimates and SALib's."""
    from SALib.analyze import sobol as sobol_analyze
    si = sobol_analyze.analyze(C.SALIB_PROBLEM, y, calc_second_order=False,
                               num_resamples=10, seed=C.SEED,
                               print_to_console=False)
    mine = indices(y)
    return float(max(np.abs(mine["S1"] - si["S1"]).max(),
                     np.abs(mine["ST"] - si["ST"]).max()))
