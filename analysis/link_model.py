"""Asymptotic decoy-state BB84 link model.

Implements equations (1)-(11) of the accompanying article. All functions are vectorised
over an array of link distances so that one parameter sample can be evaluated
against every candidate edge of a topology in a single call.

Reference model: Ma, X.; Qi, B.; Zhao, Y.; Lo, H.-K. Practical decoy state for
quantum key distribution. Phys. Rev. A 2005, 72, 012326.
"""

from __future__ import annotations

import numpy as np

from config import E_0, F_REP, Q_SIFT


def binary_entropy(x: np.ndarray) -> np.ndarray:
    """Binary Shannon entropy H_2, in bits.

    Defined as 0 at x <= 0 and saturating at 1 for x >= 1/2. The saturation is
    deliberate: an error rate at or above 1/2 carries no extractable key, and
    clamping keeps the key-rate bound monotone rather than letting H_2 fall
    again above 1/2.
    """
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mid = (x > 0.0) & (x < 0.5)
    xm = x[mid]
    out[mid] = -xm * np.log2(xm) - (1.0 - xm) * np.log2(1.0 - xm)
    out[x <= 0.0] = 0.0
    return out


def channel_transmittance(d_km: np.ndarray, alpha: float, L0: float) -> np.ndarray:
    """Channel transmittance for length `d_km`.

    `alpha` is the fibre attenuation coefficient in dB/km and `L0` a fixed
    insertion loss in dB representing connectors, patch panels and the splice
    budget. The fixed term matters most on short metropolitan links, where it
    can exceed the distance-dependent contribution entirely -- see
    PARAMETER_RANGES.md section 2.1.
    """
    return 10.0 ** (-(alpha * np.asarray(d_km, dtype=float) + L0) / 10.0)


def secret_key_rate(
    d_km: np.ndarray,
    alpha: float,
    eta_det: float,
    p_dc: float,
    e_d: float,
    mu: float,
    f_EC: float,
    L0: float,
) -> np.ndarray:
    """Secret-key rate in bit/s for each link length in `d_km`.

    Returns `f_rep * max(0, R_pulse)` with R_pulse the asymptotic decoy-state
    BB84 lower bound of manuscript eq. (1).
    """
    eta_ch = channel_transmittance(d_km, alpha, L0)
    eta = eta_det * eta_ch

    # Two detectors, independent dark counts per gate.
    Y0 = 1.0 - (1.0 - p_dc) ** 2

    # Overall gain and error gain.
    one_minus_exp = -np.expm1(-eta * mu)          # 1 - exp(-eta*mu), stable
    Q_mu = Y0 + one_minus_exp - Y0 * one_minus_exp  # = 1 - (1-Y0)exp(-eta*mu)
    E_mu_Q_mu = E_0 * Y0 + e_d * one_minus_exp

    with np.errstate(divide="ignore", invalid="ignore"):
        E_mu = np.where(Q_mu > 0.0, E_mu_Q_mu / Q_mu, 0.5)

    # Single-photon contributions under the asymptotic decoy-state estimate.
    Y1 = 1.0 - (1.0 - Y0) * (1.0 - eta)
    Q1 = mu * np.exp(-mu) * Y1
    with np.errstate(divide="ignore", invalid="ignore"):
        e1 = np.where(Y1 > 0.0, (E_0 * Y0 + e_d * eta) / Y1, 0.5)

    r_pulse = Q_SIFT * (
        Q1 * (1.0 - binary_entropy(e1)) - Q_mu * f_EC * binary_entropy(E_mu)
    )
    return F_REP * np.maximum(0.0, r_pulse)


def max_secure_distance(
    alpha: float,
    eta_det: float,
    p_dc: float,
    e_d: float,
    mu: float,
    f_EC: float,
    L0: float,
    r_min: float = 0.0,
    hi_km: float = 1000.0,
    tol_km: float = 0.01,
) -> float:
    """Largest distance whose SKR still exceeds `r_min`, by bisection.

    Returns 0.0 if the rate already fails at zero distance. The key rate is
    monotonically decreasing in distance for fixed parameters, so bisection is
    exact to `tol_km`.
    """
    def rate(d):
        return float(secret_key_rate(
            np.array([d]), alpha, eta_det, p_dc, e_d, mu, f_EC, L0)[0])

    if rate(0.0) <= r_min:
        return 0.0
    if rate(hi_km) > r_min:
        return hi_km

    lo, hi = 0.0, hi_km
    while hi - lo > tol_km:
        mid = 0.5 * (lo + hi)
        if rate(mid) > r_min:
            lo = mid
        else:
            hi = mid
    return lo
