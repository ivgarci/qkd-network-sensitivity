"""Validation of the link model against published reference results.

Run:  .venv/bin/python analysis/validate.py

The model is checked against the GYS parameter set as tabulated by Ma et al.
(2005), Table 1, for which that paper reports two independent, quotable
results: an asymptotic secure distance of about 140 km, and an optimal signal
intensity of about 0.48 for f(e) = 1.22 (about 0.54 for f(e) = 1).

Reproducing both is a genuine test: the distance probes the dark-count and
loss handling, the optimum probes the single-photon gain and entropy terms.
"""

from __future__ import annotations

import sys

import numpy as np

from link_model import max_secure_distance, secret_key_rate

# GYS parameters, Ma et al. (2005) Table 1. Y0 = 1.7e-6 is the background rate
# per pulse; the model derives it from p_dc via Y0 = 1-(1-p_dc)^2 ~ 2*p_dc.
GYS = dict(
    alpha=0.21,
    eta_det=0.045,   # eta_Bob: total receiver efficiency, not bare detector QE
    p_dc=0.85e-6,    # gives Y0 = 1.7e-6
    e_d=0.033,
    f_EC=1.22,
    L0=0.0,          # Ma et al. model carries no fixed insertion loss
)

CHECKS: list[tuple[str, float, float, str]] = []


def check(name, got, expected, tol, unit=""):
    ok = abs(got - expected) <= tol
    CHECKS.append((name, got, expected, "PASS" if ok else "FAIL"))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}\n"
          f"         got {got:.4g}{unit}, expected {expected:.4g}{unit} "
          f"(tol {tol:g}{unit})")
    return ok


def main() -> int:
    print("Validating link model against Ma et al. (2005), GYS parameters\n")

    # --- 1. Y0 recovers the tabulated background rate ------------------------
    Y0 = 1.0 - (1.0 - GYS["p_dc"]) ** 2
    check("Background rate Y0", Y0, 1.7e-6, 1e-9)

    # --- 2. Optimal mu -------------------------------------------------------
    # Ma et al: mu_optimal ~ 0.48 for f(e)=1.22, ~0.54 for f(e)=1.
    grid = np.linspace(0.05, 1.5, 2901)
    for f_ec, expected_mu in ((1.22, 0.48), (1.00, 0.54)):
        p = dict(GYS, f_EC=f_ec)
        # Optimise at a short distance, as Ma et al. do when deriving the
        # closed-form optimum (the optimum is distance-insensitive).
        rates = np.array([
            secret_key_rate(np.array([0.0]), mu=m, **p)[0] for m in grid
        ])
        mu_opt = float(grid[int(np.argmax(rates))])
        check(f"Optimal mu (f_EC={f_ec})", mu_opt, expected_mu, 0.03)

    # --- 3. Maximum secure distance -----------------------------------------
    # Ma et al. state decoy-state QKD is secure over 140 km with GYS.
    d_max = max_secure_distance(mu=0.48, **GYS)
    check("Max secure distance", d_max, 140.0, 8.0, " km")

    # --- 4. Sanity: monotone decrease in distance ----------------------------
    d = np.linspace(0, 200, 401)
    r = secret_key_rate(d, mu=0.48, **GYS)
    positive = r[r > 0]
    mono = bool(np.all(np.diff(positive) <= 1e-9 * np.maximum(1, positive[:-1])))
    CHECKS.append(("SKR monotone decreasing", float(mono), 1.0,
                   "PASS" if mono else "FAIL"))
    print(f"  [{'PASS' if mono else 'FAIL'}] SKR monotone decreasing in distance")

    # --- 5. Sanity: zero beyond the secure distance --------------------------
    beyond = float(secret_key_rate(np.array([d_max + 20.0]), mu=0.48, **GYS)[0])
    ok = beyond == 0.0
    CHECKS.append(("SKR zero beyond secure distance", beyond, 0.0,
                   "PASS" if ok else "FAIL"))
    print(f"  [{'PASS' if ok else 'FAIL'}] SKR is exactly 0 beyond secure distance")

    failed = [c for c in CHECKS if c[3] == "FAIL"]
    # The manuscript quotes the reproduced values (secure distance, optimal
    # mu), so they are stored like every other quoted number.
    import json
    from config import RESULTS
    (RESULTS / "validation.json").write_text(json.dumps(
        {"reference": "Ma et al. (2005), GYS parameters", "parameters": GYS,
         "checks": [dict(name=n, reproduced=g, reference=e, status=st)
                    for n, g, e, st in CHECKS]}, indent=2))
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(c[0] for c in failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
