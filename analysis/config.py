"""Configuration for the QKD optoelectronics sensitivity analysis.

Every numerical choice recorded here is either (a) grounded in the literature
review documented in PARAMETER_RANGES.md, or (b) an explicit modelling decision
flagged in the same document. Nothing here is arbitrary.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results"

# ---------------------------------------------------------------------------
# Uncertain inputs
# ---------------------------------------------------------------------------
# Ranges and their justification: see analysis/PARAMETER_RANGES.md section 2.
# `log` marks inputs sampled log-uniformly: the sampler draws log10(x) uniformly
# over the stated bounds and the model exponentiates.

PARAMETERS = [
    # name,      bounds,            log,   unit,           label
    ("alpha",    (0.17, 0.25),      False, "dB/km",        r"$\alpha$"),
    ("eta_det",  (0.10, 0.85),      False, "-",            r"$\eta_{\mathrm{det}}$"),
    ("p_dc",     (-8.0, -5.0),      True,  "gate$^{-1}$",  r"$p_{\mathrm{dc}}$"),
    ("e_d",      (0.005, 0.040),    False, "-",            r"$e_{\mathrm{d}}$"),
    ("mu",       (0.20, 0.80),      False, "photons/pulse", r"$\mu$"),
    ("f_EC",     (1.05, 1.22),      False, "-",            r"$f_{\mathrm{EC}}$"),
    ("L0",       (1.0, 4.0),        False, "dB",           r"$L_0$"),
]

PARAM_NAMES = [p[0] for p in PARAMETERS]
LOG_PARAMS = {p[0] for p in PARAMETERS if p[2]}

SALIB_PROBLEM = {
    "num_vars": len(PARAMETERS),
    "names": PARAM_NAMES,
    "bounds": [list(p[1]) for p in PARAMETERS],
}

# ---------------------------------------------------------------------------
# Fixed protocol / system constants
# ---------------------------------------------------------------------------
# q: basis-sifting factor. 1/2 for standard BB84 with symmetric basis choice,
#    as used by Ma et al. (2005).
Q_SIFT = 0.5

# e_0: error rate of background/dark-count events (random) = 1/2.
E_0 = 0.5

# f_rep: source repetition rate [Hz]. Held fixed (manuscript eq. 2) because its
# effect on R_bps is purely multiplicative. 1 GHz is representative of modern
# commercial fibre QKD transmitters.
#
# NOTE: because R_bps = f_rep * R_pulse, the usable-edge test R_bps >= R_min is
# identical to R_pulse >= R_min / f_rep. The threshold sweep over R_min below
# therefore doubles as an f_rep sweep spanning a factor of twenty (2e7 / 1e6),
# so no separate f_rep robustness analysis is required. (An earlier version of
# this comment said "two orders of magnitude", which was wrong.)
F_REP = 1.0e9

# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
# Candidate-edge rule: a node pair is a candidate link if its great-circle
# distance is at most MAX_EDGE_KM. 45 km matches the AdjacencyMatrixNamed45
# convention of Garcia-Cobo & Menendez (2021), keeping this study's topology
# continuous with that earlier medoid-based work.
MAX_EDGE_KM = 45.0

# `seed_offset` is stated explicitly rather than derived from the scenario name.
# Deriving it with Python's built-in hash() is not reproducible: string hashing
# is randomised per interpreter, so every process -- including each spawned
# multiprocessing worker -- would draw a different source set, silently
# contaminating the estimators that depend on it.
SCENARIOS = {
    "cyl": {
        "label": "Castilla y Leon",
        "csv": DATA / "cyl_267.csv",
        "expected_nodes": 267,
        "seed_offset": 101,
    },
    "spain": {
        "label": "Mainland Spain",
        "csv": DATA / "spain_3102.csv",
        "expected_nodes": 3102,
        "seed_offset": 202,
    },
}

# ---------------------------------------------------------------------------
# Operational thresholds
# ---------------------------------------------------------------------------
# The binding quantity is R_min / f_rep, the required key bits per pulse. With
# a 45 km candidate-edge rule and f_rep = 1 GHz, the model yields R_pulse in
# roughly [2.1e-5, 1.8e-2] across the parameter space, so a threshold below
# about 2e4 bit/s is met by every link in every sample and the network outputs
# degenerate to constants. The thresholds below place R_min/f_rep inside that
# window, framing the study as a high-rate backbone service requirement.
#
# A threshold scan over 1e6 .. 3e7 bit/s locates the connectivity transition by
# the spread of S_LCC across the parameter sample:
#
#   R_min      1e6     2e6     3e6     5e6     7e6     1e7     2e7     3e7
#   S_LCC sd   .019    .146    .260    .378    .387    .339    .135    .038
#
# The transition is centred near 5-7e6, so 5 Mbit/s is the primary threshold
# and the sweep brackets the transition from near-saturation to near-collapse.
#
# Round-3 correction. The table above was typed in from an interactive check
# whose scenario and sample were not recorded; it is kept as the historical
# record of how the threshold was chosen. The stored, reproducible scan is
# analysis/threshold_scan.py -> results/threshold_scan.csv, on the principal
# design. It does not reproduce those eight values (differences up to 0.05 for
# Castilla y Leon and 0.06 for mainland Spain; S_LCC sd at 1e6 is 0.063 / 0.047,
# not 0.019). It does confirm the conclusion: the spread peaks at 6e6 (CyL) and
# 7-8e6 (Spain) and is within 10% of its peak from 5e6 to 8e6 / 1e7.
#
# This is a deviation from analysis/EXPERIMENT_DESIGN.md, which pre-specified
# 1e3 bit/s (sweep 1e2, 1e3, 1e4). The threshold was moved after model output
# had been seen, to the region where the outputs vary most.
# 1e5 was dropped from the sweep: every link passes it in every sample, which
# leaves the network outputs constant and their Sobol indices undefined.
R_MIN_PRIMARY = 5.0e6                       # bit/s
R_MIN_SWEEP = (1.0e6, 5.0e6, 2.0e7)         # bit/s

# ---------------------------------------------------------------------------
# Sampling and estimation
# ---------------------------------------------------------------------------
N_BASE = 4096                 # Sobol base sample size
CALC_SECOND_ORDER = False     # first- and total-order only
N_BOOTSTRAP = 1000            # bootstrap resamples for confidence intervals
CONF_LEVEL = 0.95
CONVERGENCE_SIZES = (256, 512, 1024, 2048, 4096)

SEED = 20260917               # master seed, recorded for reproducibility

# Global efficiency and bottleneck capacity are estimated from a FIXED set of
# source nodes drawn once per scenario with SEED. The set must not be redrawn
# per Sobol sample: resampling would inject estimator noise into the variance
# decomposition and inflate the apparent interaction terms.
N_SOURCES_EFFICIENCY = 96
N_SOURCES_BOTTLENECK = 96

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
# Threshold-free outputs are computed once per parameter sample; the remainder
# are computed once per (sample, threshold) pair.

OUTPUTS_THRESHOLD_FREE = [
    ("median_skr_log10", "Median link SKR", r"$\widetilde{R}$"),
]

OUTPUTS_THRESHOLDED = [
    ("reach_km",        "Operational reach",         r"$d_{\max}$"),
    ("usable_frac",     "Usable-edge fraction",      r"$\rho_E$"),
    ("lcc_frac",        "Largest-component fraction", r"$S_{\mathrm{LCC}}$"),
    ("reachable_pairs", "Reachable-pair fraction",   r"$\rho_P$"),
    ("global_eff",      "Global efficiency",         r"$E_G$"),
    ("bottleneck_log10", "Median bottleneck capacity", r"$\widetilde{B}$"),
    ("isolated_frac",   "Isolated-node fraction",    r"$\rho_I$"),
]
