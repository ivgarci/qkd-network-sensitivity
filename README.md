# Global sensitivity analysis of geographically realistic QKD networks

Analysis code for the article *From Optoelectronic Parameters to Network-Level
Performance: A Global Sensitivity Analysis of Geographically Realistic Quantum
Key Distribution Networks* (Iván García-Cobo, University of Salamanca;
submitted to *Optoelectronics*, MDPI).

The code propagates uncertainty in seven optoelectronic and protocol inputs
through an asymptotic decoy-state BB84 link model to network-level outputs,
and decomposes the output variance with first- and total-order Sobol indices.
The networks are hypothetical: candidate links join municipality pairs at most
45 km apart, laid over the real locations of 267 municipalities in Castilla y
León and 3102 in mainland Spain. No deployed network is modelled.

## Layout

| Path | Contents |
|---|---|
| `analysis/` | Link model, network metrics, sampling and estimation, controls, figures |
| `analysis/PARAMETER_RANGES.md` | Literature grounding of each input range |
| `analysis/EXPERIMENT_DESIGN.md` | Pre-specified design, and the recorded deviations from it |
| `data/fetch_data.py` | Rebuilds the two coordinate files (not redistributed; see `data/README.md`) |
| `results/`, `figures/` | Empty; filled by the scripts |

## Reproducing the results

Python 3.11 or later.

```
python3 -m venv .venv
./.venv/bin/python -m pip install -r analysis/requirements.txt
./.venv/bin/python data/fetch_data.py      # downloads and checks the upstream listing

cd analysis

# Checks that need no simulation output
../.venv/bin/python validate.py            # link model vs two published quantities (Ma et al. 2005)
../.venv/bin/python test_metrics.py        # graph metrics vs brute-force reference implementations

# Principal analysis
../.venv/bin/python run_sobol.py           # Sobol design N = 4096, both scenarios, plus convergence study
../.venv/bin/python summarize_results.py   # results/summary.md and summary.json
../.venv/bin/python response_surfaces.py   # conditional response surfaces

# Controls and follow-up analyses
../.venv/bin/python robustness.py          # surrogate model, single-generation designs, distance model, source sets
../.venv/bin/python generation_stats.py    # largest-component distributions per detector-parameter range
../.venv/bin/python threshold_scan.py      # output spread versus service-rate threshold
../.venv/bin/python count_rates.py         # detector count rates implied by the link model
../.venv/bin/python interaction_shares.py  # interaction shares, all outputs and thresholds, with intervals
../.venv/bin/python bottleneck_decomposition.py
../.venv/bin/python generation_rankings.py
../.venv/bin/python claim_support.py

../.venv/bin/python summarize_results.py   # again, to add the follow-up sections
../.venv/bin/python make_figures.py        # all figures as vector PDF
```

`run_sobol.py` evaluates 36,864 configurations per scenario plus the
convergence study; on six to nine cores the full sequence takes a few hours,
most of it the mainland-Spain scenario. `run_sobol.py --n-base 64 --scenarios
cyl --skip-convergence` is a quick smoke test.

All randomness is seeded from `analysis/config.py` (`SEED`), including the
fixed estimator source sets. `results/run_metadata.json` records the seed,
package versions and configuration of each run. Re-running reproduces the
published indices exactly.

## Data

See `data/README.md` for the source of the coordinates, why they are rebuilt
rather than distributed, and three known coordinate collisions.

## Licence

Code: MIT (see `LICENSE`). The coordinate data are not covered by this licence
and are not included.

## Citation

See `CITATION.cff`.
