"""Within-generation parameter rankings (roadmap R3-3).

Run:  .venv/bin/python analysis/generation_rankings.py
Reads  results/robustness_indices.csv, results/robustness_summary.json
Writes results/generation_rankings.csv, results/generation_rankings_summary.json

The single-generation Sobol runs (configs `ingaas` and `snspd` of
robustness.py, N = 1024, 200 bootstrap resamples) have been in results/ since
round 1, but no ranking from them was ever tabulated. This script ranks the
seven inputs by total-order index for every generation, scenario and output,
and applies the paper's own resolution rule (summarize_results.CONF_FRAC: the
ranking of an output is established only if the bootstrap half-width of its
leading total-order index is at most 25% of that index).

Nothing is re-estimated here. The intervals are the ST_conf half-widths stored
by robustness.py; first-order indices carry no interval in that file.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import config as C
from summarize_results import CONF_FRAC

GENERATION_CONFIGS = ("ingaas", "snspd")
PRIMARY_TAG = f"{C.R_MIN_PRIMARY:.0e}".replace("+0", "").replace("+", "")


def ranked_table(rows):
    """Rank one output's seven rows by ST and attach resolution flags."""
    t = sorted(rows, key=lambda r: -r["ST"])
    lead = t[0]
    resolved = not (lead["ST"] <= 0 or lead["ST_conf"] > CONF_FRAC * lead["ST"])
    out = []
    for i, r in enumerate(t):
        lo, hi = r["ST"] - r["ST_conf"], r["ST"] + r["ST_conf"]
        nxt = t[i + 1] if i + 1 < len(t) else None
        prv = t[i - 1] if i else None
        out.append(dict(
            r, rank=i + 1, ST_lo=lo, ST_hi=hi,
            overlaps_next_rank=bool(nxt and lo <= nxt["ST"] + nxt["ST_conf"]),
            overlaps_previous_rank=bool(prv and hi >= prv["ST"] - prv["ST_conf"]),
            leading_parameter=lead["parameter"],
            ranking_resolved=resolved))
    return out


def load_control_rows(configs=None):
    groups = defaultdict(list)
    with open(C.RESULTS / "robustness_indices.csv") as fh:
        for r in csv.DictReader(fh):
            if configs and r["config"] not in configs:
                continue
            for k in ("S1", "ST", "ST_conf"):
                r[k] = float(r[k])
            groups[(r["config"], r["scenario"], r["n_sources"], r["output"])].append(r)
    return groups


def main() -> int:
    groups = load_control_rows(GENERATION_CONFIGS)
    rs = json.loads((C.RESULTS / "robustness_summary.json").read_text())

    fields = ["generation", "scenario", "output", "metric", "threshold",
              "parameter", "rank", "ST", "ST_conf", "ST_lo", "ST_hi", "S1",
              "overlaps_next_rank", "overlaps_previous_rank",
              "leading_parameter", "ranking_resolved", "degenerate", "n_base",
              "n_bootstrap"]
    out_rows, summary = [], {}
    for (config, scen, _, output), rows in sorted(groups.items()):
        metric, _, thr = output.partition("@")
        for r in ranked_table(rows):
            out_rows.append(dict(
                generation=config, scenario=scen, output=output, metric=metric,
                threshold=thr or "none", parameter=r["parameter"],
                rank=r["rank"], ST=r["ST"], ST_conf=r["ST_conf"],
                ST_lo=r["ST_lo"], ST_hi=r["ST_hi"], S1=r["S1"],
                overlaps_next_rank=r["overlaps_next_rank"],
                overlaps_previous_rank=r["overlaps_previous_rank"],
                leading_parameter=r["leading_parameter"],
                ranking_resolved=r["ranking_resolved"], degenerate=False,
                n_base=r["n_base"], n_bootstrap=200))
    # Outputs that were constant in the generation: listed, never ranked.
    for config in GENERATION_CONFIGS:
        for scen in C.SCENARIOS:
            for output in rs["degenerate_outputs"].get(f"{config}/{scen}", []):
                metric, _, thr = output.partition("@")
                out_rows.append(dict.fromkeys(fields, ""))
                out_rows[-1].update(generation=config, scenario=scen,
                                    output=output, metric=metric,
                                    threshold=thr or "none", degenerate=True,
                                    n_base=rs["n_base"], n_bootstrap=200)

    with open(C.RESULTS / "generation_rankings.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    # Summary for the manuscript: what leads, where eta_det sits, what resolves.
    primary_outputs = ["median_skr_log10"] + [
        f"{n}@{PRIMARY_TAG}" for n, _, _ in C.OUTPUTS_THRESHOLDED]
    for config in GENERATION_CONFIGS:
        summary[config] = {}
        for scen in C.SCENARIOS:
            sub = [r for r in out_rows if r["generation"] == config
                   and r["scenario"] == scen and not r["degenerate"]]
            outputs = sorted({r["output"] for r in sub})
            unresolved = sorted({r["output"] for r in sub
                                 if not r["ranking_resolved"]})
            cell = {"n_outputs_nondegenerate": len(outputs),
                    "n_outputs_unresolved": len(unresolved),
                    "unresolved_outputs": unresolved,
                    "primary_threshold": {}}
            for o in primary_outputs:
                t = sorted((r for r in sub if r["output"] == o),
                           key=lambda r: r["rank"])
                if not t:
                    continue
                cell["primary_threshold"][o] = {
                    "order": [r["parameter"] for r in t],
                    "ST": {r["parameter"]: r["ST"] for r in t},
                    "ST_conf": {r["parameter"]: r["ST_conf"] for r in t},
                    "eta_det_rank": next(r["rank"] for r in t
                                         if r["parameter"] == "eta_det"),
                    "alpha_rank": next(r["rank"] for r in t
                                       if r["parameter"] == "alpha"),
                    "ranking_resolved": t[0]["ranking_resolved"],
                    "first_two_overlap": t[0]["overlaps_next_rank"],
                    "second_third_overlap": t[1]["overlaps_next_rank"],
                }
            summary[config][scen] = cell
            lead = {o: v["order"][0] for o, v in cell["primary_threshold"].items()}
            eta = [v["eta_det_rank"] for v in cell["primary_threshold"].values()]
            print(f"  {config:<7} {scen:<6} unresolved "
                  f"{cell['n_outputs_unresolved']}/{len(outputs)}; "
                  f"eta_det ranks at primary threshold {eta}; leaders "
                  f"{sorted(set(lead.values()))}")
    tot = {c: [sum(summary[c][s]["n_outputs_unresolved"] for s in C.SCENARIOS),
               sum(summary[c][s]["n_outputs_nondegenerate"] for s in C.SCENARIOS)]
           for c in GENERATION_CONFIGS}
    summary["totals_unresolved_of_nondegenerate"] = tot
    summary["resolution_rule"] = (
        f"ranking of an output is established only if the bootstrap half-width "
        f"of its leading total-order index is at most {CONF_FRAC:.0%} of the "
        f"index (summarize_results.CONF_FRAC)")
    summary["source"] = "results/robustness_indices.csv (N=1024, 200 resamples)"
    (C.RESULTS / "generation_rankings_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"  totals unresolved/non-degenerate: {tot}")
    print(f"Written to {C.RESULTS / 'generation_rankings.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
