"""Condense the Sobol output into the quantities the manuscript cites.

Run:  .venv/bin/python analysis/summarize_results.py

Writes results/summary.md and results/summary.json. Every RESULT marker in
manuscript.tex must be filled from one of these, per the integrity rule in
analysis/EXPERIMENT_DESIGN.md. Nothing in the manuscript may be written from
memory of a figure.
"""

from __future__ import annotations

import csv
import json

import numpy as np

import config as C

PRIMARY_TAG = f"{C.R_MIN_PRIMARY:.0e}".replace("+0", "").replace("+", "")
TAGS = {r: f"{r:.0e}".replace("+0", "").replace("+", "") for r in C.R_MIN_SWEEP}
# The paper's resolution rule: a ranking is established only if the bootstrap
# half-width on the leading total-order index is at most this fraction of the
# index. Module-level so the control-run tables apply the identical rule.
CONF_FRAC = 0.25

SYM = {"alpha": "alpha", "eta_det": "eta_det", "p_dc": "p_dc", "e_d": "e_d",
       "mu": "mu", "f_EC": "f_EC", "L0": "L0"}


def read_rows(path):
    if not path.exists():
        return []
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k in ("S1", "S1_conf", "ST", "ST_conf", "output_std"):
                r[k] = float(r[k]) if r[k] not in ("", "nan") else np.nan
            r["degenerate"] = r["degenerate"] == "True"
            r["n_base"] = int(r["n_base"])
            out.append(r)
    return out


def table(rows, scenario, output):
    sub = [r for r in rows if r["scenario"] == scenario
           and r["output"] == output and not r["degenerate"]]
    return sorted(sub, key=lambda r: -r["ST"])


def main() -> int:
    rows = read_rows(C.RESULTS / "sobol_indices.csv")
    conv = read_rows(C.RESULTS / "convergence.csv")
    if not rows:
        print("No results yet. Run run_sobol.py first.")
        return 1

    summary, lines = {}, []
    lines.append("# Results summary\n")
    lines.append(f"Generated from `results/sobol_indices.csv`. "
                 f"Base sample N = {C.N_BASE}, "
                 f"{C.N_BASE * (len(C.PARAMETERS) + 2):,} evaluations per "
                 f"scenario. Primary threshold "
                 f"R_min = {C.R_MIN_PRIMARY:.0e} bit/s.\n")

    # --- degenerate outputs -------------------------------------------------
    degen = sorted({(r["scenario"], r["output"]) for r in rows
                    if r["degenerate"]})
    lines.append("## Degenerate outputs\n")
    if degen:
        lines.append("These outputs were constant across the whole sample, so "
                     "their variance decomposition is undefined and they are "
                     "excluded from every ranking below:\n")
        for s, o in degen:
            lines.append(f"- `{s}` / `{o}`")
    else:
        lines.append("None: every output varied across the sample.")
    lines.append("")
    summary["degenerate"] = [{"scenario": s, "output": o} for s, o in degen]

    # --- poorly resolved outputs -------------------------------------------
    # An output can vary and still yield indices that are pure estimator noise.
    # The right test is not the size of the output - global efficiency is
    # inherently ~1e-2 because it is a mean inverse distance in km^-1, yet its
    # relative variation is large and its indices are perfectly well resolved.
    # The test that matters is whether the bootstrap actually resolves the
    # indices: if the confidence half-width on the leading ST is comparable to
    # ST itself, the ranking is not established and must not be quoted.
    lines.append("## Poorly resolved outputs\n")
    lines.append(f"Outputs whose leading total-order index is not resolved by "
                 f"the bootstrap: confidence half-width exceeds "
                 f"{CONF_FRAC:.0%} of the index itself. Rankings for these are "
                 f"not established and must not be quoted.\n")
    poor = []
    for scen in C.SCENARIOS:
        seen_o = {r["output"] for r in rows if r["scenario"] == scen}
        for o in sorted(seen_o):
            t = table(rows, scen, o)
            if not t:
                continue
            lead = t[0]
            if lead["ST"] <= 0 or lead["ST_conf"] > CONF_FRAC * lead["ST"]:
                poor.append((scen, o, lead["parameter"], lead["ST"],
                             lead["ST_conf"]))
    if poor:
        lines.append("| scenario | output | leading | ST | +-conf |")
        lines.append("|---|---|---|---|---|")
        for a, b, c, d_, e in poor:
            lines.append(f"| {a} | `{b}` | {SYM[c]} | {d_:.3f} | {e:.3f} |")
    else:
        lines.append("None: every output's leading index is resolved.")
    lines.append("")
    summary["poorly_resolved"] = [
        dict(scenario=a, output=b, parameter=c, ST=d_, ST_conf=e)
        for a, b, c, d_, e in poor]

    # --- baseline output ranges --------------------------------------------
    lines.append("## Baseline output ranges\n")
    base = {}
    for scen in C.SCENARIOS:
        p = C.RESULTS / f"raw_outputs_{scen}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        Y, cols = d["Y"], [str(c) for c in d["columns"]]
        base[scen] = {}
        lines.append(f"### {C.SCENARIOS[scen]['label']}\n")
        lines.append("| output | min | median | max | sd |")
        lines.append("|---|---|---|---|---|")
        want = ["median_skr_log10"] + [f"{n}@{PRIMARY_TAG}"
                                       for n, _, _ in C.OUTPUTS_THRESHOLDED]
        for col in want:
            if col not in cols:
                continue
            y = Y[:, cols.index(col)]
            st = dict(min=float(y.min()), median=float(np.median(y)),
                      max=float(y.max()), sd=float(y.std()))
            base[scen][col] = st
            lines.append(f"| `{col}` | {st['min']:.4g} | {st['median']:.4g} "
                         f"| {st['max']:.4g} | {st['sd']:.4g} |")
        # Median link SKR is stored as log10(1+R); report bit/s as well.
        if "median_skr_log10" in base[scen]:
            s = base[scen]["median_skr_log10"]
            lines.append(f"\nMedian link SKR in bit/s: "
                         f"{10**s['min'] - 1:.4g} to {10**s['max'] - 1:.4g} "
                         f"(median {10**s['median'] - 1:.4g}).")
        lines.append("")
    summary["baseline"] = base

    # --- rankings -----------------------------------------------------------
    lines.append("## Parameter rankings by total-order index\n")
    rank = {}
    outs = ["median_skr_log10"] + [f"{n}@{PRIMARY_TAG}"
                                   for n, _, _ in C.OUTPUTS_THRESHOLDED]
    for scen in C.SCENARIOS:
        lines.append(f"### {C.SCENARIOS[scen]['label']}\n")
        lines.append("| output | 1st (ST) | 2nd | 3rd | sum S1 | "
                     "interaction share |")
        lines.append("|---|---|---|---|---|---|")
        rank[scen] = {}
        for o in outs:
            t = table(rows, scen, o)
            if not t:
                continue
            s1_sum = float(np.nansum([r["S1"] for r in t]))
            inter = max(0.0, 1.0 - s1_sum)
            top = [(r["parameter"], r["ST"], r["S1"], r["ST_conf"]) for r in t[:3]]
            rank[scen][o] = dict(
                top=[dict(parameter=p, ST=st, S1=s1, ST_conf=c)
                     for p, st, s1, c in top],
                sum_S1=s1_sum, interaction_share=inter)
            cells = " | ".join(
                f"{SYM[p]} {st:.3f}+-{c:.3f}" for p, st, _, c in top)
            lines.append(f"| `{o}` | {cells} | {s1_sum:.3f} | {inter:.3f} |")
        lines.append("")
    summary["rankings"] = rank

    # --- interaction strength ----------------------------------------------
    lines.append("## Strongest interactions (largest ST - S1)\n")
    lines.append("| scenario | output | parameter | S1 | ST | ST - S1 |")
    lines.append("|---|---|---|---|---|---|")
    gaps = []
    for r in rows:
        if r["degenerate"] or r["output"] not in outs:
            continue
        gaps.append((r["ST"] - r["S1"], r))
    for g, r in sorted(gaps, key=lambda t: -t[0])[:10]:
        lines.append(f"| {r['scenario']} | `{r['output']}` | "
                     f"{SYM[r['parameter']]} | {r['S1']:.3f} | {r['ST']:.3f} "
                     f"| {g:.3f} |")
    summary["top_interactions"] = [
        dict(scenario=r["scenario"], output=r["output"],
             parameter=r["parameter"], S1=r["S1"], ST=r["ST"], gap=g)
        for g, r in sorted(gaps, key=lambda t: -t[0])[:10]]
    lines.append("")

    # --- link vs network comparison ----------------------------------------
    lines.append("## Link-level versus network-level ranking\n")
    lines.append("| scenario | link output top | network output top | same? |")
    lines.append("|---|---|---|---|")
    cmp_out = {}
    for scen in C.SCENARIOS:
        lk = table(rows, scen, "median_skr_log10")
        nw = table(rows, scen, f"lcc_frac@{PRIMARY_TAG}")
        if not lk or not nw:
            continue
        same = lk[0]["parameter"] == nw[0]["parameter"]
        cmp_out[scen] = dict(link=lk[0]["parameter"], link_ST=lk[0]["ST"],
                             network=nw[0]["parameter"], network_ST=nw[0]["ST"],
                             same=same)
        lines.append(f"| {scen} | {SYM[lk[0]['parameter']]} "
                     f"({lk[0]['ST']:.3f}) | {SYM[nw[0]['parameter']]} "
                     f"({nw[0]['ST']:.3f}) | {'yes' if same else 'NO'} |")
    summary["link_vs_network"] = cmp_out
    lines.append("")

    # --- threshold robustness ----------------------------------------------
    lines.append("## Threshold robustness of the ranking\n")
    lines.append("| scenario | output | " +
                 " | ".join(f"R_min={t}" for t in TAGS.values()) + " |")
    lines.append("|---" * (2 + len(TAGS)) + "|")
    thr = {}
    for scen in C.SCENARIOS:
        thr[scen] = {}
        for name, _, _ in C.OUTPUTS_THRESHOLDED:
            cells, tops = [], []
            for r_min, tag in TAGS.items():
                t = table(rows, scen, f"{name}@{tag}")
                if t:
                    cells.append(f"{SYM[t[0]['parameter']]} {t[0]['ST']:.2f}")
                    tops.append(t[0]["parameter"])
                else:
                    cells.append("--")
            if tops:
                thr[scen][name] = dict(tops=tops, stable=len(set(tops)) == 1)
                lines.append(f"| {scen} | `{name}` | " + " | ".join(cells) + " |")
    summary["threshold_robustness"] = thr
    lines.append("")

    # --- convergence --------------------------------------------------------
    if conv:
        lines.append("## Convergence\n")
        lines.append("Largest absolute change in ST between the two final base "
                     "sample sizes, per scenario and output.\n")
        lines.append("| scenario | output | max |dST| (N=2048 -> 4096) | "
                     "ranking stable? |")
        lines.append("|---|---|---|---|")
        cv = {}
        for scen in C.SCENARIOS:
            cv[scen] = {}
            for o in outs:
                a = {r["parameter"]: r["ST"] for r in conv
                     if r["scenario"] == scen and r["output"] == o
                     and r["n_base"] == 2048 and not r["degenerate"]}
                b = {r["parameter"]: r["ST"] for r in conv
                     if r["scenario"] == scen and r["output"] == o
                     and r["n_base"] == 4096 and not r["degenerate"]}
                shared = set(a) & set(b)
                if not shared:
                    continue
                delta = max(abs(a[p] - b[p]) for p in shared)
                ra = sorted(shared, key=lambda p: -a[p])
                rb = sorted(shared, key=lambda p: -b[p])
                stable = ra[:3] == rb[:3]
                cv[scen][o] = dict(max_abs_delta=delta, top3_stable=stable)
                lines.append(f"| {scen} | `{o}` | {delta:.4f} | "
                             f"{'yes' if stable else 'NO'} |")
        summary["convergence"] = cv
        worst = max((v["max_abs_delta"] for s in cv.values()
                     for v in s.values()), default=float("nan"))
        lines.append(f"\nWorst case across all outputs and scenarios: "
                     f"|dST| = {worst:.4f}.")
        summary["convergence_worst_delta"] = worst
        lines.append("")

    # --- round-3 additions --------------------------------------------------
    import summarize_round3
    summarize_round3.append_all(lines, summary)

    (C.RESULTS / "summary.md").write_text("\n".join(lines))
    (C.RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {C.RESULTS / 'summary.md'} and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
