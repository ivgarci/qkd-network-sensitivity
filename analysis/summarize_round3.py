"""Round-3 sections of results/summary.md.

Called by summarize_results.py; not run on its own. Each section reads one of
the artifacts added in the round-3 revision and is skipped, with a notice in
the summary, if that artifact is missing. Nothing is computed here beyond
formatting: every number comes from the file named in its section heading.
"""

from __future__ import annotations

import csv
import json

import config as C

TAGS = [f"{r:.0e}".replace("+0", "").replace("+", "") for r in C.R_MIN_SWEEP]
PRIMARY = f"{C.R_MIN_PRIMARY:.0e}".replace("+0", "").replace("+", "")
METRICS = [n for n, _, _ in C.OUTPUTS_THRESHOLDED]


def _json(name):
    p = C.RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def _csv(name):
    p = C.RESULTS / name
    return list(csv.DictReader(open(p))) if p.exists() else None


def _ci(v, lo, hi, nd=3):
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def bottleneck(lines, summary):
    d = _json("bottleneck_decomposition.json")
    lines.append("## Bottleneck capacity and the empty-graph sentinel "
                 "(`bottleneck_decomposition.json`)\n")
    if not d:
        lines.append("Not generated. Run `bottleneck_decomposition.py`.\n")
        return
    lines.append(
        "`bottleneck_log10` is stored as log10(1 + B) with B = 0 when no "
        "tracked source has a usable link. The table gives the fraction of "
        "samples taking that value, the share of the stored output's variance "
        "carried by the empty/non-empty indicator alone, and the interaction "
        "share 1 - sum S1 [95% bootstrap interval] under four codings of the "
        "empty case. Every coding is a choice; none replaces the stored "
        "result.\n")
    lines.append("| model | scenario | R_min | empty | indicator variance share "
                 "| stored | indicator | floor | linear |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    keep = {}
    for model, by_scen in d["models"].items():
        for scen, by_tag in by_scen.items():
            for tag, b in by_tag.items():
                cells = []
                for name in ("stored", "indicator", "floor", "linear"):
                    c = b["codings"][name]
                    cells.append("constant" if c["degenerate"] else _ci(
                        c["interaction_share_unclipped"],
                        *c["interaction_share_ci"]))
                lines.append(
                    f"| {model} | {scen} | {tag} | {100*b['frac_sentinel']:.2f}% "
                    f"| {b['variance_share_of_indicator']:.3f} | "
                    + " | ".join(cells) + " |")
                keep[f"{model}/{scen}/{tag}"] = {
                    "frac_sentinel": b["frac_sentinel"],
                    "variance_share_of_indicator": b["variance_share_of_indicator"],
                    "interaction_share": {
                        n: (None if b["codings"][n]["degenerate"] else
                            b["codings"][n]["interaction_share_unclipped"])
                        for n in b["codings"]}}
    lines.append("")
    full = d["models"].get(f"full_N{C.N_BASE}", {})
    for scen, by_tag in full.items():
        b = by_tag[PRIMARY]
        lines.append(f"Leading indices at {PRIMARY}, {scen} (S1 / ST):")
        lines.append("")
        lines.append("| coding | eta_det | e_d | L0 | mu |")
        lines.append("|---|---|---|---|---|")
        for name in ("stored", "indicator", "floor", "linear"):
            p = b["codings"][name]["parameters"]
            lines.append(f"| {name} | " + " | ".join(
                f"{p[k]['S1']:.3f} / {p[k]['ST']:.3f}"
                for k in ("eta_det", "e_d", "L0", "mu")) + " |")
        lines.append("")
    summary["bottleneck_decomposition"] = keep


def interaction_shares(lines, summary):
    rows = _csv("interaction_shares.csv")
    s = _json("interaction_shares_summary.json")
    lines.append("## Interaction shares at all thresholds "
                 "(`interaction_shares.csv`)\n")
    if not rows or not s:
        lines.append("Not generated. Run `interaction_shares.py`.\n")
        return
    lines.append(
        "1 - sum S1 with 95% percentile bootstrap intervals (1000 resamples of "
        "the base rows). `full 4096` is the principal run. The two N = 1024 "
        "columns use the same 1024 base points and the same resamples, so "
        "`full - surrogate` is a paired difference; the surrogate was not run "
        "at N = 4096. Intervals are of the unclipped value.\n")
    idx = {(r["scenario"], r["model"], r["n_base"], r["output"], r["coding"]): r
           for r in rows}

    def cell(scen, model, n, out):
        r = idx.get((scen, model, str(n), out, "stored"))
        if not r or r["degenerate"] == "True":
            return "constant"
        return _ci(float(r["share_raw"]), float(r["share_ci_lo"]),
                   float(r["share_ci_hi"]))

    outputs = ["median_skr_log10"] + [f"{m}@{t}" for t in TAGS for m in METRICS]
    for scen in C.SCENARIOS:
        if (scen, "full", str(C.N_BASE), "median_skr_log10", "stored") not in idx:
            continue
        diff = s["full_minus_surrogate_N1024"].get(scen, {})
        lines.append(f"### {C.SCENARIOS[scen]['label']}\n")
        lines.append("| output | full 4096 | full 1024 | surrogate 1024 | "
                     "full - surrogate (1024) |")
        lines.append("|---|---|---|---|---|")
        for o in outputs:
            dd = diff.get(o)
            dtxt = ("n/a" if not dd else
                    f"{dd['difference']:+.3f} [{dd['ci'][0]:+.3f}, {dd['ci'][1]:+.3f}]"
                    + ("" if dd["resolved"] else " ns"))
            lines.append(f"| `{o}` | {cell(scen, 'full', C.N_BASE, o)} | "
                         f"{cell(scen, 'full', 1024, o)} | "
                         f"{cell(scen, 'surrogate', 1024, o)} | {dtxt} |")
        lines.append("\n`ns`: interval of the difference includes zero.\n")
    lines.append("### Is the share monotone from link to network outputs?\n")
    lines.append("Two output orders: the one used in the manuscript and "
                 "Figure 9 (chosen after seeing the primary-threshold shares) "
                 "and the order of manuscript Table 3 (fixed beforehand). A "
                 "fall is `resolved` when the paired bootstrap interval of the "
                 "step excludes zero.\n")
    lines.append("| scenario | R_min | model | manuscript order | Table 3 order |")
    lines.append("|---|---|---|---|---|")
    keep = {}
    for scen, by_tag in s["ordering"].items():
        for tag, by_model in by_tag.items():
            for model, o in by_model.items():
                cells = []
                for name in ("manuscript_fig9", "table3"):
                    v = o[name]
                    if not v["evaluable"]:
                        cells.append("not evaluable (constant output)")
                        continue
                    mono = v["monotone_nondecreasing_in_point_estimates"]
                    cells.append(
                        "monotone" if mono else
                        f"NOT monotone: {v['n_steps_that_fall']} of "
                        f"{v['n_steps']} steps fall, {v['n_falls_resolved']} resolved")
                    keep[f"{scen}/{tag}/{model}/{name}"] = mono
                lines.append(f"| {scen} | {tag} | {model} | " + " | ".join(cells) + " |")
    lines.append("")
    summary["interaction_share_monotone"] = keep


def generation_rankings(lines, summary):
    g = _json("generation_rankings_summary.json")
    lines.append("## Within-generation rankings "
                 "(`generation_rankings.csv`)\n")
    if not g:
        lines.append("Not generated. Run `generation_rankings.py`.\n")
        return
    t = g["totals_unresolved_of_nondegenerate"]
    lines.append(
        f"Total-order ranking inside each detector generation (N = 1024, 200 "
        f"resamples, primary threshold). `resolved` applies the rule of the "
        f"section above: {g['resolution_rule']}. Unresolved scenario-outputs "
        f"over all thresholds: InGaAs {t['ingaas'][0]} of {t['ingaas'][1]}, "
        f"SNSPD {t['snspd'][0]} of {t['snspd'][1]}. `=` marks adjacent ranks "
        f"whose intervals overlap.\n")
    lines.append("| generation | scenario | output | order (ST+-conf) | "
                 "eta_det rank | resolved |")
    lines.append("|---|---|---|---|---|---|")
    rows = _csv("generation_rankings.csv") or []
    by = {}
    for r in rows:
        if r["degenerate"] != "True":
            by.setdefault((r["generation"], r["scenario"], r["output"]), []).append(r)
    for gen in ("ingaas", "snspd"):
        for scen in C.SCENARIOS:
            for o, v in g[gen][scen]["primary_threshold"].items():
                tab = sorted(by[(gen, scen, o)], key=lambda r: int(r["rank"]))
                parts = []
                for r in tab[:5]:
                    parts.append(f"{r['parameter']} {float(r['ST']):.2f}+-"
                                 f"{float(r['ST_conf']):.2f}")
                    parts.append("=" if r["overlaps_next_rank"] == "True" else ">")
                lines.append(f"| {gen} | {scen} | `{o}` | "
                             + " ".join(parts[:-1]) + f" | {v['eta_det_rank']} | "
                             f"{'yes' if v['ranking_resolved'] else 'NO'} |")
    lines.append("")
    summary["generation_rankings"] = {
        "unresolved_of_nondegenerate": t,
        "eta_det_rank_primary": {
            f"{gen}/{scen}": {o: v["eta_det_rank"] for o, v in
                              g[gen][scen]["primary_threshold"].items()}
            for gen in ("ingaas", "snspd") for scen in C.SCENARIOS}}


def generation_stats(lines, summary):
    g = _json("generation_stats.json")
    lines.append("## Largest-component fraction by generation and threshold "
                 "(`generation_stats.json`)\n")
    if not g or "by_threshold" not in g:
        lines.append("Not generated. Run `generation_stats.py`.\n")
        return
    d = g["design"]
    lines.append(
        f"Each cell is the {d['n_samples']} rows of a Saltelli cross design "
        f"({d['n_base']} base blocks x {d['rows_per_block']} rows), not "
        f"independent draws; intervals are a cluster bootstrap over the blocks "
        f"({d['bootstrap_resamples']} resamples, 95%). `intermediate` is a "
        f"gap-filling case (eta_det 0.30-0.60), not a device class.\n")
    keep = {}
    for variant, by_gen in g["by_threshold"].items():
        lines.append(f"### Variant: {variant}\n")
        lines.append("| generation | scenario | R_min | mean | median | "
                     "q05 / q25 / q75 / q95 | > 0.95 | < 0.05 | no usable link |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for gen, by_scen in by_gen.items():
            for scen, by_tag in by_scen.items():
                for tag, s in by_tag.items():
                    lines.append(
                        f"| {gen} | {scen} | {tag} | "
                        f"{_ci(s['lcc_mean'], *s['lcc_mean_ci'])} | "
                        f"{_ci(s['lcc_median'], *s['lcc_median_ci'])} | "
                        f"{s['lcc_q05']:.3f} / {s['lcc_q25']:.3f} / "
                        f"{s['lcc_q75']:.3f} / {s['lcc_q95']:.3f} | "
                        f"{_ci(s['lcc_frac_above_095'], *s['lcc_frac_above_095_ci'])} | "
                        f"{_ci(s['lcc_frac_below_005'], *s['lcc_frac_below_005_ci'])} | "
                        f"{s['frac_no_usable_link']:.3f} |")
                    keep[f"{variant}/{gen}/{scen}/{tag}"] = {
                        "lcc_mean": s["lcc_mean"], "lcc_mean_ci": s["lcc_mean_ci"],
                        "lcc_median": s["lcc_median"]}
        lines.append("")
    summary["generation_lcc_by_threshold"] = keep


def threshold_scan(lines, summary):
    rows = _csv("threshold_scan.csv")
    t = _json("threshold_scan.json")
    lines.append("## Threshold scan (`threshold_scan.csv`)\n")
    if not rows or not t:
        lines.append("Not generated. Run `threshold_scan.py`.\n")
        return
    lines.append("Statistics of the largest-component fraction over the "
                 "principal design as R_min varies.\n")
    lines.append("| scenario | R_min | mean | sd | median | > 0.95 | < 0.05 | "
                 "no usable link | every link usable |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        f = {k: float(v) for k, v in r.items()
             if k not in ("scenario", "in_stated_scan_range")}
        lines.append(
            f"| {r['scenario']} | {f['r_min_bps']:.1e} | {f['lcc_mean']:.3f} | "
            f"{f['lcc_sd']:.3f} | {f['lcc_median']:.3f} | "
            f"{f['frac_lcc_above_095']:.3f} | {f['frac_lcc_below_005']:.3f} | "
            f"{f['frac_no_usable_link']:.4f} | {f['frac_all_links_usable']:.4f} |")
    lines.append("")
    for scen, s in t["scenarios"].items():
        lo, hi = s["lcc_sd_within_10pct_of_peak_bps"]
        lines.append(
            f"- {scen}: sd peaks at {s['lcc_sd_peak_r_min_bps']:.0e} bit/s "
            f"({s['lcc_sd_peak']:.3f}); within 10% of the peak from {lo:.0e} to "
            f"{hi:.0e}. Largest difference from the eight values in the "
            f"`config.py` comment: {s['max_abs_diff_from_config_comment']:.3f}.")
    lines.append("")
    summary["threshold_scan"] = t["scenarios"]


def claim_support(lines, summary):
    c = _json("claim_support.json")
    lines.append("## Claim support (`claim_support.json`, `rank_orders.csv`)\n")
    if not c:
        lines.append("Not generated. Run `claim_support.py`.\n")
        return
    keep = {}

    lines.append("### Pointwise effect of dark counts on the key rate\n")
    lines.append("Relative reduction in rate when p_dc goes from 1e-8 to 1e-5 "
                 "with the other six inputs fixed.\n")
    lines.append("| length | worst corner: rate low -> high p_dc | reduction | "
                 "range centre | design median | design q95 | design max | "
                 "rows losing > 5% |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for km, d in c["dark_counts"]["by_length_km"].items():
        w, a = d["worst_corner"], d["across_principal_design"]
        lines.append(
            f"| {km} km | {w['rate_p_dc_low_bps']:.3e} -> "
            f"{w['rate_p_dc_high_bps']:.3e} bit/s | "
            f"{100*w['relative_reduction']:.1f}% | "
            f"{100*d['range_centre']['relative_reduction']:.2f}% | "
            f"{100*a['median']:.2f}% | {100*a['q95']:.2f}% | {100*a['max']:.1f}% | "
            f"{100*a['frac_samples_reduction_above_5pct']:.1f}% |")
    w = c["dark_counts"]["by_length_km"]["45"]["worst_corner"]
    lines.append(f"\nWorst corner: {w['parameters']}; eta*mu there is "
                 f"{w['eta_times_mu']:.2e}.\n")
    keep["dark_count_worst_reduction_45km"] = w["relative_reduction"]

    lines.append("### Rate bounds\n")
    lines.append("| scenario | longest link | every link usable in the whole "
                 "parameter box below | R_pulse range on the longest link | "
                 "no link usable anywhere above |")
    lines.append("|---|---|---|---|---|")
    for scen, r in c["rate_bounds"]["by_scenario"].items():
        lo, hi = r["R_pulse_range_on_longest_link"]
        lines.append(
            f"| {scen} | {r['longest_candidate_link_km']:.2f} km | "
            f"{r['every_link_usable_in_whole_box_below_bps']:.4g} bit/s | "
            f"{lo:.3g} to {hi:.3g} | "
            f"{r['no_link_usable_anywhere_in_box_above_bps']:.4g} bit/s |")
    lines.append("")
    keep["every_link_usable_below_bps"] = {
        s: r["every_link_usable_in_whole_box_below_bps"]
        for s, r in c["rate_bounds"]["by_scenario"].items()}

    lines.append("### Outputs exceeding the 0.02 convergence criterion\n")
    lines.append("Largest |dST| between N = 2048 and N = 4096, all 22 outputs.\n")
    lines.append("| scenario | output | max abs dST | ranking resolved at 4096 |")
    lines.append("|---|---|---|---|")
    for scen, s in c["convergence"]["scenarios"].items():
        for o, v in s["outputs_exceeding_criterion"].items():
            lines.append(f"| {scen} | `{o}` | {v['max_abs_dST']:.4f} | "
                         f"{'yes' if v['ranking_resolved_at_4096'] else 'NO'} |")
    lines.append("")
    for scen, s in c["convergence"]["scenarios"].items():
        lines.append(f"- {scen}: leading three not stable from N = 1024 on "
                     f"{s['outputs_whose_top3_is_not_stable_from_N1024'] or 'no output'}.")
    lines.append("")
    keep["outputs_exceeding_convergence_criterion"] = {
        s: list(v["outputs_exceeding_criterion"])
        for s, v in c["convergence"]["scenarios"].items()}

    lines.append("### Full rank orders, principal run and route factor 1.3\n")
    lines.append("All seven inputs by total-order index. `=` joins adjacent "
                 "ranks whose intervals overlap (a tie); `>` separates ranks "
                 "whose intervals do not. `(u)` marks an output whose ranking "
                 "is unresolved by the 25% rule.\n")
    for design, by_scen in c["rank_orders"].items():
        lines.append(f"**{design}**\n")
        lines.append("| scenario | output | order |")
        lines.append("|---|---|---|")
        for scen, by_out in by_scen.items():
            for o, t in by_out.items():
                parts = []
                for p, tie in zip(t["order"], t["tie_with_next_rank"]):
                    parts += [p, "=" if tie else ">"]
                lines.append(f"| {scen} | `{o}`"
                             f"{'' if t['ranking_resolved'] else ' (u)'} | "
                             + " ".join(parts[:-1]) + " |")
        lines.append("")

    z = c["zero_length_links"]
    lines.append("### Zero-length candidate links\n")
    for scen, cell in z.items():
        if not cell["n_zero_length_links"]:
            lines.append(f"- {scen}: none.")
            continue
        worst = max((o["max_abs_ST_difference"], f"{n}@{t}",
                     o["stored_ST_conf_of_that_parameter"])
                    for t, b in cell["by_threshold"].items()
                    for n, o in b["outputs"].items())
        lines.append(
            f"- {scen}: {cell['n_zero_length_links']} of "
            f"{cell['n_candidate_links']:,} ({cell['pairs']}). Removing them "
            f"changes no total-order index of S_LCC, rho_P or rho_I by more "
            f"than {worst[0]:.4f} (`{worst[1]}`, whose half-width is "
            f"{worst[2]:.3f}). Samples in which they are the only usable "
            f"links: " + ", ".join(
                f"{100*b['frac_samples_where_only_zero_length_links_are_usable']:.2f}% at {t}"
                for t, b in cell["by_threshold"].items()) + ".")
        keep["zero_length_max_abs_dST"] = worst[0]
    lines.append("")

    sp = c.get("input_spans_db")
    if sp:
        lines.append("### Input ranges as link-budget spans\n")
        a = sp["alpha_db_by_scenario"]
        lines.append(
            f"eta_det {sp['eta_det_db']:.2f} dB; L0 {sp['L0_db']:.2f} dB; alpha "
            + "; ".join(f"{v['at_median_link']:.2f} dB at the {s} median link "
                        f"({v['median_link_km']:.1f} km), {v['at_28_km']:.2f} dB "
                        f"at 28 km" for s, v in a.items()) + ".\n")
    summary["claim_support"] = keep


def count_rates(lines, summary):
    c = _json("count_rates.json")
    lines.append("## Detector count rates (`count_rates.json`)\n")
    if not c:
        lines.append("Not generated. Run `count_rates.py`.\n")
        return
    lines.append(f"Definition: {c['definition']}\n")
    lines.append(
        f"Largest key/detection ratio R/D over the sampled box and link "
        f"lengths {c['length_range_km'][0]:g} to {c['length_range_km'][1]:g} "
        f"km: **{c['max_key_over_detection']:.4f}**.\n")
    lines.append("| search | max R/D | at |")
    lines.append("|---|---|---|")
    for k, v in c["searches"].items():
        at = ", ".join(f"{n}={x:.4g}" for n, x in v["at"].items())
        lines.append(f"| {k} | {v['max_key_over_detection']:.4f} | {at} |")
    lines.append("")
    lines.append("| R_min (bit/s) | minimum detection rate of a usable link (1/s) |")
    lines.append("|---|---|")
    for t, v in c["min_detection_rate_at_threshold_per_s"].items():
        lines.append(f"| {t} | {v:.3e} |")
    lines.append("")
    lines.append("At the centre of every range (p_dc at the centre of its log10 "
                 "range):\n")
    lines.append("| box | length | detection rate (1/s) | key rate (bit/s) | R/D |")
    lines.append("|---|---|---|---|---|")
    for box, by_d in c["range_centre"].items():
        for d, v in by_d.items():
            lines.append(f"| {box} | {d} km | {v['detection_rate_per_s']:.3e} | "
                         f"{v['key_rate_bps']:.3e} | "
                         f"{v['key_over_detection']:.3f} |")
    lines.append("")
    summary["count_rates"] = {
        "max_key_over_detection": c["max_key_over_detection"],
        "min_detection_rate_at_threshold_per_s":
            c["min_detection_rate_at_threshold_per_s"],
        "range_centre_principal_28km_detection_rate_per_s":
            c["range_centre"]["principal"]["28"]["detection_rate_per_s"]}


def append_all(lines, summary):
    lines.append("---\n")
    lines.append("# Round-3 additions\n")
    for section in (bottleneck, interaction_shares, generation_rankings,
                    generation_stats, threshold_scan, claim_support,
                    count_rates):
        section(lines, summary)
