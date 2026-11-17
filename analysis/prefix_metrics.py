"""Connectivity outputs as functions of the number of usable links.

Every link of a sample receives the same seven parameter values, and the key
rate falls monotonically with link length. The usable graph of a sample is
therefore always "the k shortest candidate links" for some k, and the
structural outputs (largest-component fraction, reachable-pair fraction,
isolated-node fraction, usable-edge fraction, operational reach) are fixed
functions of k that depend on geography alone.

This module tabulates those functions once per topology with an incremental
union-find, so that scans over many thresholds or many small designs cost one
link-model call per sample instead of one graph traversal per (sample,
threshold). It is an exact re-expression of `network_metrics.evaluate` for
those outputs, not an approximation: `test_metrics.py` checks it against
`evaluate` sample by sample, and `lookup` refuses to use the table if the
usable set of a sample is not a prefix of the length-sorted edge list.

It does NOT replace `evaluate` in the principal run or the robustness suite.
It does not provide global efficiency or bottleneck capacity.

The same reduction is a limitation of the study design (homogeneous parameters
across links): see the threshold-scan and bottleneck analyses in this folder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PrefixCurves:
    order: np.ndarray        # edge indices sorted by ascending length
    km: np.ndarray           # edge lengths in that order
    lcc_frac: np.ndarray     # index k = number of usable (shortest) edges
    reachable_pairs: np.ndarray
    isolated_frac: np.ndarray
    n_nodes: int

    @property
    def n_edges(self) -> int:
        return len(self.km)


def build(topo, edge_km=None) -> PrefixCurves:
    """Tabulate the structural outputs for every prefix of the sorted edges."""
    km_all = topo.edge_km if edge_km is None else edge_km
    n, m = topo.n_nodes, len(km_all)
    order = np.argsort(km_all, kind="stable")
    eu, ev = topo.edge_u[order], topo.edge_v[order]

    parent = list(range(n))
    size = [1] * n
    degree = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    lcc = np.empty(m + 1)
    pairs = np.empty(m + 1)
    iso = np.empty(m + 1)
    total_pairs = n * (n - 1) / 2.0
    largest, n_pairs, n_iso = 1, 0, n
    lcc[0], pairs[0], iso[0] = 1.0 / n, 0.0, 1.0
    for k, (a, b) in enumerate(zip(eu.tolist(), ev.tolist()), start=1):
        for node in (a, b):
            if degree[node] == 0:
                n_iso -= 1
            degree[node] += 1
        ra, rb = find(a), find(b)
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            n_pairs += size[ra] * size[rb]
            parent[rb] = ra
            size[ra] += size[rb]
            if size[ra] > largest:
                largest = size[ra]
        lcc[k] = largest / n
        pairs[k] = n_pairs / total_pairs
        iso[k] = n_iso / n
    return PrefixCurves(order=order, km=km_all[order], lcc_frac=lcc,
                        reachable_pairs=pairs, isolated_frac=iso, n_nodes=n)


def usable_count(skr_sorted: np.ndarray, r_min: float) -> int:
    """Number of usable links, given rates in ascending-length order.

    Raises if the usable set is not a prefix of the sorted list, which would
    mean the monotonicity the table relies on failed for this sample.
    """
    usable = skr_sorted >= r_min
    k = int(usable.sum())
    if k and not usable[:k].all():
        raise ArithmeticError("usable set is not a prefix of the sorted links")
    return k


def lookup(curves: PrefixCurves, k: int) -> dict:
    """Outputs of `network_metrics.evaluate` that depend on k alone.

    Matches evaluate() including its empty-graph conventions (k = 0).
    """
    m = curves.n_edges
    return {
        "usable_frac": k / m if m else 0.0,
        "reach_km": float(curves.km[k - 1]) if k else 0.0,
        "lcc_frac": float(curves.lcc_frac[k]),
        "reachable_pairs": float(curves.reachable_pairs[k]),
        "isolated_frac": float(curves.isolated_frac[k]),
    }
