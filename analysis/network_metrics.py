"""Network-level metrics on the usable graph.

Given the per-edge secret-key rates of one parameter sample and an operational
threshold, compute the link and network outputs of manuscript Table 3.

Two estimators are approximate by design, for tractability at 3102 nodes:

* Global efficiency is averaged over a fixed set of source nodes rather than
  all nodes.
* Median bottleneck capacity is taken over pairs from that same fixed source
  set rather than all reachable pairs.

Both use source sets fixed per scenario (topology.load_topology), so the
approximation is a deterministic function of the parameter sample. It shifts
the estimate but does not inject sampling noise into the Sobol decomposition,
which is what would actually corrupt the variance attribution.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree


def _symmetric_csr(n, u, v, w) -> csr_matrix:
    """Undirected weighted adjacency in CSR form."""
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    vals = np.concatenate([w, w])
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def global_efficiency(n, u, v, length_km, sources) -> float:
    """Mean inverse shortest-path length over ordered pairs (source, target).

    Path length is physical distance in km along usable links. Unreachable
    pairs contribute zero, per manuscript Table 3.
    """
    if len(u) == 0:
        return 0.0
    graph = _symmetric_csr(n, u, v, length_km)
    dist = dijkstra(graph, directed=False, indices=sources)
    with np.errstate(divide="ignore"):
        inv = np.where(np.isfinite(dist) & (dist > 0), 1.0 / dist, 0.0)
    # Exclude the (source, source) diagonal from the pair count.
    n_pairs = len(sources) * (n - 1)
    return float(inv.sum() / n_pairs) if n_pairs else 0.0


def median_bottleneck(n, u, v, capacity, sources) -> float:
    """Median maximum-bottleneck capacity over (source, target) pairs.

    The maximum-bottleneck path between two nodes is the path maximising its
    minimum edge capacity, and a standard result places that path on the
    maximum spanning tree. Rather than traversing the tree once per source,
    this adds tree edges in order of decreasing capacity under a union-find:
    when an edge of capacity c first joins components A and B, every pair with
    one endpoint in each has bottleneck exactly c, because every edge already
    added has capacity at least c.

    That yields the exact pair distribution as (capacity, count) pairs without
    ever materialising it, from which the weighted median is read off. Pairs in
    different components of the usable graph are never joined and so are
    correctly excluded.
    """
    if len(u) == 0:
        return 0.0

    # scipy builds minimum spanning trees; negate to obtain the maximum one.
    mst = minimum_spanning_tree(_symmetric_csr(n, u, v, -capacity)).tocoo()
    tw = -mst.data
    order = np.argsort(tw)[::-1]          # decreasing capacity
    tu, tv = mst.row[order], mst.col[order]
    tw = tw[order]

    parent = np.arange(n)
    size = np.ones(n, dtype=np.int64)         # nodes per component
    n_src = np.zeros(n, dtype=np.int64)       # tracked sources per component
    n_src[np.asarray(sources, dtype=int)] = 1

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    caps, counts = [], []
    for a, b, c in zip(tu, tv, tw):
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            continue
        pairs = int(n_src[ra]) * int(size[rb]) + int(n_src[rb]) * int(size[ra])
        if pairs:
            caps.append(float(c))
            counts.append(pairs)
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        n_src[ra] += n_src[rb]

    if not caps:
        return 0.0

    caps = np.asarray(caps)
    counts = np.asarray(counts, dtype=np.int64)
    idx = np.argsort(caps)
    caps, counts = caps[idx], counts[idx]
    cum = np.cumsum(counts)
    total = int(cum[-1])

    # Standard median convention, matching numpy: for an odd number of pairs
    # take the middle order statistic, for an even number average the two
    # straddling it. Taking the lower one instead is a defensible order
    # statistic but disagrees with np.median whenever the two middle values
    # differ, which is exactly what a reference check against a brute-force
    # implementation catches.
    def value_at(rank: int) -> float:
        return float(caps[int(np.searchsorted(cum, rank, side="left"))])

    if total % 2 == 1:
        return value_at((total + 1) // 2)
    return 0.5 * (value_at(total // 2) + value_at(total // 2 + 1))


def evaluate(topo, skr_bps: np.ndarray, r_min: float, want_bottleneck=True) -> dict:
    """All threshold-dependent outputs for one parameter sample."""
    n = topo.n_nodes
    usable = skr_bps >= r_min
    u, v = topo.edge_u[usable], topo.edge_v[usable]
    w_km = topo.edge_km[usable]
    cap = skr_bps[usable]

    res = {
        "usable_frac": float(usable.mean()) if topo.n_edges else 0.0,
        "reach_km": float(w_km.max()) if len(w_km) else 0.0,
    }

    if len(u) == 0:
        res.update(lcc_frac=1.0 / n, reachable_pairs=0.0, isolated_frac=1.0,
                   global_eff=0.0, bottleneck_log10=0.0)
        return res

    graph = _symmetric_csr(n, u, v, w_km)
    n_comp, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=n_comp)

    res["lcc_frac"] = float(sizes.max() / n)
    total_pairs = n * (n - 1) / 2.0
    res["reachable_pairs"] = float((sizes * (sizes - 1) / 2.0).sum() / total_pairs)

    degree = np.zeros(n, dtype=int)
    np.add.at(degree, u, 1)
    np.add.at(degree, v, 1)
    res["isolated_frac"] = float((degree == 0).mean())

    res["global_eff"] = global_efficiency(n, u, v, w_km, topo.eff_sources)

    if want_bottleneck:
        med = median_bottleneck(n, u, v, cap, topo.bott_sources)
        res["bottleneck_log10"] = float(np.log10(1.0 + med))
    else:
        res["bottleneck_log10"] = np.nan

    return res
