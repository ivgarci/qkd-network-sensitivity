"""Correctness tests for the network metrics.

Run:  .venv/bin/python analysis/test_metrics.py

The link model is validated against published results in validate.py. These
tests cover the other half of the pipeline: the graph metrics, and in
particular the maximum-bottleneck estimator, which uses a union-find
formulation rather than the obvious tree traversal and so needs checking
against a definition-level reference implementation.

It guards against an implementation bug that yields numerically plausible
but wrong results.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import networkx as nx

from network_metrics import evaluate, global_efficiency, median_bottleneck

RESULTS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def brute_force_bottleneck(n, u, v, cap, sources) -> float:
    """Median max-bottleneck over (source, target) pairs, by definition.

    Enumerates every simple path with networkx and maximises the minimum edge
    capacity. Correct but exponential, so only usable on tiny graphs -- which
    is exactly what a reference implementation is for.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, c in zip(u, v, cap):
        G.add_edge(int(a), int(b), cap=float(c))

    vals = []
    for s in sources:
        for t in range(n):
            if t == s or not nx.has_path(G, s, t):
                continue
            best = max(
                min(G[p[i]][p[i + 1]]["cap"] for i in range(len(p) - 1))
                for p in nx.all_simple_paths(G, s, t)
            )
            vals.append(best)
    return float(np.median(vals)) if vals else 0.0


def brute_force_efficiency(n, u, v, w, sources) -> float:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, c in zip(u, v, w):
        G.add_edge(int(a), int(b), w=float(c))
    total = 0.0
    for s in sources:
        lengths = nx.single_source_dijkstra_path_length(G, s, weight="w")
        for t, d in lengths.items():
            if t != s and d > 0:
                total += 1.0 / d
    return total / (len(sources) * (n - 1))


def test_bottleneck_random_graphs() -> None:
    rng = np.random.default_rng(11)
    worst = 0.0
    for trial in range(40):
        n = int(rng.integers(5, 10))
        edges = [
            (a, b) for a, b in itertools.combinations(range(n), 2)
            if rng.random() < 0.45
        ]
        if not edges:
            continue
        u = np.array([e[0] for e in edges])
        v = np.array([e[1] for e in edges])
        cap = rng.uniform(1.0, 100.0, size=len(edges))
        srcs = np.array(sorted(rng.choice(n, size=min(3, n), replace=False)))
        got = median_bottleneck(n, u, v, cap, srcs)
        want = brute_force_bottleneck(n, u, v, cap, srcs)
        worst = max(worst, abs(got - want))
    check("median_bottleneck matches brute force on 40 random graphs",
          worst < 1e-9, f"max |diff| = {worst:.2e}")


def test_bottleneck_chain() -> None:
    # A chain 0-1-2-3 with capacities 10, 3, 7. From node 0 the bottleneck to
    # node 3 is 3; to node 2 it is 3; to node 1 it is 10. Median of {10,3,3}=3.
    n = 4
    u = np.array([0, 1, 2]); v = np.array([1, 2, 3])
    cap = np.array([10.0, 3.0, 7.0])
    got = median_bottleneck(n, u, v, cap, np.array([0]))
    check("median_bottleneck on a hand-computed chain", abs(got - 3.0) < 1e-12,
          f"got {got}")


def test_bottleneck_prefers_wider_detour() -> None:
    # Direct 0-2 has capacity 1; detour 0-1-2 has min capacity 5. The maximum
    # bottleneck path is the detour, so the answer must be 5, not 1. A naive
    # shortest-path implementation would return 1.
    n = 3
    u = np.array([0, 0, 1]); v = np.array([2, 1, 2])
    cap = np.array([1.0, 5.0, 5.0])
    got = median_bottleneck(n, u, v, cap, np.array([0]))
    check("max-bottleneck takes the wider detour, not the direct link",
          abs(got - 5.0) < 1e-12, f"got {got} (direct=1, detour=5)")


def test_efficiency_matches_networkx() -> None:
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(20):
        n = int(rng.integers(6, 14))
        edges = [
            (a, b) for a, b in itertools.combinations(range(n), 2)
            if rng.random() < 0.4
        ]
        if not edges:
            continue
        u = np.array([e[0] for e in edges]); v = np.array([e[1] for e in edges])
        w = rng.uniform(1.0, 50.0, size=len(edges))
        srcs = np.array(sorted(rng.choice(n, size=min(4, n), replace=False)))
        got = global_efficiency(n, u, v, w, srcs)
        want = brute_force_efficiency(n, u, v, w, srcs)
        worst = max(worst, abs(got - want))
    check("global_efficiency matches networkx Dijkstra", worst < 1e-9,
          f"max |diff| = {worst:.2e}")


def test_disconnected_handling() -> None:
    # Two isolated pairs plus one lone node: 5 nodes, components {0,1},{2,3},{4}
    n = 5
    u = np.array([0, 2]); v = np.array([1, 3])

    class T:
        n_nodes = n
        n_edges = 2
        edge_u = u
        edge_v = v
        edge_km = np.array([10.0, 10.0])
        eff_sources = np.array([0, 2, 4])
        bott_sources = np.array([0, 2, 4])

    res = evaluate(T, np.array([1e9, 1e9]), 1.0)
    ok = (
        abs(res["lcc_frac"] - 2 / 5) < 1e-12
        and abs(res["isolated_frac"] - 1 / 5) < 1e-12
        # reachable pairs: 1 + 1 out of C(5,2)=10
        and abs(res["reachable_pairs"] - 2 / 10) < 1e-12
    )
    check("disconnected graph: LCC, isolated and reachable-pair fractions", ok,
          f"lcc={res['lcc_frac']:.3f} iso={res['isolated_frac']:.3f} "
          f"pairs={res['reachable_pairs']:.3f}")


def test_threshold_monotonicity() -> None:
    # Raising the threshold can only remove links, so usable fraction and
    # largest-component fraction must be non-increasing in R_min.
    rng = np.random.default_rng(3)
    n = 30
    edges = [(a, b) for a, b in itertools.combinations(range(n), 2)
             if rng.random() < 0.15]
    u = np.array([e[0] for e in edges]); v = np.array([e[1] for e in edges])

    class T:
        n_nodes = n
        n_edges = len(edges)
        edge_u = u
        edge_v = v
        edge_km = rng.uniform(1, 40, size=len(edges))
        eff_sources = np.arange(0, n, 3)
        bott_sources = np.arange(0, n, 3)

    skr = rng.uniform(0, 1e7, size=len(edges))
    prev_u = prev_l = 1.1
    ok = True
    for r in (1e5, 1e6, 3e6, 5e6, 8e6):
        res = evaluate(T, skr, r)
        ok &= res["usable_frac"] <= prev_u + 1e-12
        ok &= res["lcc_frac"] <= prev_l + 1e-12
        prev_u, prev_l = res["usable_frac"], res["lcc_frac"]
    check("usable fraction and LCC are non-increasing in the threshold", ok)


def _worker_checksum(_):
    from topology import load_topology
    t = load_topology("cyl")
    return int(t.eff_sources.sum()), int(t.bott_sources.sum())


def test_source_sets_are_reproducible() -> None:
    """The fixed source sets must be identical in every process.

    They were originally seeded with `SEED + abs(hash(key)) % 10000`. Python
    randomises string hashing per interpreter, so that seed differed between
    runs and, worse, between spawned multiprocessing workers inside a single
    run -- meaning different samples of one Sobol design were measured with
    different source sets. That is exactly the estimator contamination the
    fixed source set exists to prevent, and it is invisible in the output.
    """
    from multiprocessing import Pool
    from topology import load_topology

    local = load_topology("cyl")
    here = (int(local.eff_sources.sum()), int(local.bott_sources.sum()))
    with Pool(4) as pool:
        remote = pool.map(_worker_checksum, range(4))
    ok = len(set(remote)) == 1 and remote[0] == here
    check("source sets identical across processes", ok,
          f"parent={here} workers={set(remote)}")


def test_prefix_curves_match_evaluate() -> None:
    """prefix_metrics must reproduce evaluate() exactly on the real topology.

    The threshold scan and the generation statistics read the structural
    outputs from a table indexed by the number of usable links. That is only
    legitimate if the table and the graph traversal agree sample by sample.
    """
    import config as C
    import prefix_metrics as PM
    from link_model import secret_key_rate
    from topology import load_topology

    rng = np.random.default_rng(C.SEED)
    for scen, n_samples in (("cyl", 300), ("spain", 12)):
        topo = load_topology(scen)
        curves = PM.build(topo)
        worst, n_checked = 0.0, 0
        for _ in range(n_samples):
            d = {}
            for name, (lo, hi), is_log, _, _ in C.PARAMETERS:
                x = rng.uniform(lo, hi)
                d[name] = 10.0 ** x if is_log else x
            skr = secret_key_rate(topo.edge_km, **d)
            for r_min in (2e4, 1e6, 3e6, 5e6, 1e7, 2e7):
                ref = evaluate(topo, skr, r_min, want_bottleneck=False)
                got = PM.lookup(curves, PM.usable_count(skr[curves.order], r_min))
                for key, val in got.items():
                    worst = max(worst, abs(val - ref[key]))
                n_checked += 1
        check(f"prefix table equals evaluate() on {scen}", worst < 1e-12,
              f"max |diff|={worst:.1e} over {n_checked} (sample, threshold) pairs")


def main() -> int:
    print("Network metric correctness tests\n")
    test_bottleneck_chain()
    test_bottleneck_prefers_wider_detour()
    test_bottleneck_random_graphs()
    test_efficiency_matches_networkx()
    test_disconnected_handling()
    test_threshold_monotonicity()
    test_source_sets_are_reproducible()
    test_prefix_curves_match_evaluate()
    passed = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
