"""Geographical scenario loading and candidate-edge construction."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from config import MAX_EDGE_KM, N_SOURCES_BOTTLENECK, N_SOURCES_EFFICIENCY, SCENARIOS, SEED

EARTH_RADIUS_KM = 6371.0088


@dataclass
class Topology:
    """A scenario reduced to what the sensitivity analysis actually needs."""

    key: str
    label: str
    names: list[str]
    lat: np.ndarray
    lon: np.ndarray
    edge_u: np.ndarray      # candidate edge endpoints, u < v
    edge_v: np.ndarray
    edge_km: np.ndarray     # candidate edge lengths [km]
    eff_sources: np.ndarray    # fixed source nodes for the efficiency estimator
    bott_sources: np.ndarray   # fixed source nodes for the bottleneck estimator

    @property
    def n_nodes(self) -> int:
        return len(self.names)

    @property
    def n_edges(self) -> int:
        return len(self.edge_km)

    def summary(self) -> dict:
        return {
            "scenario": self.key,
            "label": self.label,
            "nodes": self.n_nodes,
            "candidate_edges": self.n_edges,
            "max_edge_km": MAX_EDGE_KM,
            "median_edge_km": float(np.median(self.edge_km)),
            "mean_edge_km": float(self.edge_km.mean()),
            "max_edge_length_km": float(self.edge_km.max()),
            "min_edge_length_km": float(self.edge_km.min()),
            "mean_degree": float(2.0 * self.n_edges / self.n_nodes),
        }


def _read_csv(path) -> tuple[list[str], np.ndarray, np.ndarray]:
    names, lat, lon = [], [], []
    with open(path, encoding="utf-8-sig") as fh:
        reader = csv.reader(fh, delimiter=";")
        next(reader)  # header: Poblacion;Latitud;Longitud
        for row in reader:
            if len(row) < 3 or not row[1] or not row[2]:
                continue
            names.append(row[0])
            lat.append(float(row[1]))
            lon.append(float(row[2]))
    return names, np.asarray(lat), np.asarray(lon)


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Full great-circle distance matrix in km."""
    la = np.radians(lat)[:, None]
    lo = np.radians(lon)[:, None]
    dla = la - la.T
    dlo = lo - lo.T
    a = np.sin(dla / 2.0) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlo / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_topology(key: str, max_edge_km: float = MAX_EDGE_KM) -> Topology:
    """Build a Topology, verifying the node count against the manuscript."""
    spec = SCENARIOS[key]
    names, lat, lon = _read_csv(spec["csv"])
    if len(names) != spec["expected_nodes"]:
        raise ValueError(
            f"{key}: read {len(names)} nodes, manuscript states "
            f"{spec['expected_nodes']}"
        )

    dist = haversine_km(lat, lon)
    iu, iv = np.triu_indices(len(names), k=1)
    keep = dist[iu, iv] <= max_edge_km
    edge_u, edge_v = iu[keep], iv[keep]
    edge_km = dist[edge_u, edge_v]

    # Fixed source sets, drawn once per scenario. Holding these constant across
    # every Sobol sample is what keeps estimator noise out of the variance
    # decomposition -- see config.N_SOURCES_EFFICIENCY. The offset comes from
    # the scenario record, not from hash(key): built-in string hashing is
    # randomised per interpreter, so a hash-derived seed differs between runs
    # and between multiprocessing workers within a single run.
    rng = np.random.default_rng(SEED + spec["seed_offset"])
    n = len(names)
    eff_sources = np.sort(rng.choice(n, size=min(N_SOURCES_EFFICIENCY, n),
                                     replace=False))
    bott_sources = np.sort(rng.choice(n, size=min(N_SOURCES_BOTTLENECK, n),
                                      replace=False))

    return Topology(
        key=key,
        label=spec["label"],
        names=names,
        lat=lat,
        lon=lon,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_km=edge_km,
        eff_sources=eff_sources,
        bott_sources=bott_sources,
    )


if __name__ == "__main__":
    import json

    for k in SCENARIOS:
        print(json.dumps(load_topology(k).summary(), indent=2))
