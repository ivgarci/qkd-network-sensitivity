"""Rebuild the two municipality coordinate files used by the analysis.

The coordinate files are not distributed with this repository: their upstream
listing carries no stated licence (see data/README.md). This script downloads
that listing, checks it against the SHA-256 recorded below, and writes

    data/cyl_267.csv      Castilla y Leon, municipalities with >= 1000 inhabitants
    data/spain_3102.csv   mainland Spain, same criterion

in the exact format, row order and content the analysis was run on
(semicolon-separated, UTF-8 with BOM, CRLF line endings, header
`Poblacion;Latitud;Longitud` with accents). Row order matters: the estimator
source sets are drawn by row index.

Run from the repository root:  python data/fetch_data.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://www.businessintelligence.info/resources/assets/"
       "listado-longitud-latitud-municipios-espana.txt")
SHA256 = "8dfb8031f51bb12b87f669cd4eb9a725464fd7bb4764950573b29f04a93b41eb"
MIN_INHABITANTS = 1000
NOT_MAINLAND = {"Canarias", "Islas Baleares", "Ceuta y Melilla"}

DATA = Path(__file__).resolve().parent
OUTPUTS = {
    "cyl_267.csv": (lambda region: region == "Castilla León", 267),
    "spain_3102.csv": (lambda region: region not in NOT_MAINLAND, 3102),
}


def download() -> bytes:
    cache = DATA / "upstream_listing.txt"
    if cache.exists():
        raw = cache.read_bytes()
    else:
        req = urllib.request.Request(URL, headers={"User-Agent": "qkd-network-sensitivity/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
        cache.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        sys.exit(f"Upstream listing changed: SHA-256 {digest}, expected {SHA256}.\n"
                 f"Delete {cache} to re-download, or do not use it: the published "
                 "results were computed from the recorded version.")
    return raw


def main() -> None:
    lines = download().decode("utf-16").splitlines()
    # Two comment lines, then the header:
    # Comunidad, Provincia, Poblacion, Latitud, Longitud, Altitud, Habitantes, Hombres, Mujeres
    rows = [line.split("\t") for line in lines[3:] if line.strip()]
    for name, (keep, expected) in OUTPUTS.items():
        out = ["﻿Población;Latitud;Longitud"]
        for region, _prov, town, lat, lon, _alt, inhabitants, *_ in rows:
            if int(inhabitants) >= MIN_INHABITANTS and keep(region):
                # Decimal commas become points; the one comma inside a name
                # becomes a point, as in the files the analysis was run on.
                out.append(f"{town.replace(',', '.')};{lat.replace(',', '.')};{lon.replace(',', '.')}")
        if len(out) - 1 != expected:
            sys.exit(f"{name}: {len(out) - 1} rows, expected {expected}")
        (DATA / name).write_bytes(("\r\n".join(out) + "\r\n").encode("utf-8"))
        print(f"wrote data/{name} ({expected} municipalities)")


if __name__ == "__main__":
    main()
