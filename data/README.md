# Geographical input data

The analysis runs on two lists of municipality coordinates. They are **not
distributed** with this repository; `fetch_data.py` rebuilds them.

```
python data/fetch_data.py
```

writes `data/cyl_267.csv` and `data/spain_3102.csv`, byte-identical to the
files the published results were computed from (checked against a SHA-256 of
the upstream file; row order is preserved because the estimator source sets
are drawn by row index).

| File | Nodes | Selection |
|---|---|---|
| `cyl_267.csv` | 267 | Castilla y León, municipalities with at least 1000 inhabitants |
| `spain_3102.csv` | 3102 | Mainland Spain (all communities except Canarias, Islas Baleares, Ceuta and Melilla), same criterion |

Format: semicolon-separated, UTF-8 with BOM, CRLF line endings, header
`Población;Latitud;Longitud`, WGS84 decimal degrees.

## Source

"Listado de municipios de España con longitud y latitud", published by
*Business Intelligence fácil* (2010):
<https://www.businessintelligence.info/varios/longitud-latitud-pueblos-espana.html>.
Municipality names and population come from INE; coordinates were geocoded by
the compiler through Google Maps. The same municipality selection was used in
García-Cobo and Menéndez (2021), *Future Generation Computer Systems* 115,
814–824, doi:10.1016/j.future.2020.09.037.

## Why the files are not included

The upstream listing states no licence. INE data (names, population) are
CC BY 4.0 ("Elaboración propia con datos extraídos del sitio web del INE:
www.ine.es"), but the terms for the geocoded coordinates are not established,
so they are not redistributed here.

## Known quirks

Three pairs of distinct municipalities in `spain_3102.csv` share identical
coordinates (Píñar / Pinar (El); Valle (El) / Valle del Zalabí; Aramaio /
Ibarra), giving three candidate links of zero length. Their effect on the
total-order indices is at most 0.0004 at the primary threshold. The
coordinates are geocoder output for a place name, not official municipality
centroids.
