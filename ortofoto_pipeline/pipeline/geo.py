"""Transformaciones píxel raster ↔ coordenadas mapa ↔ UTM."""

from __future__ import annotations

import math

import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import Affine


def utm_epsg_from_lon_lat(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def parse_transform(manifest_transform) -> Affine:
    if isinstance(manifest_transform, list):
        return Affine(*manifest_transform[:6])
    return manifest_transform


def pixel_to_map(
    transform: Affine, col_px: float, row_px: float
) -> tuple[float, float]:
    x, y = rasterio.transform.xy(transform, row_px, col_px, offset="center")
    return float(x), float(y)


def bbox_px_to_map_polygon(
    transform: Affine,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> list[list[float]]:
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    ring = []
    for cx, cy in corners:
        lon, lat = pixel_to_map(transform, cx, cy)
        ring.append([lon, lat])
    return ring


def make_transformers(crs_wkt: str, utm_epsg: int) -> tuple[Transformer, Transformer]:
    src = CRS.from_user_input(crs_wkt)
    dst = CRS.from_epsg(utm_epsg)
    to_utm = Transformer.from_crs(src, dst, always_xy=True)
    to_map = Transformer.from_crs(dst, src, always_xy=True)
    return to_utm, to_map


def geodesic_distance_m(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    return abs(dist)


def utm_distance_m(
    to_utm: Transformer, lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    x1, y1 = to_utm.transform(lon1, lat1)
    x2, y2 = to_utm.transform(lon2, lat2)
    return math.hypot(x2 - x1, y2 - y1)
