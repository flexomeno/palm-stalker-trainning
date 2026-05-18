"""Fase init: manifest + grilla de tiles en SQLite."""

from __future__ import annotations

import hashlib
from pathlib import Path

import rasterio

from .config import OVERLAP_PX_DEFAULT, TILE_PX_DEFAULT
from .db import (
    connect,
    init_schema,
    load_manifest,
    save_manifest,
    set_meta,
    work_paths,
)
from .geo import parse_transform, utm_epsg_from_lon_lat
from .grid import iter_tile_origins, tile_id


def _tif_fingerprint(path: Path) -> str:
    st = path.stat()
    payload = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_init(
    entrada: Path,
    work_dir: Path,
    tile_px: int = TILE_PX_DEFAULT,
    overlap_px: int = OVERLAP_PX_DEFAULT,
    force: bool = False,
) -> dict:
    entrada = entrada.expanduser().resolve()
    if not entrada.is_file():
        raise FileNotFoundError(f"No existe el GeoTIFF: {entrada}")

    manifest_path, db_path = work_paths(work_dir)
    if manifest_path.exists() and not force:
        existing = load_manifest(manifest_path)
        if existing.get("tif_path") == str(entrada):
            print(f"Manifest ya existe en {work_dir} (usa --force para regenerar).")
            return existing
        raise SystemExit(
            "work_dir ya tiene otro proyecto. Usa otro --work-dir o --force."
        )

    stride = tile_px - overlap_px
    if stride < 1:
        raise ValueError("overlap_px debe ser menor que tile_px")

    with rasterio.open(entrada) as src:
        width, height = src.width, src.height
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        transform = list(src.transform)[:6]
        bounds = src.bounds
        center_lon = (bounds.left + bounds.right) / 2
        center_lat = (bounds.top + bounds.bottom) / 2

    utm_epsg = utm_epsg_from_lon_lat(center_lon, center_lat)
    tiles = iter_tile_origins(width, height, tile_px, stride)

    manifest = {
        "tif_path": str(entrada),
        "tif_fingerprint": _tif_fingerprint(entrada),
        "width": width,
        "height": height,
        "crs": crs,
        "transform": transform,
        "tile_px": tile_px,
        "overlap_px": overlap_px,
        "stride": stride,
        "utm_epsg": utm_epsg,
        "center_lon": center_lon,
        "center_lat": center_lat,
        "n_tiles": len(tiles),
    }
    save_manifest(manifest_path, manifest)

    conn = connect(db_path)
    init_schema(conn)
    conn.execute("DELETE FROM tiles")
    conn.execute("DELETE FROM detections_raw")
    conn.execute("DELETE FROM palms_unique")
    conn.executemany(
        """
        INSERT INTO tiles(tile_id, col_idx, row_idx, x0, y0, status)
        VALUES (?, ?, ?, ?, ?, 'pending')
        """,
        [
            (tile_id(c, r), c, r, x0, y0)
            for c, r, x0, y0 in tiles
        ],
    )
    set_meta(conn, "tif_fingerprint", manifest["tif_fingerprint"])
    set_meta(conn, "phase", "init")
    conn.commit()
    conn.close()

    print(f"Manifest: {manifest_path}")
    print(f"Raster: {width} x {height} px | CRS: {crs}")
    print(f"Tiles: {len(tiles)} | tile={tile_px} overlap={overlap_px} stride={stride}")
    print(f"UTM para distancias: EPSG:{utm_epsg}")
    return manifest
