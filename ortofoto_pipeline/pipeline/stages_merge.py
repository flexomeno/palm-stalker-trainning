"""Fase merge: deduplicación por distancia en UTM (NMS greedy, grilla espacial)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import D_MIN_M_DEFAULT
from .db import (
    clear_unique,
    connect,
    count_tiles,
    fetch_all_raw,
    load_manifest,
    work_paths,
)
from .geo import bbox_px_to_map_polygon, parse_transform


def nms_greedy_utm(
    coords: np.ndarray,
    confs: np.ndarray,
    d_min_m: float,
    progress_every: int = 100_000,
) -> list[int]:
    """
    NMS por confianza: conserva detecciones cuya distancia UTM al aceptado
  más cercano sea >= d_min_m. Grilla espacial O(n) en la práctica.
    """
    order = np.argsort(-confs)
    n = len(order)
    r2 = d_min_m * d_min_m
    cell = d_min_m
    grid: dict[tuple[int, int], list[int]] = {}
    accepted: list[int] = []

    for step, pos in enumerate(order):
        if progress_every and step > 0 and step % progress_every == 0:
            print(
                f"  NMS: {step}/{n} evaluadas | "
                f"{len(accepted)} aceptadas ({100 * step / n:.1f}%)"
            )

        idx = int(order[pos])
        x, y = float(coords[idx, 0]), float(coords[idx, 1])
        ix, iy = int(math.floor(x / cell)), int(math.floor(y / cell))

        suppressed = False
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for acc in grid.get((ix + di, iy + dj), ()):
                    ax, ay = coords[acc]
                    if (x - ax) ** 2 + (y - ay) ** 2 < r2:
                        suppressed = True
                        break
                if suppressed:
                    break
            if suppressed:
                break

        if not suppressed:
            accepted.append(idx)
            grid.setdefault((ix, iy), []).append(idx)

    return accepted


def run_merge(work_dir: Path, d_min_m: float = D_MIN_M_DEFAULT) -> int:
    manifest_path, db_path = work_paths(work_dir)
    manifest = load_manifest(manifest_path)
    conn = connect(db_path)

    pending = count_tiles(conn, "pending")
    if pending > 0:
        print(f"Advertencia: quedan {pending} tiles sin inferir.")

    raw = fetch_all_raw(conn)
    if not raw:
        print("No hay detecciones crudas.")
        conn.close()
        return 0

    n_raw = len(raw)
    print(f"Deduplicando {n_raw} detecciones crudas (d_min={d_min_m} m)...")

    coords = np.array([[r["utm_x"], r["utm_y"]] for r in raw], dtype=np.float64)
    confs = np.array([r["conf"] for r in raw], dtype=np.float64)

    accepted = nms_greedy_utm(coords, confs, d_min_m)
    print(f"  NMS listo: {len(accepted)} palmas únicas de {n_raw} crudas")

    transform = parse_transform(manifest["transform"])
    clear_unique(conn)

    rows = []
    for n, idx in enumerate(accepted):
        r = raw[idx]
        ring = bbox_px_to_map_polygon(
            transform,
            r["x1_px"],
            r["y1_px"],
            r["x2_px"],
            r["y2_px"],
        )
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        rows.append(
            (
                f"p_{n:07d}",
                r["lon"],
                r["lat"],
                r["utm_x"],
                r["utm_y"],
                r["x1_px"],
                r["y1_px"],
                r["x2_px"],
                r["y2_px"],
                min(lons),
                min(lats),
                max(lons),
                max(lats),
                r["conf"],
                r["cls"],
                r["tile_id"],
            )
        )
        if (n + 1) % 50_000 == 0:
            print(f"  Preparando filas: {n + 1}/{len(accepted)}")

    print(f"  Guardando {len(rows)} palmas en SQLite...")
    conn.executemany(
        """
        INSERT INTO palms_unique(
            palm_id, lon, lat, utm_x, utm_y,
            x1_px, y1_px, x2_px, y2_px,
            bbox_lon_min, bbox_lat_min, bbox_lon_max, bbox_lat_max,
            conf, cls, source_tile
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )

    conn.commit()
    conn.close()
    print(f"Palmas únicas: {len(accepted)} (de {n_raw} crudas)")
    return len(accepted)
