"""Fase merge: deduplicación por distancia en UTM (NMS greedy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .config import D_MIN_M_DEFAULT
from .db import (
    clear_unique,
    connect,
    count_tiles,
    fetch_all_raw,
    insert_unique_palm,
    load_manifest,
    work_paths,
)
from .geo import bbox_px_to_map_polygon, parse_transform


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

    print(f"Deduplicando {len(raw)} detecciones crudas (d_min={d_min_m} m)...")

    coords = np.array([[r["utm_x"], r["utm_y"]] for r in raw], dtype=np.float64)
    confs = np.array([r["conf"] for r in raw], dtype=np.float64)
    order = np.argsort(-confs)

    accepted: list[int] = []
    tree: cKDTree | None = None

    for idx in order:
        pt = coords[idx]
        if tree is not None:
            neighbors = tree.query_ball_point(pt, r=d_min_m)
            if neighbors:
                continue
        accepted.append(int(idx))
        if tree is None:
            tree = cKDTree(coords[accepted])
        else:
            tree = cKDTree(coords[accepted])

    transform = parse_transform(manifest["transform"])
    clear_unique(conn)

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
        palm_id = f"p_{n:07d}"
        insert_unique_palm(
            conn,
            {
                "palm_id": palm_id,
                "lon": r["lon"],
                "lat": r["lat"],
                "utm_x": r["utm_x"],
                "utm_y": r["utm_y"],
                "x1_px": r["x1_px"],
                "y1_px": r["y1_px"],
                "x2_px": r["x2_px"],
                "y2_px": r["y2_px"],
                "bbox_lon_min": min(lons),
                "bbox_lat_min": min(lats),
                "bbox_lon_max": max(lons),
                "bbox_lat_max": max(lats),
                "conf": r["conf"],
                "cls": r["cls"],
                "source_tile": r["tile_id"],
            },
        )

    conn.commit()
    conn.close()
    print(f"Palmas únicas: {len(accepted)} (de {len(raw)} crudas)")
    return len(accepted)
