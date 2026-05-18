"""Fase vecinos: distancia al vecino más cercano en UTM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .db import connect, fetch_all_unique, update_neighbor, work_paths


def run_neighbors(work_dir: Path) -> None:
    _, db_path = work_paths(work_dir)
    conn = connect(db_path)
    palms = fetch_all_unique(conn)
    if len(palms) < 2:
        print("Menos de 2 palmas; no hay vecinos que calcular.")
        conn.close()
        return

    ids = [p["palm_id"] for p in palms]
    coords = np.array([[p["utm_x"], p["utm_y"]] for p in palms], dtype=np.float64)
    tree = cKDTree(coords)

    for i, palm in enumerate(palms):
        dists, indices = tree.query(coords[i], k=2)
        if np.ndim(dists) == 0:
            continue
        neighbor_idx = int(indices[1])
        dist_m = float(dists[1])
        update_neighbor(conn, palm["palm_id"], dist_m, ids[neighbor_idx])

    conn.commit()
    conn.close()
    print(f"Vecinos calculados para {len(palms)} palmas.")
