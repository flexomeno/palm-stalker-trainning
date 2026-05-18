"""Exportar palms_unique a GeoJSON."""

from __future__ import annotations

import json
from pathlib import Path

from .db import connect, fetch_all_unique, load_manifest, work_paths


def run_export(work_dir: Path, salida: Path) -> None:
    manifest_path, db_path = work_paths(work_dir)
    manifest = load_manifest(manifest_path)
    conn = connect(db_path)
    palms = fetch_all_unique(conn)
    conn.close()

    features = []
    for p in palms:
        bbox_ring = None
        if p["bbox_lon_min"] is not None:
            xmin, ymin = p["bbox_lon_min"], p["bbox_lat_min"]
            xmax, ymax = p["bbox_lon_max"], p["bbox_lat_max"]
            bbox_ring = [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
                [xmin, ymin],
            ]

        props = {
            "palm_id": p["palm_id"],
            "conf": p["conf"],
            "clase": p["cls"],
            "source_tile": p["source_tile"],
            "bbox_lon_min": p["bbox_lon_min"],
            "bbox_lat_min": p["bbox_lat_min"],
            "bbox_lon_max": p["bbox_lon_max"],
            "bbox_lat_max": p["bbox_lat_max"],
            "x1_px": p["x1_px"],
            "y1_px": p["y1_px"],
            "x2_px": p["x2_px"],
            "y2_px": p["y2_px"],
            "dist_vecino_m": p["dist_neighbor_m"],
            "id_vecino": p["neighbor_palm_id"],
        }

        geom_centro = {
            "type": "Point",
            "coordinates": [p["lon"], p["lat"]],
        }

        features.append(
            {
                "type": "Feature",
                "id": p["palm_id"],
                "geometry": geom_centro,
                "properties": props,
            }
        )

        if bbox_ring:
            features.append(
                {
                    "type": "Feature",
                    "id": f"{p['palm_id']}_bbox",
                    "geometry": {"type": "Polygon", "coordinates": [bbox_ring]},
                    "properties": {**props, "tipo": "bbox"},
                }
            )

    collection = {
        "type": "FeatureCollection",
        "name": "palmas",
        "crs": {
            "type": "name",
            "properties": {"name": manifest.get("crs", "EPSG:4326")},
        },
        "properties": {
            "tif_path": manifest.get("tif_path"),
            "n_palms": len(palms),
            "utm_epsg": manifest.get("utm_epsg"),
        },
        "features": features,
    }

    salida = salida.expanduser().resolve()
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(
        json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"GeoJSON: {salida} ({len(palms)} palmas, {len(features)} features)")
