#!/usr/bin/env python3
"""
Visualiza un tile del GeoTIFF con bounding boxes del GeoJSON (validación).

Ejemplo:
  python validar_tile.py --work-dir ./trabajo_finca --geojson ./trabajo_finca/palmas.geojson \\
    --tile c00040_r00040 --salida ./validacion/tile_c00040_r00040.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.db import connect, load_manifest, work_paths  # noqa: E402
from pipeline.stages_infer import _read_tile_rgb  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dibujar bboxes del GeoJSON sobre un tile del ortofoto"
    )
    p.add_argument("--work-dir", type=Path, required=True, help="Carpeta con manifest.json")
    p.add_argument(
        "--geojson",
        type=Path,
        default=None,
        help="palmas.geojson (default: work-dir/palmas.geojson)",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tile", type=str, help="ID del tile, ej. c00040_r00040")
    g.add_argument(
        "--col-row",
        type=str,
        metavar="COL,ROW",
        help="Índices de grilla, ej. 40,40",
    )
    p.add_argument(
        "--salida",
        type=Path,
        default=None,
        help="Imagen JPG de salida (default: work-dir/validacion/<tile>.jpg)",
    )
    p.add_argument(
        "--mostrar-vecino",
        action="store_true",
        help="Etiqueta con distancia al vecino más cercano",
    )
    p.add_argument(
        "--sin-etiquetas",
        action="store_true",
        help="Solo cajas, sin texto",
    )
    return p.parse_args()


def resolve_tile(conn, tile: str | None, col_row: str | None) -> dict:
    if tile:
        row = conn.execute(
            "SELECT * FROM tiles WHERE tile_id=?", (tile,)
        ).fetchone()
        if not row:
            raise SystemExit(f"No existe el tile: {tile}")
        return dict(row)

    col_s, row_s = col_row.split(",")
    col_idx, row_idx = int(col_s.strip()), int(row_s.strip())
    row = conn.execute(
        "SELECT * FROM tiles WHERE col_idx=? AND row_idx=?",
        (col_idx, row_idx),
    ).fetchone()
    if not row:
        raise SystemExit(f"No hay tile en col={col_idx} row={row_idx}")
    return dict(row)


def load_palms_from_geojson(geojson_path: Path) -> list[dict]:
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    palms = []
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        if props.get("tipo") == "bbox":
            continue
        if props.get("x1_px") is None:
            continue
        palms.append(props)
    return palms


def palms_in_tile(
    palms: list[dict], x0: int, y0: int, w: int, h: int
) -> list[dict]:
    x1_tile, y1_tile = x0, y0
    x2_tile, y2_tile = x0 + w, y0 + h
    dentro = []
    for p in palms:
        bx1, by1 = float(p["x1_px"]), float(p["y1_px"])
        bx2, by2 = float(p["x2_px"]), float(p["y2_px"])
        if bx2 < x1_tile or bx1 > x2_tile or by2 < y1_tile or by1 > y2_tile:
            continue
        local = {
            **p,
            "lx1": bx1 - x0,
            "ly1": by1 - y0,
            "lx2": bx2 - x0,
            "ly2": by2 - y0,
            "cx": (bx1 + bx2) / 2 - x0,
            "cy": (by1 + by2) / 2 - y0,
        }
        dentro.append(local)
    return dentro


def dibujar(
    img_bgr: np.ndarray,
    palms: list[dict],
    tile_id: str,
    mostrar_vecino: bool,
    sin_etiquetas: bool,
) -> np.ndarray:
    out = img_bgr.copy()
    for p in palms:
        x1, y1 = int(p["lx1"]), int(p["ly1"])
        x2, y2 = int(p["lx2"]), int(p["ly2"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cx, cy = int(p["cx"]), int(p["cy"])
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

        if sin_etiquetas:
            continue

        label = p.get("palm_id", "")
        conf = p.get("conf")
        if conf is not None:
            label = f"{label} {conf:.2f}"
        if mostrar_vecino and p.get("dist_vecino_m") is not None:
            label += f" | {p['dist_vecino_m']:.1f}m"

        ty = max(y1 - 6, 14)
        cv2.putText(
            out,
            label,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    bar_h = 36
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], bar_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
    titulo = f"{tile_id} | palmas: {len(palms)}"
    cv2.putText(
        out,
        titulo,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main() -> None:
    args = parse_args()
    manifest_path, db_path = work_paths(args.work_dir)
    if not manifest_path.exists():
        raise SystemExit(f"No hay manifest en {args.work_dir}")

    geojson_path = args.geojson or (Path(args.work_dir) / "palmas.geojson")
    if not geojson_path.is_file():
        raise SystemExit(f"No existe GeoJSON: {geojson_path}")

    manifest = load_manifest(manifest_path)
    tif_path = Path(manifest["tif_path"])
    if not tif_path.is_file():
        raise SystemExit(f"No existe el TIF: {tif_path}")

    conn = connect(db_path)
    tile = resolve_tile(conn, args.tile, args.col_row)
    conn.close()

    tile_id = tile["tile_id"]
    x0, y0 = int(tile["x0"]), int(tile["y0"])
    tile_px = manifest["tile_px"]
    width, height = manifest["width"], manifest["height"]
    w = min(tile_px, width - x0)
    h = min(tile_px, height - y0)

    print(f"Tile {tile_id} @ ({x0}, {y0}) tamaño {w}x{h}")
    palms_all = load_palms_from_geojson(geojson_path)
    palms = palms_in_tile(palms_all, x0, y0, w, h)
    print(f"Palmas en tile: {len(palms)} (de {len(palms_all)} en GeoJSON)")

    with rasterio.open(tif_path) as src:
        rgb = _read_tile_rgb(src, x0, y0, w, h)

    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    resultado = dibujar(
        img_bgr,
        palms,
        tile_id,
        args.mostrar_vecino,
        args.sin_etiquetas,
    )

    salida = args.salida or (Path(args.work_dir) / "validacion" / f"{tile_id}.jpg")
    salida = salida.expanduser().resolve()
    salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(salida), resultado, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Guardado: {salida}")


if __name__ == "__main__":
    main()
