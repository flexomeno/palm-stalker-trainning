#!/usr/bin/env python3
"""
Genera una imagen completa del ortofoto con detecciones del GeoJSON (cajas + centros).

El TIF original (~87k×69k px) es demasiado grande para abrir entero en RAM; se
lee reducido (--ancho-max) y las coordenadas en píxeles se escalan igual.

Ejemplo:
  python recrear_mapa.py \\
    --work-dir ./trabajo_finca \\
    --geojson ./palmas.geojson \\
    --salida ./trabajo_finca/mapa_palmas.jpg \\
    --ancho-max 12000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.db import connect, fetch_all_unique, load_manifest, work_paths  # noqa: E402
from pipeline.stages_infer import _to_uint8_rgb  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Imagen completa del ortofoto con bboxes/centros desde GeoJSON"
    )
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--geojson", type=Path, default=None)
    p.add_argument(
        "--fuente",
        choices=("geojson", "sqlite"),
        default="geojson",
        help="geojson (default) o sqlite (más rápido de cargar)",
    )
    p.add_argument("--salida", type=Path, required=True, help="JPG o PNG")
    p.add_argument(
        "--ancho-max",
        type=int,
        default=12000,
        help="Ancho máximo de la imagen de salida en píxeles",
    )
    p.add_argument(
        "--solo-puntos",
        action="store_true",
        help="Solo centros (más rápido con muchas palmas)",
    )
    p.add_argument(
        "--sin-cajas",
        action="store_true",
        help="Alias útil: equivalente a --solo-puntos",
    )
    p.add_argument("--grosor-caja", type=int, default=1)
    p.add_argument("--radio-punto", type=int, default=2)
    return p.parse_args()


def load_palms_geojson(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        if props.get("tipo") == "bbox":
            continue
        if props.get("x1_px") is None:
            continue
        out.append(props)
    return out


def load_palms_sqlite(work_dir: Path) -> list[dict]:
    _, db_path = work_paths(work_dir)
    conn = connect(db_path)
    rows = fetch_all_unique(conn)
    conn.close()
    return [
        {
            "palm_id": r["palm_id"],
            "conf": r["conf"],
            "clase": r["cls"],
            "x1_px": r["x1_px"],
            "y1_px": r["y1_px"],
            "x2_px": r["x2_px"],
            "y2_px": r["y2_px"],
            "lon": r["lon"],
            "lat": r["lat"],
        }
        for r in rows
    ]


def leer_ortofoto_reducida(
    tif_path: Path, ancho_max: int
) -> tuple[np.ndarray, float]:
    with rasterio.open(tif_path) as src:
        scale = min(1.0, ancho_max / src.width)
        out_w = max(1, int(src.width * scale))
        out_h = max(1, int(src.height * scale))
        print(f"Ortofoto {src.width}×{src.height} → salida {out_w}×{out_h} (escala {scale:.4f})")
        bandas = min(3, src.count)
        data = src.read(
            list(range(1, bandas + 1)),
            out_shape=(bandas, out_h, out_w),
            resampling=Resampling.bilinear,
        )
        if bandas == 1:
            data = np.repeat(data, 3, axis=0)
    rgb = _to_uint8_rgb(data)
    return rgb, scale


def dibujar_detecciones(
    img_bgr: np.ndarray,
    palms: list[dict],
    scale: float,
    solo_puntos: bool,
    grosor: int,
    radio: int,
) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    n = len(palms)
    print(f"Dibujando {n} palmas...")

    for i, p in enumerate(palms):
        if i > 0 and i % 50_000 == 0:
            print(f"  {i}/{n} ({100 * i / n:.1f}%)")

        cx = int((float(p["x1_px"]) + float(p["x2_px"])) / 2 * scale)
        cy = int((float(p["y1_px"]) + float(p["y2_px"])) / 2 * scale)

        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(out, (cx, cy), radio, (0, 0, 255), -1)

        if solo_puntos:
            continue

        x1 = int(float(p["x1_px"]) * scale)
        y1 = int(float(p["y1_px"]) * scale)
        x2 = int(float(p["x2_px"]) * scale)
        y2 = int(float(p["y2_px"]) * scale)
        if x2 < 0 or y2 < 0 or x1 >= w or y1 >= h:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), grosor)

    return out


def anadir_leyenda(img_bgr: np.ndarray, n: int, scale: float) -> np.ndarray:
    bar_h = 40
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (img_bgr.shape[1], bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0, img_bgr)
    texto = f"Palmas: {n} | escala {scale:.4f} | verde=caja rojo=centro"
    cv2.putText(
        img_bgr,
        texto,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img_bgr


def main() -> None:
    args = parse_args()
    manifest_path, _ = work_paths(args.work_dir)
    if not manifest_path.exists():
        raise SystemExit(f"No hay manifest en {args.work_dir}")

    manifest = load_manifest(manifest_path)
    tif_path = Path(manifest["tif_path"])
    if not tif_path.is_file():
        raise SystemExit(f"No existe TIF: {tif_path}")

    solo_puntos = args.solo_puntos or args.sin_cajas

    if args.fuente == "sqlite":
        palms = load_palms_sqlite(args.work_dir)
    else:
        geojson = args.geojson or (Path(args.work_dir) / "palmas.geojson")
        if not geojson.is_file():
            raise SystemExit(f"No existe GeoJSON: {geojson}")
        print(f"Cargando {geojson}...")
        palms = load_palms_geojson(geojson)

    if not palms:
        raise SystemExit("No hay palmas para dibujar.")

    rgb, scale = leer_ortofoto_reducida(tif_path, args.ancho_max)
    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    del rgb

    img_bgr = dibujar_detecciones(
        img_bgr,
        palms,
        scale,
        solo_puntos,
        args.grosor_caja,
        args.radio_punto,
    )
    img_bgr = anadir_leyenda(img_bgr, len(palms), scale)

    salida = args.salida.expanduser().resolve()
    salida.parent.mkdir(parents=True, exist_ok=True)
    ext = salida.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(salida), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        if ext != ".png":
            salida = salida.with_suffix(".png")
        cv2.imwrite(str(salida), img_bgr)

    print(f"Guardado: {salida} ({img_bgr.shape[1]}×{img_bgr.shape[0]} px)")


if __name__ == "__main__":
    main()
