"""Fase infer: YOLO por tile con checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from ultralytics import YOLO

from .config import CONF_DEFAULT, IMGSZ_DEFAULT
from .db import (
    connect,
    count_tiles,
    delete_raw_for_tile,
    insert_raw_detection,
    iter_pending_tiles,
    load_manifest,
    mark_tile,
    work_paths,
)
from .geo import make_transformers, parse_transform, pixel_to_map
from .grid import parse_segmento


def _to_uint8_rgb(data: np.ndarray) -> np.ndarray:
    """data shape (H, W, 3)."""
    arr = np.moveaxis(data, 0, -1) if data.ndim == 3 and data.shape[0] in (3, 4) else data
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() > 255:
            arr = arr / (arr.max() / 255.0 + 1e-6)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr[:, :, :3]


def _read_tile_rgb(src, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    ventana = Window(x0, y0, w, h)
    bandas = min(3, src.count)
    data = src.read(list(range(1, bandas + 1)), window=ventana)
    if bandas == 1:
        data = np.repeat(data, 3, axis=0)
    rgb = _to_uint8_rgb(data)
    if rgb.shape[0] != h or rgb.shape[1] != w:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[: rgb.shape[0], : rgb.shape[1]] = rgb
        rgb = canvas
    return rgb


def run_infer(
    work_dir: Path,
    pesos: Path,
    conf: float = CONF_DEFAULT,
    imgsz: int = IMGSZ_DEFAULT,
    resume: bool = False,
    segmento: str | None = None,
    max_tiles: int | None = None,
    reprocess: bool = False,
) -> None:
    manifest_path, db_path = work_paths(work_dir)
    manifest = load_manifest(manifest_path)
    tif_path = Path(manifest["tif_path"])
    if not tif_path.is_file():
        raise FileNotFoundError(f"GeoTIFF no encontrado: {tif_path}")

    if not pesos.exists():
        raise FileNotFoundError(f"No hay pesos: {pesos}")

    seg = parse_segmento(segmento) if segmento else None
    conn = connect(db_path)

    if reprocess or not resume:
        q = "SELECT * FROM tiles"
        params: list = []
        if seg:
            c0, r0, c1, r1 = seg
            q += " WHERE col_idx >= ? AND col_idx < ? AND row_idx >= ? AND row_idx < ?"
            params = [c0, c1, r0, r1]
        for t in conn.execute(q, params).fetchall():
            mark_tile(conn, t["tile_id"], "pending")
        conn.commit()

    tiles = iter_pending_tiles(conn, seg, max_tiles)

    if not tiles:
        print("No hay tiles pendientes.")
        conn.close()
        return

    transform = parse_transform(manifest["transform"])
    to_utm, _ = make_transformers(manifest["crs"], manifest["utm_epsg"])
    tile_px = manifest["tile_px"]
    width, height = manifest["width"], manifest["height"]

    model = YOLO(str(pesos))
    print(f"Inferencia: {len(tiles)} tiles | conf={conf} imgsz={imgsz}")

    done = 0
    with rasterio.open(tif_path) as src:
        for row in tiles:
            tid = row["tile_id"]
            x0, y0 = int(row["x0"]), int(row["y0"])
            w = min(tile_px, width - x0)
            h = min(tile_px, height - y0)

            try:
                delete_raw_for_tile(conn, tid)
                rgb = _read_tile_rgb(src, x0, y0, w, h)
                results = model.predict(
                    source=rgb,
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False,
                )
                r = results[0]
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    gx1, gy1 = x0 + x1, y0 + y1
                    gx2, gy2 = x0 + x2, y0 + y2
                    gcx = (gx1 + gx2) / 2
                    gcy = (gy1 + gy2) / 2
                    lon, lat = pixel_to_map(transform, gcx, gcy)
                    utm_x, utm_y = to_utm.transform(lon, lat)
                    insert_raw_detection(
                        conn,
                        tid,
                        gx1,
                        gy1,
                        gx2,
                        gy2,
                        gcx,
                        gcy,
                        lon,
                        lat,
                        float(utm_x),
                        float(utm_y),
                        float(box.conf.item()),
                        int(box.cls.item()),
                    )
                mark_tile(conn, tid, "done")
                conn.commit()
                done += 1
                if done % 10 == 0:
                    pend = count_tiles(conn, "pending")
                    err = count_tiles(conn, "error")
                    print(f"  {done}/{len(tiles)} en esta corrida | pendientes={pend} errores={err}")
            except Exception as exc:
                mark_tile(conn, tid, "error", str(exc))
                conn.commit()
                print(f"  ERROR {tid}: {exc}")

    conn.close()
    print(f"Inferencia terminada: {done} tiles procesados en esta corrida.")
