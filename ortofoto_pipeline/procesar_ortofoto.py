#!/usr/bin/env python3
"""
Pipeline por tiles sobre ortofoto GeoTIFF: detección, deduplicación, vecinos, GeoJSON.

Uso rápido:
  python procesar_ortofoto.py init --entrada /ruta/ortofoto.tif --work-dir ./mi_trabajo
  python procesar_ortofoto.py infer --work-dir ./mi_trabajo --pesos ../best.pt --resume
  python procesar_ortofoto.py merge --work-dir ./mi_trabajo
  python procesar_ortofoto.py vecinos --work-dir ./mi_trabajo
  python procesar_ortofoto.py export --work-dir ./mi_trabajo --salida palmas.geojson
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar desde ortofoto_pipeline/ o desde la raíz del repo
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.config import (  # noqa: E402
    CONF_DEFAULT,
    D_MIN_M_DEFAULT,
    IMGSZ_DEFAULT,
    OVERLAP_PX_DEFAULT,
    PESOS_DEFECTO_REL,
    TILE_PX_DEFAULT,
)
from pipeline.db import connect, count_tiles, load_manifest, work_paths  # noqa: E402
from pipeline.stages_export import run_export  # noqa: E402
from pipeline.stages_infer import run_infer  # noqa: E402
from pipeline.stages_init import run_init  # noqa: E402
from pipeline.stages_merge import run_merge  # noqa: E402
from pipeline.stages_neighbors import run_neighbors  # noqa: E402


def cmd_status(args: argparse.Namespace) -> None:
    manifest_path, db_path = work_paths(args.work_dir)
    if not manifest_path.exists():
        raise SystemExit(f"No hay manifest en {args.work_dir}. Ejecuta init primero.")
    manifest = load_manifest(manifest_path)
    conn = connect(db_path)
    total = count_tiles(conn)
    done = count_tiles(conn, "done")
    pending = count_tiles(conn, "pending")
    errors = count_tiles(conn, "error")
    raw = conn.execute("SELECT COUNT(*) AS n FROM detections_raw").fetchone()["n"]
    unique = conn.execute("SELECT COUNT(*) AS n FROM palms_unique").fetchone()["n"]
    conn.close()
    print(f"Trabajo: {args.work_dir.resolve()}")
    print(f"TIF: {manifest['tif_path']}")
    print(f"Tiles: {done}/{total} hechos | {pending} pendientes | {errors} errores")
    print(f"Detecciones crudas: {raw} | Palmas únicas: {unique}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detección de palmas sobre ortofoto por tiles (reanudable)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    s_init = sub.add_parser("init", help="Crear grilla y manifest")
    s_init.add_argument("--entrada", type=Path, required=True, help="GeoTIFF")
    s_init.add_argument("--work-dir", type=Path, required=True, help="Carpeta de trabajo")
    s_init.add_argument("--tile", type=int, default=TILE_PX_DEFAULT)
    s_init.add_argument("--overlap", type=int, default=OVERLAP_PX_DEFAULT)
    s_init.add_argument("--force", action="store_true", help="Regenerar grilla")

    s_infer = sub.add_parser("infer", help="Inferencia YOLO por tile")
    s_infer.add_argument("--work-dir", type=Path, required=True)
    s_infer.add_argument(
        "--pesos",
        type=Path,
        default=_ROOT / PESOS_DEFECTO_REL,
        help="Ruta a best.pt (default: ../best.pt)",
    )
    s_infer.add_argument("--conf", type=float, default=CONF_DEFAULT)
    s_infer.add_argument("--imgsz", type=int, default=IMGSZ_DEFAULT)
    s_infer.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Solo tiles pendientes (default)",
    )
    s_infer.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Procesar todos los del segmento aunque estén done",
    )
    s_infer.add_argument(
        "--segmento",
        type=str,
        default=None,
        help="col_min,row_min,col_max,row_max (max exclusivo)",
    )
    s_infer.add_argument("--max-tiles", type=int, default=None)
    s_infer.add_argument(
        "--reprocess",
        action="store_true",
        help="Marcar tiles del segmento como pending y volver a inferir",
    )

    s_merge = sub.add_parser("merge", help="Deduplicar detecciones (NMS en metros)")
    s_merge.add_argument("--work-dir", type=Path, required=True)
    s_merge.add_argument("--d-min", type=float, default=D_MIN_M_DEFAULT, help="Metros UTM")

    s_vec = sub.add_parser("vecinos", help="Distancia al vecino más cercano")
    s_vec.add_argument("--work-dir", type=Path, required=True)

    s_exp = sub.add_parser("export", help="Exportar GeoJSON")
    s_exp.add_argument("--work-dir", type=Path, required=True)
    s_exp.add_argument(
        "--salida",
        type=Path,
        default=None,
        help="palmas.geojson (default: work-dir/palmas.geojson)",
    )

    s_st = sub.add_parser("status", help="Estado del trabajo")
    s_st.add_argument("--work-dir", type=Path, required=True)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init":
        run_init(args.entrada, args.work_dir, args.tile, args.overlap, args.force)
    elif args.command == "infer":
        if not args.resume:
            run_infer(
                args.work_dir,
                args.pesos,
                args.conf,
                args.imgsz,
                resume=False,
                segmento=args.segmento,
                max_tiles=args.max_tiles,
                reprocess=True,
            )
        else:
            run_infer(
                args.work_dir,
                args.pesos,
                args.conf,
                args.imgsz,
                resume=True,
                segmento=args.segmento,
                max_tiles=args.max_tiles,
                reprocess=args.reprocess,
            )
    elif args.command == "merge":
        run_merge(args.work_dir, args.d_min)
    elif args.command == "vecinos":
        run_neighbors(args.work_dir)
    elif args.command == "export":
        salida = args.salida or (Path(args.work_dir) / "palmas.geojson")
        run_export(args.work_dir, salida)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
