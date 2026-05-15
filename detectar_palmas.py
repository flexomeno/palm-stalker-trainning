"""
Detecta palmas en las imágenes de muestras_para_roboflow usando un modelo YOLOv8 local.

Por defecto usa los pesos generados por entrenar_palmas.py.
Si descargaste best.pt desde Roboflow, pásalo con --pesos ruta/al/best.pt
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
PESOS_DEFECTO = BASE / "best.pt"
CARPETA_ENTRADA = BASE / "muestras_para_roboflow"
CARPETA_SALIDA = BASE / "detecciones"
CONF = 0.25
IMGSZ = 1024
ALTURA_AVISO = 44


def parse_args():
    p = argparse.ArgumentParser(description="Detección local de palmas con YOLOv8")
    p.add_argument(
        "--pesos",
        type=Path,
        default=PESOS_DEFECTO,
        help="Ruta al archivo .pt (best.pt)",
    )
    p.add_argument(
        "--entrada",
        type=Path,
        default=CARPETA_ENTRADA,
        help="Carpeta con imágenes JPG",
    )
    p.add_argument(
        "--salida",
        type=Path,
        default=CARPETA_SALIDA,
        help="Carpeta donde guardar resultados",
    )
    p.add_argument("--conf", type=float, default=CONF, help="Umbral de confianza")
    p.add_argument("--imgsz", type=int, default=IMGSZ, help="Tamaño de inferencia")
    return p.parse_args()


def anadir_aviso_detecciones(imagen_bgr, num_detecciones: int):
    """Barra inferior con el total de detecciones."""
    h, w = imagen_bgr.shape[:2]
    texto = f"Detecciones: {num_detecciones}"

    overlay = imagen_bgr.copy()
    cv2.rectangle(overlay, (0, h - ALTURA_AVISO), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, imagen_bgr, 0.25, 0, imagen_bgr)

    escala = max(0.55, min(0.9, w / 1400))
    grosor = max(1, int(escala * 2))
    cv2.putText(
        imagen_bgr,
        texto,
        (14, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        (255, 255, 255),
        grosor,
        cv2.LINE_AA,
    )
    return imagen_bgr


def main():
    args = parse_args()

    if not args.pesos.exists():
        raise FileNotFoundError(
            f"No hay pesos en {args.pesos}.\n"
            "Opciones:\n"
            "  1) Entrena primero: python entrenar_palmas.py\n"
            "  2) Descarga best.pt desde Roboflow (Deploy) y usa --pesos ruta/best.pt"
        )

    if not args.entrada.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {args.entrada}")

    args.salida.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.pesos))
    resultados = model.predict(
        source=str(args.entrada),
        conf=args.conf,
        imgsz=args.imgsz,
        save=False,
    )

    total = 0
    for r in resultados:
        n = len(r.boxes)
        total += n

        img = r.plot(labels=False, conf=False, boxes=True)
        img = anadir_aviso_detecciones(img, n)

        out_path = args.salida / Path(r.path).name
        cv2.imwrite(str(out_path), img)

    print(f"\nProcesadas {len(resultados)} imágenes, {total} detecciones en total.")
    print(f"Imágenes guardadas en: {args.salida}")


if __name__ == "__main__":
    main()
