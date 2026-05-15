"""
Exporta detecciones a CSV: imagen + coordenadas de cada caja (píxeles).

No modifica detectar_palmas.py ni las imágenes en detecciones/.
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
PESOS_DEFECTO = BASE / "best.pt"
CARPETA_ENTRADA = BASE / "muestras_para_roboflow"
CSV_DEFECTO = BASE / "detecciones.csv"
CONF = 0.25
IMGSZ = 1024

CAMPOS = [
    "imagen",
    "clase",
    "confianza",
    "x1",
    "y1",
    "x2",
    "y2",
    "ancho",
    "alto",
]


def parse_args():
    p = argparse.ArgumentParser(description="Exportar detecciones YOLOv8 a CSV")
    p.add_argument("--pesos", type=Path, default=PESOS_DEFECTO)
    p.add_argument("--entrada", type=Path, default=CARPETA_ENTRADA)
    p.add_argument("--csv", type=Path, default=CSV_DEFECTO, help="Ruta del CSV de salida")
    p.add_argument("--conf", type=float, default=CONF)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.pesos.exists():
        raise FileNotFoundError(f"No hay pesos en {args.pesos}")
    if not args.entrada.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {args.entrada}")

    model = YOLO(str(args.pesos))
    resultados = model.predict(
        source=str(args.entrada),
        conf=args.conf,
        imgsz=args.imgsz,
        save=False,
    )

    filas = []
    imagenes_con_detecciones = 0

    for r in resultados:
        nombre = Path(r.path).name
        if len(r.boxes) == 0:
            continue

        imagenes_con_detecciones += 1
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            filas.append(
                {
                    "imagen": nombre,
                    "clase": int(box.cls.item()),
                    "confianza": round(float(box.conf.item()), 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "ancho": round(x2 - x1, 2),
                    "alto": round(y2 - y1, 2),
                }
            )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS)
        writer.writeheader()
        writer.writerows(filas)

    print(f"CSV guardado: {args.csv}")
    print(f"  {len(filas)} filas ({imagenes_con_detecciones} imágenes con detecciones)")
    print(f"  {len(resultados) - imagenes_con_detecciones} imágenes sin detecciones (no aparecen en el CSV)")


if __name__ == "__main__":
    main()
