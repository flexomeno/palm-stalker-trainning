"""
Entrena un detector YOLOv8 con el dataset exportado de Roboflow (oil palm.v1i.yolov8.zip).

El ZIP contiene imágenes y etiquetas, no el modelo ya entrenado de Universe.
Tras este script tendrás best.pt en runs/detect/<nombre>/weights/
"""
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DATASET_DIR = BASE / "oil-palm-dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
MODELO_BASE = "yolov8n.pt"  # n=rápido; usa yolov8s.pt o yolov8m.pt si tienes GPU y más tiempo
EPOCHS = 50
IMGSZ = 1024
BATCH = 8
NOMBRE_RUN = "oil-palm"


def escribir_data_yaml():
    """Ultralytics necesita path absoluto; 'path: .' falla si el cwd no es el dataset."""
    DATA_YAML.write_text(
        f"""path: {DATASET_DIR}
train: train/images
val: valid/images
test: test/images

nc: 5
names: ["0", "1", "2", "3", "4"]
"""
    )


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"No se encontró {DATASET_DIR}. Descomprime 'oil palm.v1i.yolov8.zip' ahí."
        )

    escribir_data_yaml()
    model = YOLO(MODELO_BASE)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(BASE / "runs" / "detect"),
        name=NOMBRE_RUN,
        exist_ok=True,
    )

    pesos = BASE / "runs" / "detect" / NOMBRE_RUN / "weights" / "best.pt"
    print(f"\nEntrenamiento listo. Pesos: {pesos}")
    print("Siguiente paso: python detectar_palmas.py")


if __name__ == "__main__":
    main()
