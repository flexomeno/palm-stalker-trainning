# Palm detector training

Entrenamiento y despliegue de un detector **YOLOv8** de palmas de aceite sobre ortofotos, como parte del proyecto **Palm-stalker**. El repositorio está organizado en **dos módulos** según el tipo de trabajo.

## Estructura del repositorio

```
palm-detector-training/
├── README.md                 ← esta guía
├── best.pt                   ← pesos entrenados (Colab → copiar aquí)
├── resultados_training.md    ← log completo del entrenamiento en Colab
├── requirements.txt          ← dependencias base (YOLO, torch, etc.)
├── env/                      ← entorno Python 3.11 recomendado
│
├── pruebas_previas/          ← muestreo, entrenamiento, detección en recortes
│   └── README.md
│
└── ortofoto_pipeline/        ← ortofoto completa: tiles, GeoJSON, mapa
    └── README.md
```

## ¿Qué carpeta usar?

| Necesitas… | Carpeta |
|------------|---------|
| Sacar muestras JPG, entrenar, probar el modelo en **recortes** | [**pruebas_previas/**](pruebas_previas/README.md) |
| Procesar el **GeoTIFF completo** (~14 GB), inventario georreferenciado, reanudar por tiles | [**ortofoto_pipeline/**](ortofoto_pipeline/README.md) |

### [pruebas_previas/](pruebas_previas/README.md)

Flujo clásico de experimentación:

- `sacar_muestras.py` — recortes aleatorios 1024×1024 desde el `.tif`
- `entrenar_palmas.py` / notebook Colab — entrenamiento con dataset Roboflow
- `detectar_palmas.py` — inferencia visual + conteo por imagen
- `exportar_detecciones_csv.py` — cajas en píxeles para Excel/GIS ligero

### [ortofoto_pipeline/](ortofoto_pipeline/README.md)

Pipeline de producción sobre la finca entera:

- Grilla con solape, inferencia **reanudable** (`state.sqlite`)
- Deduplicación global (~303k palmas únicas en la corrida de referencia)
- GeoJSON con centro, bbox, distancia al vecino
- `validar_tile.py` y `recrear_mapa.py` para validación visual

---

## Inicio rápido

```bash
# Entorno (una vez; usar Python 3.11, no 3.14)
python3.11 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Pruebas en recortes
cd pruebas_previas
python sacar_muestras.py --entrada /ruta/ortofoto.tif --salida muestras -n 50

# Ortofoto completa (desde ortofoto_pipeline/)
cd ../ortofoto_pipeline
pip install -r requirements.txt
python procesar_ortofoto.py init --entrada /ruta/ortofoto.tif --work-dir ./trabajo_finca
python procesar_ortofoto.py infer --work-dir ./trabajo_finca --pesos ../best.pt --resume
```

El archivo **`best.pt`** debe estar en la **raíz del repo** (ambas carpetas lo referencian como `../best.pt`).

---

## Review: `resultados_training.md`

Archivo de **~518 líneas** con el log crudo del entrenamiento en **Google Colab** (Ultralytics 8.4.51, GPU Tesla T4). Resume lo relevante para usar el modelo con confianza.

### Configuración del entrenamiento

| Parámetro | Valor |
|-----------|--------|
| Modelo base | `yolov8n.pt` (nano, ~3M parámetros) |
| Clases | 5 (`0`–`4`, dataset Roboflow *oil palm*) |
| Imágenes | 1024×1024 |
| Épocas | 50 (~1.07 h) |
| Batch | 16 |
| Optimizer | AdamW (auto) |
| Train / val | 1612 / 461 imágenes |
| Instancias en val | 30 579 cajas |

### Métricas finales (época 50 / `best.pt`)

Validación global en el conjunto **val**:

| Métrica | Valor |
|---------|--------|
| **Precision (P)** | 0.989 |
| **Recall (R)** | 0.981 |
| **mAP50** | 0.993 |
| **mAP50-95** | 0.930 |

Por clase (mAP50-95 destacado):

| Clase | Imágenes (val) | Instancias | P | R | mAP50 | mAP50-95 |
|-------|----------------|------------|------|------|-------|----------|
| 0 | 48 | 64 | 0.995 | 0.969 | 0.991 | 0.932 |
| 1 | 442 | 23 237 | 0.992 | 0.995 | 0.995 | **0.970** |
| 2 | 49 | 151 | 0.986 | 0.946 | 0.990 | 0.852 |
| 3 | 416 | 6 732 | 0.994 | 0.993 | 0.994 | 0.932 |
| 4 | 224 | 395 | 0.980 | 1.000 | 0.995 | 0.967 |

La clase **1** concentra la mayoría de instancias y el mejor comportamiento global. La clase **2** es la más débil (menos ejemplos en val).

### Evolución durante el entrenamiento

- **Épocas 1–10:** recall bajo (~0.35–0.70) pese a precision alta → el modelo era conservador al principio.
- **Épocas 11–20:** salto fuerte en mAP50-95 (0.55 → ~0.81).
- **Épocas 35–50:** meseta estable; mAP50-95 en val ~0.88–0.93.

### Interpretación para el proyecto Palm-stalker

**Fortalezas**

- Métricas de validación del dataset Roboflow son **muy buenas**; el modelo localiza bien palmas en tiles 1024×1024 similares al entrenamiento.
- Inferencia ~6 ms/imagen en T4 (útil para miles de tiles).

**Limitaciones (importante en ortofoto real)**

1. **Dominio:** entrenado en Roboflow; tu ortofoto (iluminación, variedad, edad) puede diferir → ajustar `--conf` (0.25–0.40) y revisar con `validar_tile.py`.
2. **Cinco clases:** el conteo en pipeline suma todas las cajas por encima del umbral; no distingue automáticamente “una palma productiva” sin reglas de clase.
3. **Val ≠ finca:** 461 imágenes de val no garantizan el mismo error en 87k×69k px; el pipeline de `ortofoto_pipeline` añade deduplicación (`d_min` 4 m) porque el solape multiplica detecciones.
4. **Log final:** parte del archivo son curvas PR embebidas (arrays largos); para decisiones usa las tablas de época 50 y la validación de `best.pt` (líneas 309–318).

### Conclusión del review

El entrenamiento en Colab **cumplió el objetivo** de obtener un `best.pt` sólido para detección en tiles. Para inventario de finca, combinar ese modelo con **`ortofoto_pipeline`** (tiles + merge + GeoJSON) es el camino adecuado; para iterar hiperparámetros o etiquetar, usar **`pruebas_previas`**.

---

## Dataset y créditos

- Dataset: [oil palm on Roboflow Universe](https://universe.roboflow.com/oil-palm-ni8i3/oil-palm-oxvyn) (YOLOv8, ~2303 imágenes, CC BY 4.0)
- Motor: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## Archivos grandes (git)

No versionar: `env/`, `*.geojson`, `state.sqlite`, ortofotos `.tif`, carpetas `trabajo_*` / `work/`. Ver `.gitignore` en cada módulo.
