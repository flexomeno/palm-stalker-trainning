# Detección de palmas de aceite (YOLOv8)

Pipeline para extraer muestras desde una ortofoto, entrenar un detector YOLOv8 con el dataset público de Roboflow, y **contar palmas** en recortes de tu finca mediante detección por cajas (bounding boxes).

Parte del proyecto **Palm-stalker**.

---

## ¿Para qué sirve?

| Objetivo | Cómo se logra |
|----------|----------------|
| Probar el modelo en **tus ortofotos** | Recortes 1024×1024 con `sacar_muestras.py` |
| **Contar palmas** por tile/imagen | `detectar_palmas.py` → número en barra inferior + suma en consola |
| Revisar visualmente | Imágenes en `detecciones/` con cajas (sin etiquetas de confianza) |
| Análisis GIS / Excel / Python | `exportar_detecciones_csv.py` → coordenadas por caja |
| Entrenar sin suscripción Roboflow | Colab + dataset ZIP, o `entrenar_palmas.py` en local |

El **conteo** no es un algoritmo aparte: es el **número de cajas** que el modelo detecta por imagen. Cada caja = una palma (o categoría 0–4 del dataset original).

---

## Flujo general

```
Ortofoto (.tif)
       │
       ▼  sacar_muestras.py
muestras_para_roboflow/  (JPG 1024×1024)
       │
       ├──────────────────────────────────┐
       ▼                                  ▼
  best.pt (modelo)              oil palm.v1i.yolov8.zip
       │                                  │
       │                          entrenar en Colab / entrenar_palmas.py
       │                                  │
       └──────────────┬───────────────────┘
                      ▼
            detectar_palmas.py  →  detecciones/ + conteo en imagen
            exportar_detecciones_csv.py  →  detecciones.csv
```

---

## Estructura de esta carpeta

```
pruebas_previas/
├── README.md
├── requirements.txt
├── sacar_muestras.py
├── entrenar_palmas.py
├── entrenar_oil_palm_colab.ipynb
├── detectar_palmas.py
└── exportar_detecciones_csv.py
```

Pesos y entorno compartidos en la raíz del repo: `../best.pt`, `../env/`, `../resultados_training.md`.  
Para ortofoto completa → [**ortofoto_pipeline**](../ortofoto_pipeline/README.md).  
Índice del proyecto → [README principal](../README.md).

---

## Requisitos e instalación

### Entorno para detección y entrenamiento (YOLO)

```bash
cd ..
python3.11 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -r pruebas_previas/requirements.txt
```

Necesitas **`best.pt`** en la raíz del repo (`../best.pt` desde aquí).

### Dependencias extra para `sacar_muestras.py` (ortofoto)

```bash
pip install rasterio numpy Pillow
```

---

## Entrenar el modelo

El ZIP de Roboflow trae **imágenes y etiquetas**, no el `.pt` ya entrenado. Sin suscripción de Roboflow, entrena tú mismo.

### Opción A — Google Colab (recomendado)

1. Sube a Google Drive (`Palm-stalker/`):
   - `oil palm.v1i.yolov8.zip`
   - `entrenar_oil_palm_colab.ipynb`
2. Abre el notebook → **Runtime → T4 GPU**.
3. Ejecuta las celdas en orden (monta Drive, copia ZIP a `/content`, entrena 50 épocas).
4. Descarga o copia **`best.pt`** a la raíz del repo (`../best.pt`).

**Nota:** No descomprimas leyendo directo desde Drive; el notebook copia el ZIP a disco local de Colab primero (evita error `Transport endpoint is not connected`).

**`data.yaml`:** debe usar ruta absoluta `path: /content/oil-palm-dataset` (ya incluido en el notebook).

Resultados típicos tras 50 épocas: **mAP50 ≈ 0.99**, Precision ≈ 0.99, Recall ≈ 0.98.

### Opción B — Entrenar en tu Mac

```bash
unzip "oil palm.v1i.yolov8.zip" -d oil-palm-dataset
python entrenar_palmas.py
```

Pesos en: `runs/detect/oil-palm/weights/best.pt` → cópialos como `best.pt`.

---

## Uso del conteo y detección

### 1. Generar muestras desde la ortofoto

Edita la configuración al final de `sacar_muestras.py`:

```python
ARCHIVO_ENTRADA = "../Ortofotomosaico 10 cm-px.tif"
CARPETA_RESULTADO = "muestras_para_roboflow"
CANTIDAD_DE_FOTOS = 50
TAMANO_DE_FOTO = 1024
```

```bash
python sacar_muestras.py
```

**Qué hace:** abre el GeoTIFF con `rasterio`, elige ventanas aleatorias 1024×1024, descarta zonas muy oscuras (sin datos) y guarda JPEG en `muestras_para_roboflow/`.

---

### 2. Detectar palmas y ver el conteo

```bash
python detectar_palmas.py
```

**Entrada:** `muestras_para_roboflow/`  
**Salida:** `detecciones/` — misma resolución, con:

- Cajas de detección **sin texto** de clase ni confianza
- Barra inferior: **`Detecciones: N`** (conteo de ese tile)

**Consola:** total de imágenes procesadas y suma global de detecciones.

#### Parámetros útiles

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--pesos` | `best.pt` | Archivo del modelo |
| `--entrada` | `muestras_para_roboflow` | Carpeta de imágenes |
| `--salida` | `detecciones` | Carpeta de salida |
| `--conf` | `0.25` | Umbral mínimo de confianza (sube a `0.4`–`0.5` para menos falsos positivos) |
| `--imgsz` | `1024` | Tamaño de inferencia (debe coincidir con el entrenamiento) |

```bash
python detectar_palmas.py --conf 0.4
python detectar_palmas.py --entrada otra_carpeta --salida resultados_prueba
```

#### Cómo interpretar el conteo

- **Por imagen:** el número en la barra = palmas detectadas en ese recorte.
- **Total del lote:** al final del script en terminal.
- **Limitación:** es conteo por **tile**, no por finca completa. Para conteo global hay que mosaificar la ortofoto en tiles, detectar en cada uno y **deduplicar** palmas en bordes solapados (no implementado aquí).
- **Clases 0–4:** el dataset Roboflow usa 5 categorías; el conteo incluye todas las cajas por encima de `--conf`.

---

### 3. Exportar coordenadas a CSV

```bash
python exportar_detecciones_csv.py
```

**Salida:** `detecciones.csv` (una fila por caja).

| Columna | Significado |
|---------|-------------|
| `imagen` | Nombre del JPG |
| `clase` | ID 0–4 |
| `confianza` | Score del modelo (0–1) |
| `x1`, `y1`, `x2`, `y2` | Esquinas de la caja en **píxeles** (origen arriba-izquierda) |
| `ancho`, `alto` | Tamaño de la caja |

Las imágenes **sin detecciones** no generan filas.

```bash
python exportar_detecciones_csv.py --csv reporte.csv --conf 0.4
```

**Para qué sirve el CSV:** cruzar con GIS (ubicar cada caja en coordenadas del ortofoto si guardas offset del tile), filtrar por confianza, estadísticas por clase, etc.

---

## Referencia de scripts

### `sacar_muestras.py`

| | |
|---|---|
| **Propósito** | Crear tiles JPEG desde ortofoto para pruebas o etiquetado |
| **Entrada** | GeoTIFF multibanda |
| **Salida** | `muestras_para_roboflow/palma_muestra_XXX.jpg` |
| **Dependencias** | `rasterio`, `numpy`, `Pillow` |

---

### `entrenar_palmas.py`

| | |
|---|---|
| **Propósito** | Entrenar YOLOv8n localmente con `oil-palm-dataset/` |
| **Entrada** | Dataset descomprimido + `data.yaml` (path absoluto auto-generado) |
| **Salida** | `runs/detect/oil-palm/weights/best.pt` |
| **Parámetros internos** | 50 épocas, imgsz 1024, batch 8 |

---

### `entrenar_oil_palm_colab.ipynb`

| | |
|---|---|
| **Propósito** | Mismo entrenamiento en Colab con GPU T4 y Google Drive |
| **Entrada** | ZIP en Drive → copia a `/content` → descomprime |
| **Salida** | `best.pt` en Drive y/o descarga directa |

---

### `detectar_palmas.py`

| | |
|---|---|
| **Propósito** | Inferencia visual + **conteo por imagen** |
| **Entrada** | Carpeta de JPG + `best.pt` |
| **Salida** | `detecciones/*.jpg` con cajas y barra de conteo |
| **No genera** | CSV (usar script aparte) |

---

### `exportar_detecciones_csv.py`

| | |
|---|---|
| **Propósito** | Tabla de todas las detecciones con coordenadas |
| **Entrada** | Misma que detección (carpeta + `best.pt`) |
| **Salida** | `detecciones.csv` |
| **No modifica** | Imágenes ni `detectar_palmas.py` |

---

## Dataset Roboflow

- Proyecto: [oil palm on Universe](https://universe.roboflow.com/oil-palm-ni8i3/oil-palm-oxvyn)
- Export: **YOLOv8** → `oil palm.v1i.yolov8.zip`
- ~2303 imágenes, 5 clases (`0`–`4`), tiles 1024×1024
- Licencia: CC BY 4.0

---

## Orden recomendado (primera vez)

1. `pip install -r requirements.txt` en `../env`
2. Entrenar en Colab → guardar `best.pt` en la raíz del repo
3. `python sacar_muestras.py` (si tienes ortofoto)
4. `python detectar_palmas.py` → revisar `detecciones/`
5. `python exportar_detecciones_csv.py` → analizar `detecciones.csv`

---

## Problemas frecuentes

| Problema | Solución |
|----------|----------|
| `No hay pesos en best.pt` | Copia `best.pt` desde Colab o entrena |
| Colab: error al descomprimir ZIP en Drive | Usar celda que copia a `/content` primero |
| Colab: `missing path /content/valid/images` | `path` en `data.yaml` debe ser absoluto al dataset |
| Muchas cajas falsas | `python detectar_palmas.py --conf 0.4` |
| Pocas detecciones | Bajar `--conf` a `0.15`–`0.2` |
| `sacar_muestras` falla al abrir TIF | Verificar ruta y `pip install rasterio` |
| Entrenamiento lento en Mac | Usar Colab con GPU |

---

## Créditos

- Dataset y referencia de métricas: Roboflow Universe — *oil palm* / *oil-palm-oxvyn*
- Modelo: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
