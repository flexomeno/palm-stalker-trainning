# Pipeline ortofoto (tiles + GeoJSON)

Procesamiento **progresivo y reanudable** de un GeoTIFF grande: inferencia YOLO por tiles con solape, deduplicación global, distancia al vecino más cercano y exportación **GeoJSON**.

Complementa el módulo [**pruebas_previas**](../pruebas_previas/README.md) (recortes y pruebas puntuales). Guía general del repo: [README principal](../README.md).

## Requisitos

- Python **3.11** (mismo `env` del proyecto padre)
- `best.pt` en la raíz del repo (o ruta con `--pesos`)
- Dependencias del padre: `pip install -r ../requirements.txt`
- Dependencias extra:

```bash
pip install -r requirements.txt
```

## Flujo

```bash
cd ortofoto_pipeline
source ../env/bin/activate

# 1) Crear grilla (~10k tiles para ortofoto 87k×69k con tile 1024 / overlap 256)
python procesar_ortofoto.py init \
  --entrada "/ruta/ortofotomosaico_10_cm-px.tif" \
  --work-dir ./trabajo_finca

# 2) Inferencia (reanudable; parar con Ctrl+C y volver a ejecutar)
python procesar_ortofoto.py infer \
  --work-dir ./trabajo_finca \
  --pesos ../best.pt \
  --conf 0.35 \
  --resume

# Por segmentos (cols/filas de la grilla, max exclusivo)
python procesar_ortofoto.py infer \
  --work-dir ./trabajo_finca \
  --segmento 0,0,20,20 \
  --max-tiles 50

# 3) Deduplicar solapes (metros UTM; ~563k crudas → minutos, no horas)
python procesar_ortofoto.py merge --work-dir ./trabajo_finca --d-min 4.0

# 4) Vecino más cercano
python procesar_ortofoto.py vecinos --work-dir ./trabajo_finca

# 5) GeoJSON
python procesar_ortofoto.py export --work-dir ./trabajo_finca --salida palmas.geojson

# Estado
python procesar_ortofoto.py status --work-dir ./trabajo_finca
```

## Validación visual (tile + GeoJSON)

Dibuja las cajas del GeoJSON sobre un recorte del ortofoto:

```bash
python validar_tile.py \
  --work-dir ./trabajo_finca \
  --geojson ./trabajo_finca/palmas.geojson \
  --tile c00040_r00040 \
  --salida ./trabajo_finca/validacion/mi_tile.jpg

# O por índices de grilla
python validar_tile.py --work-dir ./trabajo_finca --col-row 40,40

# Con distancia al vecino en la etiqueta
python validar_tile.py --work-dir ./trabajo_finca --tile c00040_r00040 --mostrar-vecino
```

Salida: JPG con cajas verdes, centro rojo y barra con `tile_id` + conteo.

## Mapa completo (ortofoto + todas las detecciones)

Genera una sola imagen reducida del ortofoto con todas las cajas/centros del GeoJSON:

```bash
python recrear_mapa.py \
  --work-dir ./trabajo_finca \
  --geojson ./palmas.geojson \
  --salida ./trabajo_finca/mapa_palmas.jpg \
  --ancho-max 12000

# Más rápido: cargar desde SQLite y solo puntos
python recrear_mapa.py \
  --work-dir ./trabajo_finca \
  --fuente sqlite \
  --solo-puntos \
  --salida ./trabajo_finca/mapa_puntos.jpg \
  --ancho-max 16000
```

No genera un GeoTIFF de 14 GB: exporta **JPG/PNG** a escala legible (`--ancho-max` controla el tamaño).

## Carpeta de trabajo (`--work-dir`)

| Archivo | Contenido |
|---------|-----------|
| `manifest.json` | TIF, CRS, transform, tile/overlap, UTM |
| `state.sqlite` | Tiles (`pending`/`done`/`error`), detecciones, palmas únicas |
| `palmas.geojson` | Salida tras `export` |

## GeoJSON

Por cada palma:

- **Point**: centro (`lon`, `lat`)
- **Polygon** (opcional): bbox en mapa
- **properties**: `palm_id`, `conf`, `clase`, `bbox_*`, `dist_vecino_m`, `id_vecino`, píxeles globales

Distancias en **metros** (proyección UTM automática según el centro del raster).

## Parámetros clave

| Flag | Default | Nota |
|------|---------|------|
| `--tile` | 1024 | Tamaño del recorte |
| `--overlap` | 256 | Solape entre tiles |
| `--d-min` | 4.0 | Distancia mínima para fusionar duplicados (m) |
| `--conf` | 0.35 | Umbral YOLO |
| `--segmento` | — | `col_min,row_min,col_max,row_max` |
| `--max-tiles` | — | Límite por ejecución (útil para pruebas) |

## Reanudar

- `infer --resume` (default): solo tiles `pending` o `error`.
- Tras interrumpir, vuelve a ejecutar el mismo comando `infer`.
- **`merge`**: si lo interrumpes, no pierdes `detections_raw`; vuelve a lanzar `merge` (rehace todo el NMS). Muestra progreso cada 100k detecciones.
- `vecinos` / `export` solo cuando `merge` termine con el mensaje `Palmas únicas: N`.

## Notas

- El ortofoto en EPSG:4326 se procesa en grados; las distancias usan UTM derivado del centro.
- Ajusta `--d-min` según el diámetro medio de copa (~4–8 m a 10 cm/px).
- Para toda la finca hacen falta miles de inferencias; usa `--segmento` y `--max-tiles` para pruebas.
