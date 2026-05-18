import argparse
import os
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

BASE = Path(__file__).resolve().parent
ARCHIVO_ENTRADA_DEFECTO = BASE.parent / "Palm-stalker/ortofotomosaico_10_cm-px.tif"
CARPETA_RESULTADO_DEFECTO = BASE / "muestras"
CANTIDAD_DE_FOTOS_DEFECTO = 50
TAMANO_DE_FOTO_DEFECTO = 1024

def extraer_muestras_final(ruta_tif, carpeta_salida, num_muestras=50, tamano_tile=1024):
    """
    Extrae recortes aleatorios de una ortofoto y los guarda en JPEG para Roboflow.
    """
    # 1. Crear carpeta si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        print(f"Directorio creado: {carpeta_salida}")

    # 2. Abrir la ortofoto
    try:
        with rasterio.open(ruta_tif) as src:
            ancho_total = src.width
            alto_total = src.height
            bandas = src.count
            
            print("--- Información de la Ortofoto ---")
            print(f"Dimensiones: {ancho_total} x {alto_total} píxeles")
            print(f"Número de bandas detectadas: {bandas}")
            print(f"Extrayendo {num_muestras} muestras de {tamano_tile}x{tamano_tile}...")
            print("----------------------------------")

            conteo = 0
            intentos = 0
            # Intentaremos hasta encontrar el número de muestras o llegar a un límite
            max_intentos = num_muestras * 20 
            
            while conteo < num_muestras and intentos < max_intentos:
                # Generar coordenadas aleatorias
                x = random.randint(0, ancho_total - tamano_tile)
                y = random.randint(0, alto_total - tamano_tile)

                # Definir la ventana (sub-cuadro)
                ventana = Window(x, y, tamano_tile, tamano_tile)
                
                # Leer solo las bandas 1, 2 y 3 (R, G, B)
                # Incluso si la imagen tiene 4 bandas, esto asegura un JPEG estándar
                try:
                    data = src.read([1, 2, 3], window=ventana)
                except Exception as e:
                    intentos += 1
                    continue

                # Reordenar de (Bandas, Alto, Ancho) a (Alto, Ancho, Bandas) para PIL
                data_reordenada = np.moveaxis(data, 0, -1)

                # --- FILTRO DE CALIDAD ---
                # Calculamos el promedio de color. Si es muy bajo (< 15), 
                # es probable que sea una zona negra sin datos.
                promedio_brillo = np.mean(data_reordenada)
                
                if promedio_brillo < 15:
                    intentos += 1
                    continue

                # Convertir a imagen de 8 bits
                # .astype('uint8') es vital para evitar errores de formato
                img = Image.fromarray(data_reordenada.astype('uint8'))
                
                # Guardar como JPEG de alta calidad
                nombre_archivo = f"palma_muestra_{conteo:03d}.jpg"
                ruta_final = os.path.join(carpeta_salida, nombre_archivo)
                img.save(ruta_final, "JPEG", quality=95)

                conteo += 1
                intentos += 1
                
                if conteo % 5 == 0:
                    print(f"Progreso: {conteo}/{num_muestras} imágenes generadas...")

    except Exception as e:
        print(f"Error crítico al abrir el archivo '{ruta_tif}': {e}")
        return

    print("\n--- Proceso Finalizado ---")
    print(f"Se generaron {conteo} imágenes en la carpeta: '{carpeta_salida}'")
    if conteo < num_muestras:
        print(f"Nota: Solo se obtuvieron {conteo} porque el resto de áreas eran oscuras o fuera de límites.")

def parse_args():
    p = argparse.ArgumentParser(
        description="Extrae recortes aleatorios de una ortofoto (JPEG para Roboflow)"
    )
    p.add_argument(
        "--entrada",
        type=Path,
        default=ARCHIVO_ENTRADA_DEFECTO,
        help="Ruta al GeoTIFF de la ortofoto",
    )
    p.add_argument(
        "--salida",
        type=Path,
        default=CARPETA_RESULTADO_DEFECTO,
        help="Carpeta donde guardar los JPEG",
    )
    p.add_argument(
        "-n",
        "--cantidad",
        type=int,
        default=CANTIDAD_DE_FOTOS_DEFECTO,
        help="Número de fotos a extraer (default: 50)",
    )
    p.add_argument(
        "--tamano",
        type=int,
        default=TAMANO_DE_FOTO_DEFECTO,
        help="Tamaño del recorte en píxeles, cuadrado (default: 1024)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    entrada = args.entrada.expanduser().resolve()
    if not entrada.is_file():
        raise SystemExit(f"No existe el archivo de entrada: {entrada}")
    if args.cantidad < 1 or args.tamano < 1:
        raise SystemExit("--cantidad y --tamano deben ser mayores que 0")
    extraer_muestras_final(
        str(entrada),
        str(args.salida.expanduser().resolve()),
        args.cantidad,
        args.tamano,
    )