"""Generación de grilla de tiles sobre el raster."""

from __future__ import annotations


def _axis_starts(size: int, tile_px: int, stride: int) -> list[int]:
    if size <= tile_px:
        return [0]
    starts = list(range(0, size - tile_px + 1, stride))
    last = size - tile_px
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_tile_origins(
    width: int, height: int, tile_px: int, stride: int
) -> list[tuple[int, int, int, int]]:
    """
    Devuelve lista de (col_idx, row_idx, x0, y0) para cada tile.
    x0,y0 = esquina superior-izquierda en píxeles del raster.
    """
    if width <= 0 or height <= 0:
        return []

    tiles: list[tuple[int, int, int, int]] = []
    for row_idx, y0 in enumerate(_axis_starts(height, tile_px, stride)):
        for col_idx, x0 in enumerate(_axis_starts(width, tile_px, stride)):
            tiles.append((col_idx, row_idx, x0, y0))
    return tiles


def tile_id(col_idx: int, row_idx: int) -> str:
    return f"c{col_idx:05d}_r{row_idx:05d}"


def parse_segmento(texto: str) -> tuple[int, int, int, int]:
    """Formato: col_min,row_min,col_max,row_max (max exclusivo)."""
    parts = [int(p.strip()) for p in texto.split(",")]
    if len(parts) != 4:
        raise ValueError("segmento debe ser col_min,row_min,col_max,row_max")
    return tuple(parts)
