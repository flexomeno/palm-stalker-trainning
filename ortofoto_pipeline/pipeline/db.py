"""Estado SQLite: tiles, detecciones crudas, palmas únicas."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import DB_NAME, MANIFEST_NAME


def work_paths(work_dir: Path) -> tuple[Path, Path]:
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir / MANIFEST_NAME, work_dir / DB_NAME


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tiles (
            tile_id TEXT PRIMARY KEY,
            col_idx INTEGER NOT NULL,
            row_idx INTEGER NOT NULL,
            x0 INTEGER NOT NULL,
            y0 INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error_msg TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_tiles_status ON tiles(status);
        CREATE INDEX IF NOT EXISTS idx_tiles_col_row ON tiles(col_idx, row_idx);

        CREATE TABLE IF NOT EXISTS detections_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tile_id TEXT NOT NULL,
            x1_px REAL, y1_px REAL, x2_px REAL, y2_px REAL,
            cx_px REAL, cy_px REAL,
            lon REAL, lat REAL,
            utm_x REAL, utm_y REAL,
            conf REAL, cls INTEGER,
            FOREIGN KEY (tile_id) REFERENCES tiles(tile_id)
        );
        CREATE INDEX IF NOT EXISTS idx_raw_tile ON detections_raw(tile_id);

        CREATE TABLE IF NOT EXISTS palms_unique (
            palm_id TEXT PRIMARY KEY,
            lon REAL NOT NULL,
            lat REAL NOT NULL,
            utm_x REAL NOT NULL,
            utm_y REAL NOT NULL,
            x1_px REAL, y1_px REAL, x2_px REAL, y2_px REAL,
            bbox_lon_min REAL, bbox_lat_min REAL,
            bbox_lon_max REAL, bbox_lat_max REAL,
            conf REAL,
            cls INTEGER,
            source_tile TEXT,
            dist_neighbor_m REAL,
            neighbor_palm_id TEXT
        );
        """
    )
    conn.commit()


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_manifest(manifest_path: Path, data: dict[str, Any]) -> None:
    manifest_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def count_tiles(conn: sqlite3.Connection, status: str | None = None) -> int:
    if status:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM tiles WHERE status=?", (status,)
        ).fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) AS n FROM tiles").fetchone()
    return int(row["n"])


def iter_pending_tiles(
    conn: sqlite3.Connection,
    segmento: tuple[int, int, int, int] | None = None,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    sql = (
        "SELECT * FROM tiles WHERE status IN ('pending', 'error') "
    )
    params: list[Any] = []
    if segmento:
        c0, r0, c1, r1 = segmento
        sql += " AND col_idx >= ? AND col_idx < ? AND row_idx >= ? AND row_idx < ?"
        params.extend([c0, c1, r0, r1])
    sql += " ORDER BY row_idx, col_idx"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).fetchall()


def mark_tile(conn: sqlite3.Connection, tile_id: str, status: str, error: str | None = None):
    conn.execute(
        "UPDATE tiles SET status=?, error_msg=?, updated_at=datetime('now') WHERE tile_id=?",
        (status, error, tile_id),
    )


def delete_raw_for_tile(conn: sqlite3.Connection, tile_id: str) -> None:
    conn.execute("DELETE FROM detections_raw WHERE tile_id=?", (tile_id,))


def insert_raw_detection(
    conn: sqlite3.Connection,
    tile_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cx: float,
    cy: float,
    lon: float,
    lat: float,
    utm_x: float,
    utm_y: float,
    conf: float,
    cls: int,
) -> None:
    conn.execute(
        """
        INSERT INTO detections_raw(
            tile_id, x1_px, y1_px, x2_px, y2_px, cx_px, cy_px,
            lon, lat, utm_x, utm_y, conf, cls
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (tile_id, x1, y1, x2, y2, cx, cy, lon, lat, utm_x, utm_y, conf, cls),
    )


def fetch_all_raw(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM detections_raw ORDER BY conf DESC"
    ).fetchall()


def clear_unique(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM palms_unique")


def insert_unique_palm(conn: sqlite3.Connection, palm: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO palms_unique(
            palm_id, lon, lat, utm_x, utm_y,
            x1_px, y1_px, x2_px, y2_px,
            bbox_lon_min, bbox_lat_min, bbox_lon_max, bbox_lat_max,
            conf, cls, source_tile
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            palm["palm_id"],
            palm["lon"],
            palm["lat"],
            palm["utm_x"],
            palm["utm_y"],
            palm.get("x1_px"),
            palm.get("y1_px"),
            palm.get("x2_px"),
            palm.get("y2_px"),
            palm.get("bbox_lon_min"),
            palm.get("bbox_lat_min"),
            palm.get("bbox_lon_max"),
            palm.get("bbox_lat_max"),
            palm["conf"],
            palm["cls"],
            palm.get("source_tile"),
        ),
    )


def fetch_all_unique(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM palms_unique ORDER BY palm_id").fetchall()


def update_neighbor(
    conn: sqlite3.Connection,
    palm_id: str,
    dist_m: float,
    neighbor_id: str,
) -> None:
    conn.execute(
        """
        UPDATE palms_unique
        SET dist_neighbor_m=?, neighbor_palm_id=?
        WHERE palm_id=?
        """,
        (dist_m, neighbor_id, palm_id),
    )
