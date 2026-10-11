"""Small persistent SQLite cache for completed video analyses."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class AnalysisCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._open_sync)

    async def close(self) -> None:
        async with self._lock:
            connection = self._connection
            self._connection = None
            if connection is not None:
                await asyncio.to_thread(connection.close)

    async def get(self, key: str, *, ttl_hours: int) -> dict[str, Any] | None:
        async with self._lock:
            return await asyncio.to_thread(self._get_sync, key, ttl_hours)

    async def put(self, key: str, value: dict[str, Any], *, max_entries: int) -> None:
        async with self._lock:
            await asyncio.to_thread(self._put_sync, key, value, max_entries)

    async def count(self) -> int:
        async with self._lock:
            return await asyncio.to_thread(self._count_sync)

    def _open_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_cache (
                cache_key TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        connection.commit()
        self._connection = connection

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self._open_sync()
        assert self._connection is not None
        return self._connection

    def _get_sync(self, key: str, ttl_hours: int) -> dict[str, Any] | None:
        connection = self._require_connection()
        row = connection.execute(
            "SELECT payload_json, created_at FROM analysis_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        ttl_seconds = max(int(ttl_hours), 1) * 3600
        if float(row[1]) < time.time() - ttl_seconds:
            connection.execute("DELETE FROM analysis_cache WHERE cache_key = ?", (key,))
            connection.commit()
            return None
        try:
            value = json.loads(str(row[0]))
        except json.JSONDecodeError:
            connection.execute("DELETE FROM analysis_cache WHERE cache_key = ?", (key,))
            connection.commit()
            return None
        return value if isinstance(value, dict) else None

    def _put_sync(self, key: str, value: dict[str, Any], max_entries: int) -> None:
        connection = self._require_connection()
        connection.execute(
            "INSERT OR REPLACE INTO analysis_cache(cache_key, payload_json, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, ensure_ascii=False), time.time()),
        )
        keep = max(int(max_entries), 1)
        connection.execute(
            """
            DELETE FROM analysis_cache
            WHERE cache_key IN (
                SELECT cache_key FROM analysis_cache
                ORDER BY created_at DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (keep,),
        )
        connection.commit()

    def _count_sync(self) -> int:
        connection = self._require_connection()
        row = connection.execute("SELECT COUNT(*) FROM analysis_cache").fetchone()
        return int(row[0]) if row else 0

