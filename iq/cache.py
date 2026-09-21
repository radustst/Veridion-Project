"""Content-addressed disk cache for API responses.

Two jobs:

1. Money. Re-running the 12-query benchmark while tuning a prompt downstream
   should not re-pay for embeddings or for stages that did not change.
2. Reproducibility. The evaluation numbers in the writeup have to be
   re-derivable. A cache hit returns byte-identical output, so a second run of
   the same configuration produces the same report.

Keys are sha256 over (namespace, model, payload), so changing a prompt or a
model automatically misses rather than silently serving a stale answer.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional


def make_key(namespace: str, model: str, payload: Any) -> str:
    blob = json.dumps(
        {"ns": namespace, "model": model, "payload": payload},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    """Small thread-safe sqlite key/value store.

    sqlite rather than one-file-per-key because the embedding stage writes 477
    entries and a JSON-file-per-key layout makes the repo unpleasant to inspect.
    """

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = path
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv ("
            "  k TEXT PRIMARY KEY,"
            "  ns TEXT,"
            "  value TEXT,"
            "  created REAL DEFAULT (strftime('%s','now'))"
            ")"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS kv_ns ON kv(ns)")
        self._conn.commit()

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE k=?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def put(self, key: str, namespace: str, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (k, ns, value) VALUES (?,?,?)",
                (key, namespace, json.dumps(value, ensure_ascii=False, default=str)),
            )
            self._conn.commit()

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ns, COUNT(*) FROM kv GROUP BY ns"
            ).fetchall()
        return {"hits": self.hits, "misses": self.misses, **{r[0]: r[1] for r in rows}}

    def clear(self, namespace: str | None = None) -> None:
        with self._lock:
            if namespace:
                self._conn.execute("DELETE FROM kv WHERE ns=?", (namespace,))
            else:
                self._conn.execute("DELETE FROM kv")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
