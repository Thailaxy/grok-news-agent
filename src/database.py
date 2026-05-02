import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    user_id TEXT,
    topic TEXT NOT NULL,
    post TEXT NOT NULL,
    research_json TEXT,
    approved INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at);
CREATE INDEX IF NOT EXISTS idx_posts_topic ON posts(topic);
"""


class Database:
    def __init__(self, file_path: str = "posts_log.db") -> None:
        self.file_path = file_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            file_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage our own txn via executescript
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def log_post(
        self,
        topic: str,
        post: str,
        research_data: Any,
        user_id: Optional[str] = None,
        approved: bool = True,
    ) -> int:
        """Insert a post row; returns the new row id."""
        row = (
            datetime.now(timezone.utc).isoformat(),
            user_id,
            topic,
            post,
            json.dumps(research_data, ensure_ascii=False) if research_data is not None else None,
            1 if approved else 0,
        )
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO posts (created_at, user_id, topic, post, research_json, approved)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                row,
            )
            row_id = cur.lastrowid
        logger.info("Logged post id=%s topic=%r approved=%s", row_id, topic, approved)
        return row_id

    def recent(self, limit: int = 10) -> list[dict]:
        """Return the most recent posts (newest first)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, created_at, user_id, topic, post, approved"
                " FROM posts ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
