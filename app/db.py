"""Tiny SQLite store for unanswered questions (content-gap log) and captured leads."""
import sqlite3
import threading
import time
from contextlib import contextmanager

_lock = threading.Lock()


class Database:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS unanswered(
                  id INTEGER PRIMARY KEY, ts TEXT, question TEXT, query TEXT, reason TEXT, best_sim REAL);
                CREATE TABLE IF NOT EXISTS leads(
                  id INTEGER PRIMARY KEY, ts TEXT, name TEXT, email TEXT, message TEXT);
                """
            )

    @contextmanager
    def _conn(self):
        with _lock:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _now():
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def log_unanswered(self, question, query, reason, best_sim):
        with self._conn() as c:
            c.execute("INSERT INTO unanswered(ts,question,query,reason,best_sim) VALUES(?,?,?,?,?)",
                      (self._now(), question[:500], query[:500], reason, best_sim))

    def add_lead(self, name, email, message):
        with self._conn() as c:
            c.execute("INSERT INTO leads(ts,name,email,message) VALUES(?,?,?,?)",
                      (self._now(), name[:120], email[:200], (message or "")[:1000]))

    def rows(self, table: str, limit: int = 200):
        assert table in ("unanswered", "leads")
        with self._conn() as c:
            return [dict(r) for r in c.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))]
