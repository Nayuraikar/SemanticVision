"""SQLite profile store (observer calibration profiles + trial log) and profile -> Observer conversion.
Storage only: no calibration or engine imports."""
from __future__ import annotations
import json
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

import numpy as np

from core.observers import Observer

DEFAULT_DB = Path("bench/logs/profiles.sqlite")
PROFILE_PREFIX = "profile:"
KINDS = ("protan", "deutan", "tritan")

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles(
  id TEXT PRIMARY KEY, created REAL NOT NULL, updated REAL NOT NULL, status TEXT NOT NULL,
  config TEXT NOT NULL, self_report TEXT NOT NULL, n_trials INTEGER NOT NULL DEFAULT 0,
  summary TEXT NOT NULL, samples TEXT);
CREATE TABLE IF NOT EXISTS trials(
  profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE, idx INTEGER NOT NULL,
  stim INTEGER NOT NULL, mode TEXT NOT NULL, response TEXT NOT NULL, PRIMARY KEY(profile_id, idx));
"""


class ProfileStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or DEFAULT_DB)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)

    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA foreign_keys=ON")
        db.row_factory = sqlite3.Row
        return _Ctx(db)

    def create(self, config: dict, self_report: dict, summary: dict) -> str:
        pid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._db() as db:
            db.execute("INSERT INTO profiles(id,created,updated,status,config,self_report,summary) VALUES(?,?,?,?,?,?,?)",
                       (pid, now, now, "calibrating", json.dumps(config), json.dumps(self_report), json.dumps(summary)))
        return pid

    def get(self, pid: str) -> dict | None:
        with self._db() as db:
            r = db.execute("SELECT * FROM profiles WHERE id=?", (pid,)).fetchone()
        if r is None:
            return None
        return dict(id=r["id"], created=r["created"], updated=r["updated"], status=r["status"], n_trials=r["n_trials"],
                    config=json.loads(r["config"]), self_report=json.loads(r["self_report"]),
                    summary=json.loads(r["summary"]), samples=json.loads(r["samples"]) if r["samples"] else None)

    def add_trial(self, pid: str, idx: int, stim: int, mode: str, response, summary: dict):
        with self._db() as db:
            db.execute("INSERT INTO trials(profile_id,idx,stim,mode,response) VALUES(?,?,?,?,?)",
                       (pid, idx, stim, mode, json.dumps(response)))
            db.execute("UPDATE profiles SET n_trials=?, summary=?, updated=? WHERE id=?",
                       (idx + 1, json.dumps(summary), time.time(), pid))

    def trials(self, pid: str) -> list[tuple[int, str, object]]:
        with self._db() as db:
            rows = db.execute("SELECT stim,mode,response FROM trials WHERE profile_id=? ORDER BY idx", (pid,)).fetchall()
        return [(r["stim"], r["mode"], json.loads(r["response"])) for r in rows]

    def finish(self, pid: str, summary: dict, samples: dict):
        with self._db() as db:
            db.execute("UPDATE profiles SET status='complete', summary=?, samples=?, updated=? WHERE id=?",
                       (json.dumps(summary), json.dumps(samples), time.time(), pid))

    def list_complete(self) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT id FROM profiles WHERE status='complete' ORDER BY updated DESC").fetchall()
        return [p for p in (self.get(r["id"]) for r in rows) if p]

    def delete(self, pid: str):
        with self._db() as db:
            db.execute("DELETE FROM profiles WHERE id=?", (pid,))


class _Ctx:
    """sqlite3 connection that commits and always closes (sqlite3's own context manager does not close)."""
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self.db

    def __exit__(self, et, ev, tb):
        try:
            self.db.commit() if et is None else self.db.rollback()
        finally:
            self.db.close()


def profile_observer(pid: str, summary: dict) -> Observer:
    """Posterior mean -> single-kind Observer for the solver (CVaR over the posterior comes in Phase 6).
    kind = argmax mean effective severity e_i; severity = that e_i. Observer.k is rescaled by S=(s_L+s_C)/2 so the
    isotropic tau(.) used by the confusion graph matches the calibrated threshold."""
    p = summary["params"]
    e = np.array([p["e_protan"]["mean"], p["e_deutan"]["mean"], p["e_tritan"]["mean"]])
    i = int(np.argmax(e))
    sL, sC = float(p["s_L"]["mean"]), float(p["s_C"]["mean"])
    S = 0.5 * (sL + sC)
    return Observer(id=f"{PROFILE_PREFIX}{pid}", split="train", model="machado", kind=KINDS[i],
                    severity=float(np.clip(e[i], 0.0, 1.0)), k=float(p["k"]["mean"]) * S, s_L=sL, s_C=sC)
