"""Sovereign memory store: SQLite only. Transcripts + BM25 (FTS5) + embeddings.

Hybrid retrieval = FTS5/BM25 + brute-force cosine over embeddings, fused with
reciprocal rank fusion. At personal scale (thousands of utterances) brute
force cosine over a 384-d matrix is sub-millisecond — no vector DB needed.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  mode TEXT NOT NULL,                 -- 'assistant' | 'meeting'
  started_at TEXT NOT NULL,
  ended_at TEXT,
  summary TEXT
);
CREATE TABLE IF NOT EXISTS utterances(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL REFERENCES sessions(id),
  t_ms INTEGER NOT NULL,
  role TEXT NOT NULL,                 -- 'user' | 'assistant' | 'speaker'
  speaker TEXT,
  text TEXT NOT NULL,
  lang TEXT,
  emb BLOB
);
CREATE INDEX IF NOT EXISTS idx_utt_session ON utterances(session_id);
CREATE VIRTUAL TABLE IF NOT EXISTS utterances_fts USING fts5(
  text, content='', tokenize='unicode61'
);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Store:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------ writes

    def start_session(self, mode: str = "assistant") -> int:
        cur = self.conn.execute("INSERT INTO sessions(mode, started_at) VALUES(?,?)", (mode, _now()))
        self.conn.commit()
        return int(cur.lastrowid)

    def end_session(self, session_id: int, summary: str | None = None) -> None:
        self.conn.execute("UPDATE sessions SET ended_at=?, summary=? WHERE id=?", (_now(), summary, session_id))
        self.conn.commit()

    def add_utterance(self, session_id: int, t_ms: int, role: str, text: str,
                      speaker: str | None = None, lang: str | None = None,
                      emb: np.ndarray | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO utterances(session_id, t_ms, role, speaker, text, lang, emb) VALUES(?,?,?,?,?,?,?)",
            (session_id, t_ms, role, speaker, text, lang, None if emb is None else emb.astype(np.float32).tobytes()),
        )
        self.conn.execute("INSERT INTO utterances_fts(rowid, text) VALUES(?,?)", (cur.lastrowid, text))
        self.conn.commit()
        return int(cur.lastrowid)

    def set_speakers(self, labels: dict[int, str]) -> None:
        with self.conn:
            for utt_id, speaker in labels.items():
                self.conn.execute("UPDATE utterances SET speaker=? WHERE id=?", (speaker, utt_id))

    def set_summary(self, session_id: int, summary: str) -> None:
        self.end_session(session_id, summary)

    # ------------------------------------------------------------ retrieval

    def search_bm25(self, query: str, k: int = 5) -> list[dict]:
        # FTS5 MATCH syntax is strict: reduce to word terms and OR them, so
        # natural questions ("what did we discuss earlier?") never error.
        terms = re.findall(r"\w+", query)
        if not terms:
            return []
        q = " OR ".join(terms)
        rows = self.conn.execute(
            "SELECT rowid, bm25(utterances_fts) AS rank FROM utterances_fts "
            "WHERE utterances_fts MATCH ? ORDER BY rank LIMIT ?",
            (q, k),
        ).fetchall()
        return self._hydrate([r[0] for r in rows])

    def search_vec(self, query_emb: np.ndarray, k: int = 5) -> list[dict]:
        rows = self.conn.execute("SELECT id, emb FROM utterances WHERE emb IS NOT NULL").fetchall()
        if not rows:
            return []
        ids = [r[0] for r in rows]
        mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        sims = mat @ query_emb  # embeddings are L2-normalised -> dot = cosine
        order = np.argsort(-sims)[:k]
        return self._hydrate([ids[i] for i in order])

    def hybrid(self, query: str, query_emb: np.ndarray | None, k: int = 5) -> list[dict]:
        """Reciprocal rank fusion of BM25 + vector search (k0=60)."""
        bm = self.search_bm25(query, k)
        vec = self.search_vec(query_emb, k) if query_emb is not None else []
        scores: dict[int, float] = {}
        for rank, hit in enumerate(bm):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (60 + rank + 1)
        for rank, hit in enumerate(vec):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (60 + rank + 1)
        ranked = sorted(scores, key=lambda i: -scores[i])[:k]
        by_id = {h["id"]: h for h in self._hydrate(ranked)}
        return [by_id[i] for i in ranked if i in by_id]

    def _hydrate(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, session_id, t_ms, role, speaker, text, lang FROM utterances WHERE id IN ({marks})",
            ids,
        ).fetchall()
        out = [dict(zip(("id", "session_id", "t_ms", "role", "speaker", "text", "lang"), r)) for r in rows]
        out.sort(key=lambda h: ids.index(h["id"]))
        return out

    # ------------------------------------------------------------ summaries

    def recent(self, session_id: int | None = None, limit: int = 50) -> list[dict]:
        if session_id is None:
            rows = self.conn.execute(
                "SELECT id, session_id, t_ms, role, speaker, text, lang FROM utterances "
                "ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, session_id, t_ms, role, speaker, text, lang FROM utterances "
                "WHERE session_id=? ORDER BY id LIMIT ?", (session_id, limit)
            ).fetchall()
        return [dict(zip(("id", "session_id", "t_ms", "role", "speaker", "text", "lang"), r)) for r in rows]

    def export_transcript(self, session_id: int) -> str:
        sess = self.conn.execute("SELECT mode, started_at, ended_at, summary FROM sessions WHERE id=?", (session_id,)).fetchone()
        lines = [f"# session {session_id} — {sess[0]} — {sess[1]} .. {sess[2]}"]
        for u in self.recent(session_id):
            who = u["speaker"] or u["role"]
            lines.append(f"[{u['t_ms'] // 60000:02d}:{u['t_ms'] // 1000 % 60:02d}] {who}: {u['text']}")
        if sess[3]:
            lines.append(f"\n## summary\n{sess[3]}")
        return "\n".join(lines)
