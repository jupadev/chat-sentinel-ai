"""SQLite persistence shared by the API and the worker (WAL mode, one file in a shared volume)."""

import hashlib
import json
import secrets
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from sentinel.models import ChatMessage, Decision

SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    webhook_url TEXT,
    webhook_secret TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL REFERENCES clients(id),
    prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    scopes TEXT NOT NULL,
    created_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL REFERENCES clients(id),
    message_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    author_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    UNIQUE (client_id, message_id)
);
CREATE INDEX IF NOT EXISTS messages_conversation ON messages (client_id, conversation_id, created_at);
CREATE INDEX IF NOT EXISTS messages_author ON messages (client_id, author_id, created_at);
CREATE TABLE IF NOT EXISTS verdicts (
    message_key TEXT PRIMARY KEY REFERENCES messages(id),
    verdict TEXT NOT NULL,
    category TEXT,
    reason TEXT NOT NULL,
    stage TEXT NOT NULL,
    risk_score REAL,
    scores TEXT NOT NULL,
    rule_hits TEXT NOT NULL,
    latency_ms TEXT NOT NULL,
    doubtful INTEGER NOT NULL DEFAULT 0,
    decided_at TEXT NOT NULL
);
-- Human reviews are kept as labeled data for a future SetFit fine-tuning.
CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    message_key TEXT NOT NULL REFERENCES messages(id),
    decision TEXT NOT NULL,
    model_verdict TEXT NOT NULL,
    final_verdict TEXT NOT NULL,
    category TEXT,
    reviewer_id TEXT,
    notes TEXT,
    reviewed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS reviews_message ON reviews (message_key);
"""

KEY_PREFIX = "sk_"


def now() -> str:
    return _iso(datetime.now(UTC))


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


@dataclass(frozen=True)
class Principal:
    client_id: str
    key_id: str
    scopes: frozenset[str]


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)

    # --- clients and keys -------------------------------------------------

    def create_client(self, name: str, webhook_url: str | None) -> tuple[str, str]:
        client_id, secret = str(uuid.uuid4()), secrets.token_urlsafe(32)
        self._db.execute(
            "INSERT INTO clients VALUES (?, ?, ?, ?, ?)", (client_id, name, webhook_url, secret, now())
        )
        return client_id, secret

    def client_by_name(self, name: str) -> sqlite3.Row | None:
        return self._db.execute("SELECT * FROM clients WHERE name = ?", (name,)).fetchone()

    def client(self, client_id: str) -> sqlite3.Row | None:
        return self._db.execute("SELECT * FROM clients WHERE id = ?", (client_id,)).fetchone()

    def create_api_key(self, client_id: str, scopes: list[str]) -> tuple[str, str]:
        """Return (key_id, raw_key). The raw key is shown once and never stored."""
        key_id, raw = str(uuid.uuid4()), KEY_PREFIX + secrets.token_urlsafe(32)
        self._db.execute(
            "INSERT INTO api_keys VALUES (?, ?, ?, ?, ?, ?, NULL)",
            (key_id, client_id, raw[:10], hash_key(raw), json.dumps(sorted(scopes)), now()),
        )
        return key_id, raw

    def revoke_api_key(self, key_id: str) -> bool:
        cur = self._db.execute(
            "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL", (now(), key_id)
        )
        return cur.rowcount == 1

    def list_api_keys(self) -> list[sqlite3.Row]:
        return self._db.execute(
            "SELECT k.id, c.name AS client, k.prefix, k.scopes, k.created_at, k.revoked_at "
            "FROM api_keys k JOIN clients c ON c.id = k.client_id ORDER BY k.created_at"
        ).fetchall()

    def principal(self, raw_key: str) -> Principal | None:
        row = self._db.execute(
            "SELECT id, client_id, scopes FROM api_keys WHERE key_hash = ? AND revoked_at IS NULL",
            (hash_key(raw_key),),
        ).fetchone()
        return Principal(row["client_id"], row["id"], frozenset(json.loads(row["scopes"]))) if row else None

    # --- messages ----------------------------------------------------------

    def insert_message(
        self, client_id: str, message_id: str, conversation_id: str, author_id: str, text: str, created_at: datetime
    ) -> tuple[str, bool]:
        """Return (key, created). A repeated message_id for the same client is a no-op."""
        key = str(uuid.uuid4())
        cur = self._db.execute(
            "INSERT OR IGNORE INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (key, client_id, message_id, conversation_id, author_id, text, _iso(created_at), now()),
        )
        if cur.rowcount == 1:
            return key, True
        row = self._db.execute(
            "SELECT id FROM messages WHERE client_id = ? AND message_id = ?", (client_id, message_id)
        ).fetchone()
        return row["id"], False

    def messages_by_keys(self, keys: list[str]) -> list[ChatMessage]:
        if not keys:
            return []
        marks = ",".join("?" * len(keys))
        rows = self._db.execute(f"SELECT * FROM messages WHERE id IN ({marks})", keys).fetchall()
        by_key = {r["id"]: _message(r) for r in rows}
        return [by_key[k] for k in keys if k in by_key]

    def received_at(self, key: str) -> datetime:
        row = self._db.execute("SELECT received_at FROM messages WHERE id = ?", (key,)).fetchone()
        return datetime.fromisoformat(row["received_at"])

    def has_verdict(self, key: str) -> bool:
        return self._db.execute("SELECT 1 FROM verdicts WHERE message_key = ?", (key,)).fetchone() is not None

    # History protocol (see sentinel.models.History)

    def recent_texts_by_author(self, message: ChatMessage, since: datetime) -> list[str]:
        rows = self._db.execute(
            "SELECT text FROM messages WHERE client_id = ? AND author_id = ? AND id != ? "
            "AND created_at >= ? AND created_at <= ? ORDER BY created_at DESC LIMIT 50",
            (message.client_id, message.author_id, message.key, _iso(since), _iso(message.created_at)),
        ).fetchall()
        return [r["text"] for r in rows]

    def conversation_context(self, message: ChatMessage, limit: int) -> list[ChatMessage]:
        if limit <= 0:
            return []
        rows = self._db.execute(
            "SELECT * FROM messages WHERE client_id = ? AND conversation_id = ? AND id != ? AND created_at <= ? "
            "ORDER BY created_at DESC LIMIT ?",
            (message.client_id, message.conversation_id, message.key, _iso(message.created_at), limit),
        ).fetchall()
        return [_message(r) for r in reversed(rows)]

    # --- verdicts and reviews ----------------------------------------------

    def save_verdict(self, key: str, d: Decision) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO verdicts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key, d.verdict, d.category, d.reason, d.stage, d.risk_score, json.dumps(d.scores),
                json.dumps([asdict(h) for h in d.rule_hits]), json.dumps(d.latency_ms), int(d.doubtful), now(),
            ),
        )  # fmt: skip

    def moderation(self, client_id: str, message_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT m.id AS key, m.message_id, m.conversation_id, v.* FROM messages m "
            "LEFT JOIN verdicts v ON v.message_key = m.id WHERE m.client_id = ? AND m.message_id = ?",
            (client_id, message_id),
        ).fetchone()
        if row is None:
            return None
        result = {"message_id": row["message_id"], "status": "done" if row["verdict"] else "pending"}
        if row["verdict"]:
            result |= _verdict(row)
            review = self._db.execute(
                "SELECT decision, final_verdict, category, reviewer_id, notes, reviewed_at FROM reviews "
                "WHERE message_key = ? ORDER BY reviewed_at DESC LIMIT 1",
                (row["key"],),
            ).fetchone()
            result["review"] = dict(review) if review else None
        return result

    def review_queue(self, client_id: str, limit: int) -> list[dict]:
        rows = self._db.execute(
            "SELECT m.message_id, m.conversation_id, m.author_id, m.text, m.created_at, v.* "
            "FROM verdicts v JOIN messages m ON m.id = v.message_key "
            "WHERE m.client_id = ? AND v.verdict IN ('flag', 'hide') "
            "AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.message_key = m.id) "
            "ORDER BY v.decided_at LIMIT ?",
            (client_id, limit),
        ).fetchall()
        return [
            {k: r[k] for k in ("message_id", "conversation_id", "author_id", "text", "created_at")} | _verdict(r)
            for r in rows
        ]

    def save_review(
        self,
        client_id: str,
        message_id: str,
        decision: str,
        final_verdict: str | None,
        category: str | None,
        reviewer_id: str | None,
        notes: str | None,
    ) -> dict | None:
        row = self._db.execute(
            "SELECT m.id, v.verdict, v.category FROM messages m JOIN verdicts v ON v.message_key = m.id "
            "WHERE m.client_id = ? AND m.message_id = ?",
            (client_id, message_id),
        ).fetchone()
        if row is None:
            return None
        if final_verdict is None:
            final_verdict = row["verdict"] if decision == "confirm" else "ok"
        if category is None and decision == "confirm":
            category = row["category"]
        review = {
            "decision": decision,
            "model_verdict": row["verdict"],
            "final_verdict": final_verdict,
            "category": category,
            "reviewer_id": reviewer_id,
            "notes": notes,
            "reviewed_at": now(),
        }
        self._db.execute(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), row["id"], *review.values()),
        )
        return review


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    # Fixed width so string comparison in SQL orders like time.
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _message(row: sqlite3.Row) -> ChatMessage:
    return ChatMessage(
        key=row["id"],
        client_id=row["client_id"],
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        author_id=row["author_id"],
        text=row["text"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _verdict(row: sqlite3.Row) -> dict:
    return {
        "verdict": row["verdict"],
        "category": row["category"],
        "reason": row["reason"],
        "stage": row["stage"],
        "risk_score": row["risk_score"],
        "scores": json.loads(row["scores"]),
        "rule_hits": json.loads(row["rule_hits"]),
        "latency_ms": json.loads(row["latency_ms"]),
        "doubtful": bool(row["doubtful"]),
        "decided_at": row["decided_at"],
    }
