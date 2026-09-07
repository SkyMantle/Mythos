"""SQLite persistence for testing-panel sessions and observations."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from fpvscan.app.domain.exceptions import IdempotencyConflictError
from fpvscan.app.domain.models import (
    DEFAULT_SCHEMA_FIELDS,
    Observation,
    ObservationSchema,
    Session,
)
from fpvscan.app.domain.clock import utc_now_iso

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT NOT NULL,
    schema_id TEXT NOT NULL,
    schema_snapshot TEXT NOT NULL,
    parameter_snapshot TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observation_schema (
    id TEXT PRIMARY KEY,
    fields TEXT NOT NULL,
    is_active INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    frequency_hz REAL,
    rssi REAL,
    snr REAL,
    video_metrics TEXT NOT NULL,
    parameter_snapshot TEXT NOT NULL,
    fields TEXT NOT NULL,
    frame_ref TEXT,
    engine_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_obs_session ON observations(session_id, timestamp_ms);
"""


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _session_row(row: sqlite3.Row) -> Session:
    return Session.from_dict({
        "id": row["id"],
        "title": row["title"],
        "mode": row["mode"],
        "status": row["status"],
        "notes": row["notes"],
        "schema_id": row["schema_id"],
        "schema_snapshot": json.loads(row["schema_snapshot"]),
        "parameter_snapshot": json.loads(row["parameter_snapshot"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    })


def _obs_row(row: sqlite3.Row) -> Observation:
    return Observation.from_dict({
        "id": row["id"],
        "session_id": row["session_id"],
        "timestamp_ms": row["timestamp_ms"],
        "frequency_hz": row["frequency_hz"],
        "rssi": row["rssi"],
        "snr": row["snr"],
        "video_metrics": json.loads(row["video_metrics"]),
        "parameter_snapshot": json.loads(row["parameter_snapshot"]),
        "fields": json.loads(row["fields"]),
        "frame_ref": row["frame_ref"],
        "engine_status": json.loads(row["engine_status"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    })


class SqliteTestStore:
    def __init__(self, path: Path | None = None) -> None:
        self._lock = threading.Lock()
        if path is None:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self.initialize_sync()

    def initialize_sync(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)
            row = self._conn.execute(
                "SELECT id FROM observation_schema WHERE is_active=1 LIMIT 1"
            ).fetchone()
            if row is None:
                now = utc_now_iso()
                self._conn.execute(
                    "INSERT INTO observation_schema(id, fields, is_active, updated_at) "
                    "VALUES(?,?,1,?)",
                    ("default", _dumps(DEFAULT_SCHEMA_FIELDS), now),
                )
            self._conn.commit()

    async def _run(self, fn):
        return await asyncio.to_thread(self._locked, fn)

    def _locked(self, fn):
        with self._lock:
            return fn()

    async def get_idempotent(self, key: str, operation: str) -> dict[str, Any] | None:
        def work() -> dict[str, Any] | None:
            row = self._conn.execute(
                "SELECT operation, payload FROM idempotency WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                return None
            if row["operation"] != operation:
                raise IdempotencyConflictError(
                    "idempotency key reused for a different operation"
                )
            return json.loads(row["payload"])
        return await self._run(work)

    async def get_idempotent_fingerprint(self, key: str) -> tuple[str, str] | None:
        def work() -> tuple[str, str] | None:
            row = self._conn.execute(
                "SELECT operation, fingerprint FROM idempotency WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                return None
            return row["operation"], row["fingerprint"]
        return await self._run(work)

    async def put_idempotent(
        self, key: str, operation: str, fingerprint: str, payload: dict[str, Any]
    ) -> None:
        def work() -> None:
            row = self._conn.execute(
                "SELECT operation, fingerprint FROM idempotency WHERE key=?", (key,)
            ).fetchone()
            if row is not None:
                if row["operation"] != operation or row["fingerprint"] != fingerprint:
                    raise IdempotencyConflictError(
                        "idempotency key reused with a different request"
                    )
                return
            self._conn.execute(
                "INSERT INTO idempotency(key, operation, fingerprint, payload, created_at) "
                "VALUES(?,?,?,?,?)",
                (key, operation, fingerprint, _dumps(payload), utc_now_iso()),
            )
            self._conn.commit()
        await self._run(work)

    async def create_session(self, session: Session) -> Session:
        def work() -> Session:
            self._conn.execute(
                "INSERT INTO sessions(id, title, mode, status, notes, schema_id, "
                "schema_snapshot, parameter_snapshot, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    session.id, session.title, session.mode, session.status,
                    session.notes, session.schema_id, _dumps(session.schema_snapshot),
                    _dumps(session.parameter_snapshot), session.created_at,
                    session.updated_at,
                ),
            )
            self._conn.commit()
            return session
        return await self._run(work)

    async def get_session(self, session_id: str) -> Session | None:
        def work() -> Session | None:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            return None if row is None else _session_row(row)
        return await self._run(work)

    async def list_sessions(self) -> list[Session]:
        def work() -> list[Session]:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            ).fetchall()
            return [_session_row(r) for r in rows]
        return await self._run(work)

    async def update_session(self, session: Session) -> Session:
        def work() -> Session:
            self._conn.execute(
                "UPDATE sessions SET title=?, status=?, notes=?, updated_at=? WHERE id=?",
                (session.title, session.status, session.notes, session.updated_at, session.id),
            )
            self._conn.commit()
            return session
        return await self._run(work)

    async def get_active_schema(self) -> ObservationSchema:
        def work() -> ObservationSchema:
            row = self._conn.execute(
                "SELECT * FROM observation_schema WHERE is_active=1 "
                "ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise RuntimeError("observation schema missing")
            return ObservationSchema(
                id=row["id"],
                fields=json.loads(row["fields"]),
                updated_at=row["updated_at"],
            )
        return await self._run(work)

    async def get_schema(self, schema_id: str) -> ObservationSchema | None:
        def work() -> ObservationSchema | None:
            row = self._conn.execute(
                "SELECT * FROM observation_schema WHERE id=?", (schema_id,)
            ).fetchone()
            if row is None:
                return None
            return ObservationSchema(
                id=row["id"],
                fields=json.loads(row["fields"]),
                updated_at=row["updated_at"],
            )
        return await self._run(work)

    async def put_schema(self, schema: ObservationSchema) -> ObservationSchema:
        def work() -> ObservationSchema:
            self._conn.execute("UPDATE observation_schema SET is_active=0")
            self._conn.execute(
                "INSERT INTO observation_schema(id, fields, is_active, updated_at) "
                "VALUES(?,?,1,?) "
                "ON CONFLICT(id) DO UPDATE SET fields=excluded.fields, "
                "is_active=1, updated_at=excluded.updated_at",
                (schema.id, _dumps(schema.fields), schema.updated_at),
            )
            self._conn.commit()
            return schema
        return await self._run(work)

    async def create_observation(self, obs: Observation) -> Observation:
        def work() -> Observation:
            self._conn.execute(
                "INSERT INTO observations(id, session_id, timestamp_ms, frequency_hz, "
                "rssi, snr, video_metrics, parameter_snapshot, fields, frame_ref, "
                "engine_status, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    obs.id, obs.session_id, obs.timestamp_ms, obs.frequency_hz,
                    obs.rssi, obs.snr, _dumps(obs.video_metrics),
                    _dumps(obs.parameter_snapshot), _dumps(obs.fields),
                    obs.frame_ref, _dumps(obs.engine_status),
                    obs.created_at, obs.updated_at,
                ),
            )
            self._conn.commit()
            return obs
        return await self._run(work)

    async def get_observation(self, session_id: str, obs_id: str) -> Observation | None:
        def work() -> Observation | None:
            row = self._conn.execute(
                "SELECT * FROM observations WHERE id=? AND session_id=?",
                (obs_id, session_id),
            ).fetchone()
            return None if row is None else _obs_row(row)
        return await self._run(work)

    async def list_observations(self, session_id: str) -> list[Observation]:
        def work() -> list[Observation]:
            rows = self._conn.execute(
                "SELECT * FROM observations WHERE session_id=? ORDER BY timestamp_ms ASC",
                (session_id,),
            ).fetchall()
            return [_obs_row(r) for r in rows]
        return await self._run(work)

    async def update_observation(self, obs: Observation) -> Observation:
        def work() -> Observation:
            self._conn.execute(
                "UPDATE observations SET timestamp_ms=?, frequency_hz=?, rssi=?, snr=?, "
                "video_metrics=?, parameter_snapshot=?, fields=?, frame_ref=?, "
                "updated_at=? WHERE id=? AND session_id=?",
                (
                    obs.timestamp_ms, obs.frequency_hz, obs.rssi, obs.snr,
                    _dumps(obs.video_metrics), _dumps(obs.parameter_snapshot),
                    _dumps(obs.fields), obs.frame_ref, obs.updated_at,
                    obs.id, obs.session_id,
                ),
            )
            self._conn.commit()
            return obs
        return await self._run(work)

    async def delete_observation(self, session_id: str, obs_id: str) -> bool:
        def work() -> bool:
            cur = self._conn.execute(
                "DELETE FROM observations WHERE id=? AND session_id=?",
                (obs_id, session_id),
            )
            self._conn.commit()
            return cur.rowcount > 0
        return await self._run(work)
