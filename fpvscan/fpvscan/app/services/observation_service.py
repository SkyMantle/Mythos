from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

from fpvscan.app.core.logging import request_id_var, session_id_var
from fpvscan.app.domain.clock import now_ms, utc_now_iso
from fpvscan.app.domain.exceptions import NotFoundError, ValidationError
from fpvscan.app.domain.models import DEFAULT_SCHEMA_FIELDS, Observation, ObservationSchema, Session
from fpvscan.app.repositories.protocols import TestStoreProtocol
from fpvscan.app.services.catalog import catalog_values
from fpvscan.app.services.idempotency import remember, replay_or_none
from fpvscan.app.services.ports import EnginePort
from fpvscan.app.services.schema_fields import normalize_schema_fields, validate_field_values

log = logging.getLogger("fpvscan.test.observations")


class ObservationService:
    def __init__(self, engine: EnginePort, repo: TestStoreProtocol) -> None:
        self._engine = engine
        self._repo = repo

    async def get_schema(self) -> ObservationSchema:
        return await self._repo.get_active_schema()

    async def put_schema(self, idempotency_key: UUID, fields: list[dict[str, Any]]) -> ObservationSchema:
        body = {"fields": fields}
        replay = await replay_or_none(self._repo, idempotency_key, "schema.put", body)
        if replay is not None:
            return ObservationSchema.from_dict(replay)
        schema = ObservationSchema(
            id=str(uuid4()),
            fields=normalize_schema_fields(fields),
            updated_at=utc_now_iso(),
        )
        await self._repo.put_schema(schema)
        await remember(self._repo, idempotency_key, "schema.put", body, schema.to_dict())
        log.info("schema_updated id=%s fields=%s rid=%s", schema.id, len(schema.fields), request_id_var.get())
        return schema

    async def add_observation(
        self,
        session_id: str,
        idempotency_key: UUID,
        fields: dict[str, Any],
        timestamp_ms: int | None = None,
        frequency_hz: float | None = None,
        rssi: float | None = None,
        snr: float | None = None,
        video_metrics: dict[str, Any] | None = None,
        parameter_snapshot: dict[str, Any] | None = None,
        frame_ref: str | None = None,
    ) -> Observation:
        session_id_var.set(session_id)
        body = {
            "session_id": session_id,
            "fields": fields,
            "timestamp_ms": timestamp_ms,
            "frequency_hz": frequency_hz,
            "rssi": rssi,
            "snr": snr,
            "video_metrics": video_metrics,
            "parameter_snapshot": parameter_snapshot,
            "frame_ref": frame_ref,
        }
        replay = await replay_or_none(self._repo, idempotency_key, "observation.create", body)
        if replay is not None:
            return Observation.from_dict(replay)
        session = await self._require_writable(session_id)
        snap = self._engine.snapshot()
        now = utc_now_iso()
        freq = _resolve_frequency_hz(frequency_hz, fields, snap)
        incoming = _autofill_fields(dict(fields or {}), freq, snap)
        obs = Observation(
            id=str(uuid4()),
            session_id=session.id,
            timestamp_ms=timestamp_ms if timestamp_ms is not None else now_ms(),
            frequency_hz=freq,
            rssi=rssi,
            snr=snr if snr is not None else _snr_from_snap(snap),
            video_metrics=video_metrics or dict(snap.get("video") or {}),
            parameter_snapshot=parameter_snapshot or catalog_values(self._engine.current_cfg()),
            fields=validate_field_values(_schema_for_values(session.schema_snapshot), incoming),
            frame_ref=frame_ref or snap.get("last_frame_ref"),
            engine_status=_status_view(snap),
            created_at=now,
            updated_at=now,
        )
        await self._repo.create_observation(obs)
        await remember(self._repo, idempotency_key, "observation.create", body, obs.to_dict())
        log.info("observation_created id=%s session=%s rid=%s", obs.id, session.id, request_id_var.get())
        return obs

    async def list_observations(self, session_id: str) -> list[Observation]:
        await self._require_session(session_id)
        return await self._repo.list_observations(session_id)

    async def patch_observation(
        self,
        session_id: str,
        obs_id: str,
        **changes: Any,
    ) -> Observation:
        session = await self._require_writable(session_id)
        obs = await self._repo.get_observation(session_id, obs_id)
        if obs is None:
            raise NotFoundError(f"observation {obs_id} not found")
        if "fields" in changes and changes["fields"] is not None:
            merged = {**obs.fields, **changes["fields"]}
            obs.fields = validate_field_values(
                _schema_for_values(session.schema_snapshot), merged,
            )
        for name in ("timestamp_ms", "frequency_hz", "rssi", "snr", "frame_ref"):
            if name in changes and changes[name] is not None:
                setattr(obs, name, changes[name])
        if changes.get("video_metrics") is not None:
            obs.video_metrics = dict(changes["video_metrics"])
        if changes.get("parameter_snapshot") is not None:
            obs.parameter_snapshot = dict(changes["parameter_snapshot"])
        obs.updated_at = utc_now_iso()
        return await self._repo.update_observation(obs)

    async def delete_observation(self, session_id: str, obs_id: str) -> None:
        await self._require_writable(session_id)
        deleted = await self._repo.delete_observation(session_id, obs_id)
        if not deleted:
            raise NotFoundError(f"observation {obs_id} not found")

    async def _require_session(self, session_id: str) -> Session:
        session = await self._repo.get_session(session_id)
        if session is None:
            raise NotFoundError(f"session {session_id} not found")
        return session

    async def _require_writable(self, session_id: str) -> Session:
        session = await self._require_session(session_id)
        if session.status == "completed":
            raise ValidationError("session is completed", {"status": "terminal"})
        return session


def _schema_for_values(snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {str(f.get("key")): f for f in snapshot if f.get("key")}
    for spec in DEFAULT_SCHEMA_FIELDS:
        index.setdefault(spec["key"], spec)
    return list(index.values())


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _resolve_frequency_hz(
    frequency_hz: float | None,
    fields: dict[str, Any],
    snap: dict[str, Any],
) -> float | None:
    if frequency_hz is not None:
        return float(frequency_hz)
    mhz = fields.get("frequency_mhz")
    if not _blank(mhz):
        try:
            return float(mhz) * 1e6
        except (TypeError, ValueError):
            pass
    return _freq_from_snap(snap)


def _autofill_fields(
    fields: dict[str, Any],
    frequency_hz: float | None,
    snap: dict[str, Any],
) -> dict[str, Any]:
    if _blank(fields.get("frequency_mhz")) and frequency_hz:
        fields["frequency_mhz"] = round(float(frequency_hz) / 1e6, 3)
    if _blank(fields.get("bandwidth_note")):
        bw = (snap.get("spectrum") or {}).get("bw_hz")
        if bw:
            mhz = float(bw) / 1e6
            fields["bandwidth_note"] = f"{mhz:g} MHz"
    return fields


def _status_view(snap: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": snap.get("mode"),
        "tuned_hz": snap.get("tuned_hz"),
        "lock_target": snap.get("lock_target"),
        "auto": snap.get("auto"),
        "fps": snap.get("fps"),
        "afc_hz": snap.get("afc_hz"),
        "freq_err_hz": snap.get("freq_err_hz"),
        "clip_frac": snap.get("clip_frac"),
        "overflows": snap.get("overflows"),
        "recording": snap.get("recording"),
        "video": snap.get("video"),
    }


def _freq_from_snap(snap: dict[str, Any]) -> float | None:
    for key in ("lock_target", "tuned_hz"):
        value = snap.get(key)
        if value:
            return float(value)
    video = snap.get("video") or {}
    if video.get("freq_hz"):
        return float(video["freq_hz"])
    return None


def _snr_from_snap(snap: dict[str, Any]) -> float | None:
    target = snap.get("lock_target") or snap.get("tuned_hz")
    if not target:
        return None
    best: float | None = None
    best_df = 1e18
    for det in snap.get("detections") or []:
        freq = det.get("freq_hz")
        if freq is None:
            continue
        df = abs(float(freq) - float(target))
        if df < best_df:
            best_df = df
            best = det.get("snr_db")
    if best is not None and best_df <= 8e6:
        return float(best)
    return None
