from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from fpvscan.app.domain.exceptions import ValidationError
from fpvscan.app.domain.models import DEFAULT_SCHEMA_FIELDS
from fpvscan.app.services.observation_service import ObservationService
from fpvscan.app.services.schema_fields import normalize_schema_fields, validate_field_values
from fpvscan.app.services.session_service import SessionService


def test_default_schema_types() -> None:
    fields = normalize_schema_fields(DEFAULT_SCHEMA_FIELDS)
    kinds = {f["key"]: f["type"] for f in fields}
    assert kinds["signal_quality"] == "rating"
    assert kinds["picture_lock"] == "bool"
    assert kinds["tags"] == "tag_list"
    assert kinds["notes"] == "text"


def test_schema_rejects_bad_type_and_duplicate() -> None:
    with pytest.raises(ValidationError, match="unknown type"):
        normalize_schema_fields([{"key": "x", "label": "X", "type": "blob"}])
    with pytest.raises(ValidationError, match="duplicate"):
        normalize_schema_fields([
            {"key": "a", "label": "A", "type": "text"},
            {"key": "a", "label": "A2", "type": "text"},
        ])
    with pytest.raises(ValidationError, match="enum_values"):
        normalize_schema_fields([{"key": "band", "label": "Band", "type": "enum"}])


def test_field_value_types() -> None:
    fields = normalize_schema_fields(DEFAULT_SCHEMA_FIELDS)
    ok = validate_field_values(fields, {
        "signal_quality": 5,
        "picture_lock": True,
        "tags": ["osd", "pal"],
        "notes": "clean",
        "frequency_mhz": 4988.2,
    })
    assert ok["signal_quality"] == 5
    assert ok["tags"] == ["osd", "pal"]
    with pytest.raises(ValidationError, match="above max"):
        validate_field_values(fields, {"signal_quality": 9})
    with pytest.raises(ValidationError, match="expected bool"):
        validate_field_values(fields, {"picture_lock": "yes"})
    ignored = validate_field_values(fields, {
        "signal_quality": 3,
        "not_a_field": 1,
        "frequency_mhz": "",
        "bandwidth_note": "   ",
        "notes": None,
        "tags": [],
    })
    assert ignored["signal_quality"] == 3
    assert "not_a_field" not in ignored
    assert "frequency_mhz" not in ignored
    assert "bandwidth_note" not in ignored
    assert "tags" not in ignored


def test_put_schema_and_use_it(engine, store) -> None:
    sessions = SessionService(engine, store)
    obs = ObservationService(engine, store)

    async def run() -> None:
        schema = await obs.put_schema(uuid4(), [
            {"key": "score", "label": "Score", "type": "rating", "required": True, "min": 1, "max": 3},
            {"key": "kind", "label": "Kind", "type": "enum", "enum_values": ["vtx", "camera"]},
        ])
        session = await sessions.create_session(uuid4(), "manual", schema_id=schema.id)
        with pytest.raises(ValidationError, match="required"):
            await obs.add_observation(session.id, uuid4(), {})
        row = await obs.add_observation(
            session.id, uuid4(), {"score": 2, "kind": "vtx"}
        )
        assert row.fields["score"] == 2
        assert row.engine_status["mode"] == "LOCK"
        assert row.frequency_hz == 4988e6

    asyncio.run(run())
