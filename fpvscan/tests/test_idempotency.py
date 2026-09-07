from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from fpvscan.app.domain.exceptions import IdempotencyConflictError
from fpvscan.app.services.observation_service import ObservationService
from fpvscan.app.services.parameter_service import ParameterService
from fpvscan.app.services.session_service import SessionService


def test_apply_parameters_is_idempotent(engine, store) -> None:
    svc = ParameterService(engine, store)
    key = uuid4()

    async def run() -> None:
        first = await svc.apply_parameters(key, {"scan.threshold_db": 6.0})
        engine.cfg["scan"]["threshold_db"] = 1.0
        second = await svc.apply_parameters(key, {"scan.threshold_db": 6.0})
        assert first.to_dict() == second.to_dict()
        with pytest.raises(IdempotencyConflictError):
            await svc.apply_parameters(key, {"scan.threshold_db": 8.0})

    asyncio.run(run())


def test_create_session_is_idempotent(engine, store) -> None:
    svc = SessionService(engine, store)
    key = uuid4()

    async def run() -> None:
        a = await svc.create_session(key, "lock", title="ch 4988")
        b = await svc.create_session(key, "lock", title="ch 4988")
        assert a.id == b.id
        items = await svc.list_sessions()
        assert len(items) == 1
        with pytest.raises(IdempotencyConflictError):
            await svc.create_session(key, "sweep", title="other")

    asyncio.run(run())


def test_observation_is_idempotent(engine, store) -> None:
    sessions = SessionService(engine, store)
    obs = ObservationService(engine, store)
    key = uuid4()

    async def run() -> None:
        session = await sessions.create_session(uuid4(), "lock")
        a = await obs.add_observation(
            session.id, key, {"signal_quality": 4, "notes": "osd readable"}
        )
        b = await obs.add_observation(
            session.id, key, {"signal_quality": 4, "notes": "osd readable"}
        )
        assert a.id == b.id
        listed = await obs.list_observations(session.id)
        assert len(listed) == 1

    asyncio.run(run())
