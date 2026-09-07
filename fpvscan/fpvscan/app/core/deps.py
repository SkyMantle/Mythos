from __future__ import annotations

from fastapi import Depends, Request

from fpvscan.app.repositories.protocols import TestStoreProtocol
from fpvscan.app.services.live_service import LiveService
from fpvscan.app.services.observation_service import ObservationService
from fpvscan.app.services.parameter_service import ParameterService
from fpvscan.app.services.ports import EnginePort
from fpvscan.app.services.session_service import SessionService


def get_engine_port(request: Request) -> EnginePort:
    return request.app.state.engine_port


def get_store(request: Request) -> TestStoreProtocol:
    return request.app.state.store


def get_parameter_service(
    engine: EnginePort = Depends(get_engine_port),
    repo: TestStoreProtocol = Depends(get_store),
) -> ParameterService:
    return ParameterService(engine=engine, repo=repo)


def get_session_service(
    engine: EnginePort = Depends(get_engine_port),
    repo: TestStoreProtocol = Depends(get_store),
) -> SessionService:
    return SessionService(engine=engine, repo=repo)


def get_observation_service(
    engine: EnginePort = Depends(get_engine_port),
    repo: TestStoreProtocol = Depends(get_store),
) -> ObservationService:
    return ObservationService(engine=engine, repo=repo)


def get_live_service(
    engine: EnginePort = Depends(get_engine_port),
) -> LiveService:
    return LiveService(engine=engine)
