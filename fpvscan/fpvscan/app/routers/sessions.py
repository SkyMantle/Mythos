from fastapi import APIRouter, Depends

from fpvscan.app.core.deps import get_observation_service, get_session_service
from fpvscan.app.schemas.observations import ObservationSchemaPutRequest, ObservationSchemaResponse
from fpvscan.app.schemas.sessions import (
    SessionCreateRequest,
    SessionListResponse,
    SessionPatchRequest,
    SessionResponse,
)
from fpvscan.app.services.observation_service import ObservationService
from fpvscan.app.services.session_service import SessionService

router = APIRouter(prefix="/api/test", tags=["test-sessions"])


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(
    req: SessionCreateRequest,
    svc: SessionService = Depends(get_session_service),
) -> SessionResponse:
    session = await svc.create_session(
        req.idempotency_key, req.mode, req.title, req.parameter_snapshot, req.schema_id
    )
    return SessionResponse.from_domain(session)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    svc: SessionService = Depends(get_session_service),
) -> SessionListResponse:
    items = await svc.list_sessions()
    return SessionListResponse.from_domain(items)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    svc: SessionService = Depends(get_session_service),
) -> SessionResponse:
    session = await svc.get_session(session_id)
    return SessionResponse.from_domain(session)


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
async def patch_session(
    session_id: str,
    req: SessionPatchRequest,
    svc: SessionService = Depends(get_session_service),
) -> SessionResponse:
    session = await svc.patch_session(session_id, req.title, req.notes, req.status)
    return SessionResponse.from_domain(session)


@router.post("/sessions/{session_id}/complete", response_model=SessionResponse)
async def complete_session(
    session_id: str,
    svc: SessionService = Depends(get_session_service),
) -> SessionResponse:
    session = await svc.complete_session(session_id)
    return SessionResponse.from_domain(session)


@router.get("/observation-schema", response_model=ObservationSchemaResponse)
async def get_observation_schema(
    svc: ObservationService = Depends(get_observation_service),
) -> ObservationSchemaResponse:
    schema = await svc.get_schema()
    return ObservationSchemaResponse.from_domain(schema)


@router.put("/observation-schema", response_model=ObservationSchemaResponse)
async def put_observation_schema(
    req: ObservationSchemaPutRequest,
    svc: ObservationService = Depends(get_observation_service),
) -> ObservationSchemaResponse:
    fields = [f.model_dump() for f in req.fields]
    schema = await svc.put_schema(req.idempotency_key, fields)
    return ObservationSchemaResponse.from_domain(schema)
