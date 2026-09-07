from fastapi import APIRouter, Depends, Response

from fpvscan.app.core.deps import get_observation_service
from fpvscan.app.schemas.observations import (
    ObservationListResponse,
    ObservationPatchRequest,
    ObservationResponse,
    ObservationWriteRequest,
)
from fpvscan.app.services.observation_service import ObservationService

router = APIRouter(
    prefix="/api/test/sessions/{session_id}/observations",
    tags=["test-observations"],
)


@router.post("", response_model=ObservationResponse, status_code=201)
async def create_observation(
    session_id: str,
    req: ObservationWriteRequest,
    svc: ObservationService = Depends(get_observation_service),
) -> ObservationResponse:
    data = req.model_dump()
    key = data.pop("idempotency_key")
    obs = await svc.add_observation(session_id, key, **data)
    return ObservationResponse.from_domain(obs)


@router.get("", response_model=ObservationListResponse)
async def list_observations(
    session_id: str,
    svc: ObservationService = Depends(get_observation_service),
) -> ObservationListResponse:
    items = await svc.list_observations(session_id)
    return ObservationListResponse.from_domain(items)


@router.patch("/{obs_id}", response_model=ObservationResponse)
async def patch_observation(
    session_id: str,
    obs_id: str,
    req: ObservationPatchRequest,
    svc: ObservationService = Depends(get_observation_service),
) -> ObservationResponse:
    obs = await svc.patch_observation(session_id, obs_id, **req.model_dump())
    return ObservationResponse.from_domain(obs)


@router.delete("/{obs_id}", response_model=None, status_code=204)
async def delete_observation(
    session_id: str,
    obs_id: str,
    svc: ObservationService = Depends(get_observation_service),
) -> Response:
    await svc.delete_observation(session_id, obs_id)
    return Response(status_code=204)
