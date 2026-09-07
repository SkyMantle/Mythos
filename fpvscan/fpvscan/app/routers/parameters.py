from typing import Literal

from fastapi import APIRouter, Depends, Query

from fpvscan.app.core.deps import get_parameter_service
from fpvscan.app.schemas.parameters import (
    ApplyParametersRequest,
    ApplyParametersResponse,
    CurrentParametersResponse,
    ParameterCatalogResponse,
)
from fpvscan.app.services.parameter_service import ParameterService

router = APIRouter(prefix="/api/test", tags=["test-parameters"])


@router.get("/parameters", response_model=ParameterCatalogResponse)
async def list_parameters(
    svc: ParameterService = Depends(get_parameter_service),
    mode: Literal["sweep", "lock"] | None = Query(default=None),
) -> ParameterCatalogResponse:
    items = await svc.list_parameters(mode)
    return ParameterCatalogResponse.from_domain(items)


@router.get("/parameters/current", response_model=CurrentParametersResponse)
async def current_parameters(
    svc: ParameterService = Depends(get_parameter_service),
) -> CurrentParametersResponse:
    values = await svc.current_values()
    return CurrentParametersResponse(values=values)


@router.put("/parameters", response_model=ApplyParametersResponse)
async def apply_parameters(
    req: ApplyParametersRequest,
    svc: ParameterService = Depends(get_parameter_service),
) -> ApplyParametersResponse:
    result = await svc.apply_parameters(req.idempotency_key, req.values)
    return ApplyParametersResponse.from_domain(result)
