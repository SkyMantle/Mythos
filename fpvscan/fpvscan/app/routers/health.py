from fastapi import APIRouter

from fpvscan.app.schemas.sessions import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
@router.get("/api/test/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="fpvscan-test")
