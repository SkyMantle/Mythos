from fastapi import APIRouter, Depends

from fpvscan.app.core.deps import get_live_service
from fpvscan.app.schemas.live import LiveContextResponse
from fpvscan.app.services.live_service import LiveService

router = APIRouter(prefix="/api/test", tags=["test-live"])


@router.get("/live", response_model=LiveContextResponse)
async def live_context(
    svc: LiveService = Depends(get_live_service),
) -> LiveContextResponse:
    data = await svc.live_context()
    return LiveContextResponse.model_validate(data)
