from fastapi import APIRouter, Request, Response

from fpvscan.app.schemas.sessions import HealthResponse

router = APIRouter(tags=["health"])
RECONFIGURE_GRACE_MS = 5_000


@router.get("/api/health", response_model=HealthResponse)
@router.get("/api/test/health", response_model=HealthResponse)
async def health(request: Request, response: Response) -> HealthResponse:
    port = getattr(request.app.state, "engine_port", None)
    snap = port.snapshot() if port is not None else {}
    alive = snap.get("engine_alive")
    sdr_state = snap.get("sdr_state")
    state_age_ms = int(snap.get("sdr_state_age_ms") or 0)
    if alive is False:
        status = "unhealthy"
    elif (
        sdr_state == "RECONFIGURING"
        and state_age_ms <= RECONFIGURE_GRACE_MS
    ):
        status = "ok"
    elif sdr_state in {
        "WAITING_DEVICE", "RECOVERING", "DEGRADED",
        "STREAM_STOPPING", "RECONFIGURING", "STREAM_RECOVERING",
    }:
        status = "degraded"
    elif sdr_state == "ERROR":
        status = "unhealthy"
    else:
        status = "ok"
    if status != "ok":
        response.status_code = 503
    return HealthResponse(
        status=status,
        service="fpvscan",
        engine_alive=alive,
        sdr_state=sdr_state,
        sdr_state_age_ms=state_age_ms,
        sdr_operation=snap.get("sdr_operation"),
        sdr_last_operation=snap.get("sdr_last_operation"),
        sdr_consecutive_errors=int(snap.get("sdr_consecutive_errors") or 0),
        sdr_open_attempts=int(snap.get("sdr_open_attempts") or 0),
        sdr_open_failures=int(snap.get("sdr_open_failures") or 0),
        sdr_next_retry_ms=int(snap.get("sdr_next_retry_ms") or 0),
        last_error=snap.get("engine_error") or snap.get("sdr_last_error"),
        last_error_code=snap.get("sdr_last_error_code"),
        last_error_time=snap.get("sdr_last_error_time"),
        last_error_operation=snap.get("sdr_last_error_operation"),
        last_action=snap.get("sdr_last_recovery_action"),
        message=(
            "SDR hardware unavailable; service remains online and will retry"
            if alive is not False
            and sdr_state in {
                "WAITING_DEVICE", "RECOVERING", "STREAM_STOPPING",
                "RECONFIGURING", "STREAM_RECOVERING",
            }
            else None
        ),
    )
