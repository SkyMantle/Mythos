from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fpvscan.app.domain.exceptions import (
    ConflictError,
    DomainError,
    IdempotencyConflictError,
    NotFoundError,
    ValidationError,
)
from fpvscan.app.core.logging import request_id_var

_STATUS = {
    ValidationError: 422,
    NotFoundError: 404,
    IdempotencyConflictError: 409,
    ConflictError: 409,
    DomainError: 400,
}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def domain_error_handler(_request: Request, exc: DomainError) -> JSONResponse:
        status = 400
        for cls, code in _STATUS.items():
            if isinstance(exc, cls):
                status = code
                break
        detail: dict[str, object] = {
            "code": exc.code,
            "message": exc.message,
            "request_id": request_id_var.get(),
        }
        fields = getattr(exc, "fields", None)
        if fields:
            detail["fields"] = fields
        return JSONResponse(status_code=status, content={"detail": detail})
