from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import UUID

from fpvscan.app.domain.exceptions import IdempotencyConflictError
from fpvscan.app.repositories.protocols import TestStoreProtocol


def fingerprint(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


async def replay_or_none(
    repo: TestStoreProtocol,
    key: UUID | str,
    operation: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    token = str(key)
    fp = fingerprint(payload)
    existing = await repo.get_idempotent_fingerprint(token)
    if existing is None:
        return None
    op, stored_fp = existing
    if op != operation or stored_fp != fp:
        raise IdempotencyConflictError(
            "idempotency key reused with a different request"
        )
    cached = await repo.get_idempotent(token, operation)
    if cached is None:
        raise IdempotencyConflictError("idempotency record is incomplete")
    return cached


async def remember(
    repo: TestStoreProtocol,
    key: UUID | str,
    operation: str,
    payload: dict[str, Any],
    result: dict[str, Any],
) -> None:
    await repo.put_idempotent(str(key), operation, fingerprint(payload), result)
