from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fpvscan.app.core.logging import request_id_var
from fpvscan.app.domain.exceptions import ValidationError
from fpvscan.app.domain.models import AppliedParams, ParamSpec
from fpvscan.app.repositories.protocols import TestStoreProtocol
from fpvscan.app.services.catalog import build_catalog, catalog_values, coerce_param
from fpvscan.app.services.catalog_tasks import curated_keys
from fpvscan.app.services.idempotency import remember, replay_or_none
from fpvscan.app.services.ports import EnginePort
from fpvscan.scan_view import affect_of, needs_lock_refresh, pending_for

log = logging.getLogger("fpvscan.test.parameters")


class ParameterService:
    def __init__(self, engine: EnginePort, repo: TestStoreProtocol) -> None:
        self._engine = engine
        self._repo = repo

    def _catalog(self) -> list[ParamSpec]:
        return build_catalog(
            self._engine.current_cfg(),
            self._engine.default_cfg(),
            self._engine.yaml_text(),
        )

    async def list_parameters(self, mode: str | None = None) -> list[ParamSpec]:
        items = self._catalog()
        if mode is None:
            return items
        if mode not in ("sweep", "lock"):
            raise ValidationError(
                f"mode must be sweep or lock, not {mode}",
                {"mode": "enum"},
            )
        allow = curated_keys(mode)
        return [item for item in items if item.key in allow]

    async def current_values(self) -> dict[str, Any]:
        return catalog_values(self._engine.current_cfg())

    async def apply_parameters(
        self,
        idempotency_key: UUID,
        values: dict[str, Any],
    ) -> AppliedParams:
        replay = await replay_or_none(
            self._repo, idempotency_key, "parameters.apply", {"values": values}
        )
        if replay is not None:
            return AppliedParams.from_dict(replay)
        specs = {s.key: s for s in self._catalog()}
        unknown = [k for k in values if k not in specs]
        if unknown:
            raise ValidationError(
                f"unknown parameter(s): {', '.join(unknown)}",
                {k: "unknown" for k in unknown},
            )
        coerced = {key: coerce_param(specs[key], raw) for key, raw in values.items()}
        if coerced:
            self._engine.apply_values(coerced)
        snap = self._engine.snapshot()
        relocked = False
        if str(snap.get("mode") or "").upper() == "LOCK" and needs_lock_refresh(coerced):
            self._engine.refresh_lock()
            relocked = True
        pending, reasons = pending_for(coerced, lock_refreshed=relocked)
        result = AppliedParams(
            values=catalog_values(self._engine.current_cfg()),
            applied_keys=sorted(coerced),
            pending_keys=pending,
            pending_reasons=reasons,
            affects={key: affect_of(key) for key in coerced},
        )
        await remember(
            self._repo,
            idempotency_key,
            "parameters.apply",
            {"values": values},
            result.to_dict(),
        )
        log.info(
            "parameters_applied keys=%s rid=%s",
            result.applied_keys,
            request_id_var.get(),
        )
        return result
