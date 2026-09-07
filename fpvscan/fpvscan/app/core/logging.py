from __future__ import annotations

import contextvars
import logging

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id", default=None
)


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        record.session_id = session_id_var.get() or "-"
        return True


def configure_test_logging() -> None:
    logger = logging.getLogger("fpvscan.test")
    if any(isinstance(f, ContextFilter) for f in logger.filters):
        return
    logger.addFilter(ContextFilter())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s rid=%(request_id)s "
            "sid=%(session_id)s %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
