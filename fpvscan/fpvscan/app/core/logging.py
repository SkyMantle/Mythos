from __future__ import annotations

import contextvars
import logging
import unicodedata

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id", default=None
)

_FACTORY_INSTALLED = False


def sanitize_log_text(value: object) -> str:
    """Return journal-safe Unicode text without embedded control bytes."""
    text = str(value)
    out: list[str] = []
    for char in text:
        if char in "\n\t":
            out.append(char)
        elif char == "\r":
            out.append("\\r")
        elif unicodedata.category(char) == "Cc":
            out.append(f"\\x{ord(char):02x}")
        else:
            out.append(char)
    return "".join(out)


def _fill_context(record: logging.LogRecord) -> None:
    """Always stamp formatter fields. Missing ContextVar → '-' (engine thread)."""
    record.request_id = request_id_var.get() or "-"
    record.session_id = session_id_var.get() or "-"


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        _fill_context(record)
        return True


class ContextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if not getattr(record, "request_id", None):
            record.request_id = "-"
        if not getattr(record, "session_id", None):
            record.session_id = "-"
        return super().format(record)


def _install_record_factory() -> None:
    """Every LogRecord gets request_id, including engine / to_thread / children.

    A Filter on logger 'fpvscan.test' is not applied to records created on
    'fpvscan.test.parameters'; without this, %(request_id)s raises KeyError
    and journalctl looks like a crash after every successful PUT.
    """
    global _FACTORY_INSTALLED
    if _FACTORY_INSTALLED:
        return
    prev = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = prev(*args, **kwargs)
        _fill_context(record)
        return record

    logging.setLogRecordFactory(factory)
    _FACTORY_INSTALLED = True


def _ensure_filter(logger: logging.Logger) -> None:
    if not any(isinstance(f, ContextFilter) for f in logger.filters):
        logger.addFilter(ContextFilter())
    for handler in logger.handlers:
        if not any(isinstance(f, ContextFilter) for f in handler.filters):
            handler.addFilter(ContextFilter())
        fmt = handler.formatter
        if fmt is not None and not isinstance(fmt, ContextFormatter):
            handler.setFormatter(ContextFormatter(fmt._fmt, fmt.datefmt))


def configure_test_logging() -> None:
    _install_record_factory()
    logger = logging.getLogger("fpvscan.test")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(ContextFilter())
        handler.setFormatter(ContextFormatter(
            "%(asctime)s %(levelname)s %(name)s rid=%(request_id)s "
            "sid=%(session_id)s %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    _ensure_filter(logger)
    mgr = logging.Logger.manager.loggerDict
    for name, obj in list(mgr.items()):
        if isinstance(obj, logging.Logger) and name.startswith("fpvscan.test."):
            _ensure_filter(obj)
