from __future__ import annotations

import io
import logging

from fpvscan.app.core.logging import configure_test_logging, request_id_var


def test_parameters_log_without_extra_request_id() -> None:
    """PUT apply logs rid= in the message; formatter still needs record.request_id."""
    configure_test_logging()
    log = logging.getLogger("fpvscan.test.parameters")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("rid=%(request_id)s %(message)s"))
    log.addHandler(handler)
    try:
        log.info("parameters_applied keys=%s rid=%s", ["scan.fft_size"], None)
    finally:
        log.removeHandler(handler)
    text = buf.getvalue()
    assert "parameters_applied" in text
    assert "rid=" in text


def test_engine_thread_log_gets_dash_request_id() -> None:
    configure_test_logging()
    log = logging.getLogger("fpvscan.engine")
    rec = log.makeRecord(log.name, logging.INFO, __file__, 1, "lock tick", (), None)
    assert rec.request_id == "-"
    assert rec.session_id == "-"


def test_request_id_var_copied_onto_record() -> None:
    configure_test_logging()
    log = logging.getLogger("fpvscan.test.parameters")
    token = request_id_var.set("abc123")
    try:
        rec = log.makeRecord(log.name, logging.INFO, __file__, 1, "hi", (), None)
        assert rec.request_id == "abc123"
    finally:
        request_id_var.reset(token)
