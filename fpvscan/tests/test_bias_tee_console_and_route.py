"""Bias-T button and API token — a mismatch powers the LNA during LOCK."""
from __future__ import annotations

import inspect
from pathlib import Path

from fpvscan.web.server import create_app


def test_apply_state_toggles_bias_button_from_snapshot():
    """applyState must mirror snapshot.bias_tee onto #b-bias.on. If the
    class is only flipped locally on click, a heartbeat 2 s later
    disagrees with the radio and the next click sends the opposite
    of what the operator sees — during LOCK that is a control call
    racing the IQ reader.
    """
    html = (Path(__file__).resolve().parents[1]
            / "fpvscan" / "web" / "static" / "index.html").read_text(
                encoding="utf-8")
    start = html.index("function applyState")
    end = html.index("async function lock")
    apply_body = html[start:end]
    assert "b-bias" in apply_body
    assert "s.bias_tee" in apply_body
    assert "classList.toggle('on', !!s.bias_tee)" in apply_body


def test_bias_tee_route_only_exact_on_enables():
    """Console posts /api/bias_tee/on|off. Treating any truthy path
    token as on (including 'off') would apply the LNA gain offset
    when the operator meant to unpower it.
    """
    src = inspect.getsource(create_app)
    assert 'engine.command("bias_tee", on=(action == "on"))' in src


def test_bias_button_posts_on_or_off():
    html = (Path(__file__).resolve().parents[1]
            / "fpvscan" / "web" / "static" / "index.html").read_text(
                encoding="utf-8")
    assert "/api/bias_tee/' + (on ? 'on' : 'off')" in html
