from __future__ import annotations

from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from fpvscan.dsp import adaptive_if
from fpvscan.engine import Detection, Engine
from fpvscan.sweep_inspect import InspectResult
from fpvscan.web.server import create_app


CFG = {
    "scan": {
        "start_hz": 1.0e9,
        "stop_hz": 1.2e9,
        "sample_rate": 20e6,
        "channel_bw_hz": 12e6,
        "fft_size": 1024,
        "averages": 1,
        "priority_bands": False,
        "auto_peek": True,
        "confirm_hits": 1,
    },
    "video": {
        "sample_rate": 20e6,
        "channel_bw_hz": 12e6,
    },
    "rotator": {"enable": False},
    "sdr": {},
}


class _Source:
    name = "stub"
    sample_rate = 20e6
    overflows = 0
    clip_frac = 0.0
    adc_rms = 0.0
    bias_tee = False

    def retune_and_read(self, _freq_hz, samples):
        return np.zeros(max(8, int(samples)), dtype=np.complex64)


def _engine() -> Engine:
    return Engine(_Source(), CFG, Queue())


def _analog(engine: Engine, freq_hz: float, *, votes: int = 2) -> Detection:
    return Detection(
        freq_hz=freq_hz,
        bandwidth_hz=8e6,
        snr_db=18.0,
        confidence=0.4,
        first_seen=1.0,
        last_seen=1.0,
        analog_evidence=True,
        stage="analog_evidence",
        prominence_db=12.0,
        inspect_votes=votes,
        inspect_windows=max(votes, 1),
        sweep_generation=engine._sweep_generation,
    )


def _selection() -> adaptive_if.Selection:
    return adaptive_if.Selection(
        decimation=2,
        effective_bw_hz=9e6,
        cutoff_hz=4.5e6,
        offset_hz=0.0,
        score=0.5,
        line_rate_hz=15_625.0,
        line_prominence_db=12.0,
        row_corr=0.03,
        analog=True,
        reason="test analogue",
        generation=1,
        candidates_evaluated=2,
        confirmed=True,
        stable=True,
        windows_evaluated=2,
        winner_votes=2,
    )


def _inspect_result() -> InspectResult:
    return InspectResult(
        rf_candidate=True,
        analog_evidence=True,
        video_confirmed=False,
        stage="analog_evidence",
        rejected_reason="",
        standard="PAL",
        line_rate_hz=15_625.0,
        prominence_db=12.0,
        line_confidence=0.4,
        row_corr=0.03,
        picture_score=0.0,
        picture_lines=0,
        picture_locked=False,
        votes=2,
        windows=2,
        retention=1.0,
        vertical_detail=1.2,
        elapsed_ms=20.0,
        preview_samples=100,
        evaluations=2,
        center_offset_hz=0.0,
        selection=_selection(),
    )


def test_manual_sweep_retires_stale_lock_evidence_and_stays_sweep() -> None:
    engine = _engine()
    stale = _analog(engine, 1.1e9)
    engine.state.detections[round(stale.freq_hz / 50e3)] = stale
    engine._sweep_handoffs[round(stale.freq_hz / 50e3)] = (
        _selection(), 20e6, engine._sweep_generation,
    )
    engine._lock_good = object()
    engine._handle_command("lock", {"freq_hz": 1.1e9})

    engine._handle_command("sweep", {"auto_lock": False})
    engine._maybe_peek(stale)

    assert engine.state.mode == "SWEEP"
    assert engine.state.lock_target is None
    assert engine._sweep_auto_lock is False
    assert engine._sweep_handoffs == {}
    assert engine._lock_good is None


def test_old_generation_inspect_result_is_ignored(monkeypatch) -> None:
    engine = _engine()
    generation = engine._sweep_generation

    def finish_after_manual_sweep(*_args, **_kwargs):
        engine.command("sweep", auto_lock=False)
        return _inspect_result()

    monkeypatch.setattr(
        "fpvscan.engine.sweep_inspect.inspect_iq",
        finish_after_manual_sweep,
    )
    occ = SimpleNamespace(
        center_hz=1.1e9,
        peak_hz=1.1e9,
        bandwidth_hz=8e6,
        snr_db=18.0,
    )
    engine._inspect(None, 1.1e9, 20e6, occ)

    assert engine._sweep_generation == generation + 1
    assert engine.state.mode == "SWEEP"
    assert engine.state.detections == {}
    assert engine._sweep_handoffs == {}


def test_fresh_candidates_never_auto_lock_during_manual_hold() -> None:
    engine = _engine()
    engine._handle_command("sweep", {"auto_lock": False})

    for index in range(4):
        det = _analog(engine, 1.1e9 + index)
        engine._merge(det)

    assert engine.state.mode == "SWEEP"
    assert engine.state.lock_target is None
    assert engine.snapshot()["sweep_hold"] is True


def test_explicit_auto_sweep_hands_off_only_after_fresh_votes() -> None:
    engine = _engine()
    engine._handle_command("sweep", {"auto_lock": True})
    engine._merge(_analog(engine, 1.1e9, votes=1))
    assert engine.state.mode == "SWEEP"

    engine._merge(_analog(engine, 1.1e9, votes=2))
    assert engine.state.mode == "LOCK"
    assert engine.state.lock_target == 1.1e9
    assert engine.state.auto is True


def test_operator_lock_exits_manual_sweep_at_requested_frequency() -> None:
    engine = _engine()
    engine._handle_command("sweep", {"auto_lock": False})
    target = 5.123456e9

    engine._handle_command("lock", {"freq_hz": target, "force": True})

    assert engine.state.mode == "LOCK"
    assert engine.state.lock_target == target
    assert engine.state.tuned_hz == target
    assert engine.state.auto is False


def test_state_exposes_sweep_policy_and_generation() -> None:
    engine = _engine()
    initial = engine.snapshot()["sweep_generation"]
    engine._handle_command("sweep", {"auto_lock": False})

    state = engine.snapshot()
    assert state["sweep_auto_lock"] is False
    assert state["sweep_hold"] is True
    assert state["sweep_generation"] == initial + 1


def test_sweep_api_is_manual_by_default_and_auto_is_explicit() -> None:
    class _WebEngine:
        cfg = {"rotator": {"enable": False}}
        events = Queue()

        def __init__(self):
            self.commands = []

        def command(self, name, **kwargs):
            self.commands.append((name, kwargs))

    engine = _WebEngine()
    client = TestClient(create_app(engine))

    manual = client.post("/api/sweep")
    automatic = client.post("/api/sweep?auto_lock=1")

    assert manual.json() == {"ok": True, "sweep_auto_lock": False}
    assert automatic.json() == {"ok": True, "sweep_auto_lock": True}
    assert engine.commands == [
        ("sweep", {"auto_lock": False}),
        ("sweep", {"auto_lock": True}),
    ]


def test_analog_evidence_pal_survives_energy_merge() -> None:
    engine = _engine()
    gen = engine._sweep_generation
    energy = Detection(
        freq_hz=1.1e9,
        bandwidth_hz=8e6,
        snr_db=18.0,
        standard="?",
        stage="rf_candidate",
        analog_evidence=False,
        first_seen=1.0,
        last_seen=1.0,
        sweep_generation=gen,
    )
    analog = Detection(
        freq_hz=1.1e9 + 50e3,
        bandwidth_hz=8e6,
        snr_db=12.0,
        standard="PAL",
        stage="analog_evidence",
        analog_evidence=True,
        line_rate=15_625.0,
        first_seen=2.0,
        last_seen=2.0,
        sweep_generation=gen,
    )
    engine._merge(energy)
    engine._merge(analog)
    det = next(iter(engine.state.detections.values()))
    assert det.standard == "PAL"
    assert det.stage == "analog_evidence"
    assert det.analog_evidence is True
    published = engine._published_detections()
    assert published
    assert published[0]["standard"] == "PAL"
    assert published[0]["stage"] == "analog_evidence"

    engine2 = _engine()
    engine2._merge(analog)
    engine2._merge(energy)
    det2 = next(iter(engine2.state.detections.values()))
    assert det2.standard == "PAL"
    assert det2.stage == "analog_evidence"


def test_ui_sweep_button_requests_manual_hold() -> None:
    static = Path(__file__).resolve().parents[1] / "fpvscan" / "web" / "static"
    api_js = (static / "js" / "api.js").read_text(encoding="utf-8")
    app_js = (static / "js" / "app.js").read_text(encoding="utf-8")

    assert '"/api/sweep?auto_lock="' in api_js
    assert "engineSweep({ autoLock: false })" in app_js


def test_ui_parameter_ack_is_transaction_scoped_without_force_relock() -> None:
    static = Path(__file__).resolve().parents[1] / "fpvscan" / "web" / "static"
    api_js = (static / "js" / "api.js").read_text(encoding="utf-8")
    app_js = (static / "js" / "app.js").read_text(encoding="utf-8")
    start = app_js.index("function afterParamsApplied")
    end = app_js.index("\nfunction currentToolset", start)
    handler = app_js[start:end]

    assert "acknowledged_keys" in handler
    assert "request_seq" in handler
    assert "pendingAckSeq" in handler
    assert "engineLock(" not in handler
    assert 'key.startsWith("scan.")' in handler
    assert "_paramKeySeq" in api_js
    assert "transactionMatches" in api_js
    assert "(this._paramKeySeq.get(key) || 0) <= requestSeq" in api_js


def test_rotator_slider_sends_only_after_pointer_drag() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "fpvscan" / "web" / "static" / "js" / "app.js"
    ).read_text(encoding="utf-8")
    start = app_js.index("function setRotatorAzimuth")
    end = app_js.index("\nfunction fmtLockMs", start)
    handler = app_js[start:end]

    assert "if (rotDragging) return" in handler
    assert "setRotatorAzimuth(rotAz.value, true)" in app_js
