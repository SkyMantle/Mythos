from __future__ import annotations

import threading
import time
from pathlib import Path
from queue import Queue

from fastapi.testclient import TestClient

from fpvscan.engine import Detection, Engine
from fpvscan.rotator import (
    AntennaRotator,
    azimuth_from_travel,
    display_span_deg,
    pulse_us_from_travel,
    quantize_command,
    travel_from_azimuth,
)


CFG = {
    "scan": {
        "start_hz": 400e6,
        "stop_hz": 6000e6,
        "sample_rate": 35e6,
        "channel_bw_hz": 10e6,
        "step_hz": 0,
        "fft_size": 8192,
        "priority_bands": False,
    },
    "video": {
        "sample_rate": 20e6,
        "channel_bw_hz": 12e6,
        "lo_offset_hz": 0.0,
    },
}


class _StubSrc:
    name = "stub"
    sample_rate = 20e6
    overflows = 0
    clip_frac = 0.0
    bias_tee = False


USER_ROT = {
    "enable": True,
    "type": 2,
    "max_range": 220,
    "gpio_pin": 12,
    "pwm_chip": 0,
    "pwm_channel": 0,
    "frequency_hz": 50,
    "servo_min_pulse_width_us": 300,
    "servo_max_pulse_width_us": 2500,
    "speed": 1,
    "reverse": True,
    "azimuth_offset": 90,
    "step_deg": 90,
}


def _fake_chip(root: Path, chip: int = 0, channel: int = 0) -> Path:
    chip_d = root / f"pwmchip{chip}"
    pwm = chip_d / f"pwm{channel}"
    pwm.mkdir(parents=True)
    (chip_d / "npwm").write_text("1\n")
    (chip_d / "export").write_text("")
    (pwm / "period").write_text("0\n")
    (pwm / "duty_cycle").write_text("0\n")
    (pwm / "enable").write_text("0\n")
    return pwm


def test_left_origin_90_is_display_center() -> None:
    assert display_span_deg(90, 220) == 180.0
    kw = dict(max_range=220.0, offset=90.0, reverse=True)
    assert travel_from_azimuth(0, **kw) == 0.0
    assert travel_from_azimuth(90, **kw) == 90.0
    assert travel_from_azimuth(180, **kw) == 180.0
    assert abs(azimuth_from_travel(90, **kw) - 90.0) < 1e-6
    left = pulse_us_from_travel(
        0, max_range=220, min_us=300, max_us=2500, reverse=True)
    mid = pulse_us_from_travel(
        90, max_range=220, min_us=300, max_us=2500, reverse=True)
    right = pulse_us_from_travel(
        180, max_range=220, min_us=300, max_us=2500, reverse=True)
    assert abs(left - 2500) < 1e-6
    assert abs(mid - 1600) < 1e-6
    assert abs(right - 700) < 1e-6


def test_rotator_empty_npwm(tmp_path: Path) -> None:
    chip = tmp_path / "pwm" / "pwmchip0"
    chip.mkdir(parents=True)
    (chip / "npwm").write_text("0\n")
    rot = AntennaRotator(USER_ROT, sysfs_root=tmp_path / "pwm")
    st = rot.status()
    assert st["available"] is False
    assert "порожній" in st["reason"] or "overlay" in st["reason"]
    rot = AntennaRotator(USER_ROT, sysfs_root=tmp_path / "missing")
    st = rot.set_azimuth(90)
    assert st["enable"] is True
    assert st["available"] is False
    assert st["target_azimuth"] == 90.0
    assert st["center_azimuth"] == 90.0
    assert st["display_max"] == 180.0
    assert st["pulse_us"] == 1600.0
    assert "немає" in st["reason"] or "overlay" in st["reason"]


def test_rotator_sysfs_pwm_pulse(tmp_path: Path) -> None:
    root = tmp_path / "pwm"
    pwm = _fake_chip(root)
    rot = AntennaRotator(USER_ROT, sysfs_root=root)
    st = rot.set_azimuth(90)
    assert st["available"] is True
    assert st["moving"] is False
    assert (pwm / "enable").read_text().strip() == "1"
    period = int((pwm / "period").read_text().strip())
    duty = int((pwm / "duty_cycle").read_text().strip())
    assert period == 20_000_000  # 50 Hz
    assert duty == 1_600_000  # 1600 µs at 90° left-origin, reverse


def test_sharp_left_ramps_pwm_instead_of_jumping(
        tmp_path: Path, monkeypatch) -> None:
    t = [100.0]
    monkeypatch.setattr("fpvscan.rotator.time.monotonic", lambda: t[0])
    monkeypatch.setattr(
        "fpvscan.rotator.time.sleep", lambda s: t.__setitem__(0, t[0] + s))
    root = tmp_path / "pwm"
    pwm = _fake_chip(root)
    rot = AntennaRotator(USER_ROT, sysfs_root=root)
    rot.set_azimuth(90)
    pulses: list[float] = []
    orig = rot._apply_pwm_locked

    def watched(pulse_us: float) -> None:
        pulses.append(float(pulse_us))
        orig(pulse_us)

    rot._apply_pwm_locked = watched  # type: ignore[method-assign]
    left = pulse_us_from_travel(
        0.0, max_range=220.0, min_us=300.0, max_us=2500.0, reverse=True)
    st = rot.set_azimuth(0)
    assert st["target_azimuth"] == 0.0
    assert pulses[-1] == left
    assert len(pulses) > 8
    step = rot._pwm_step_us() + 1e-6
    assert all(
        abs(b - a) <= step for a, b in zip(pulses, pulses[1:])
    )
    duty = int((pwm / "duty_cycle").read_text().strip())
    assert duty == int(round(left * 1000.0))


def test_rotator_disabled_skips_sysfs(tmp_path: Path) -> None:
    cfg = {**USER_ROT, "enable": False}
    rot = AntennaRotator(cfg, sysfs_root=tmp_path / "pwm")
    st = rot.set_azimuth(0)
    assert st["available"] is False
    assert "вимкнено" in st["reason"]
    assert not (tmp_path / "pwm" / "pwmchip0").exists()


def test_rotator_step_90_and_busy(monkeypatch) -> None:
    t = [100.0]
    monkeypatch.setattr("fpvscan.rotator.time.monotonic", lambda: t[0])
    rot = AntennaRotator(USER_ROT, sysfs_root=Path("/no-pwm"))
    assert rot.status()["azimuth"] == 90.0
    st = rot.nudge(-90)
    assert st["target_azimuth"] == 0.0
    assert st["moving"] is True
    assert st["busy"] is True
    assert st["eta_ms"] > 0
    t[0] += 10.0
    done = rot.status()
    assert done["moving"] is False
    assert done["azimuth"] == 0.0
    rot.nudge(90)
    t[0] += 10.0
    rot.nudge(90)
    t[0] += 10.0
    far = rot.status()
    assert far["target_azimuth"] == 180.0
    hold = rot.nudge(90)
    assert hold["target_azimuth"] == 180.0


def test_quantize_1_and_91_not_snapped_to_90() -> None:
    kw = dict(current=90.0, span=180.0, grid_deg=90.0)
    assert quantize_command(**kw, azimuth=1) == 1.0
    assert quantize_command(**kw, azimuth=91) == 91.0
    assert quantize_command(**kw, step=1) == 91.0
    assert quantize_command(**kw, step=-1) == 89.0
    assert quantize_command(**kw, step=90) == 180.0
    assert quantize_command(**kw, step=-90) == 0.0
    assert quantize_command(current=37, span=180, grid_deg=90, step=90) == 90.0
    assert quantize_command(current=37, span=180, grid_deg=90, step=-90) == 0.0


def test_rotator_step_1_deg(monkeypatch) -> None:
    t = [100.0]
    monkeypatch.setattr("fpvscan.rotator.time.monotonic", lambda: t[0])
    rot = AntennaRotator(USER_ROT, sysfs_root=Path("/no-pwm"))
    st = rot.nudge(-1)
    assert st["target_azimuth"] == 89.0
    assert st["moving"] is True
    assert st["busy"] is True
    t[0] += 10.0
    assert rot.status()["azimuth"] == 89.0
    st = rot.nudge(1)
    assert st["target_azimuth"] == 90.0
    t[0] += 10.0
    st = rot.set_azimuth(1)
    assert st["target_azimuth"] == 1.0
    t[0] += 10.0
    st = rot.set_azimuth(91)
    assert st["target_azimuth"] == 91.0
    t[0] += 10.0
    st = rot.set_azimuth(37)
    assert st["target_azimuth"] == 37.0
    t[0] += 10.0
    assert rot.nudge(1)["target_azimuth"] == 38.0
    t[0] += 10.0
    assert rot.nudge(-90)["target_azimuth"] == 0.0


def test_lock_bw_caps_occupancy_at_config() -> None:
    cfg = {**CFG, "video": {**CFG["video"], "channel_bw_hz": 9e6}}
    eng = Engine(_StubSrc(), cfg, Queue())
    eng.state.detections[1] = Detection(
        freq_hz=3369.3e6, bandwidth_hz=16e6, snr_db=20.0,
    )
    assert eng._lock_bw(3369.3e6, 9e6) == 9e6
    assert eng.snapshot()["rotator"]["enable"] is False


def test_rotator_http_put(tmp_path: Path) -> None:
    from fpvscan.web.server import create_app

    root = tmp_path / "pwm"
    _fake_chip(root)
    rot = AntennaRotator(USER_ROT, sysfs_root=root)

    class _Eng:
        cfg = {"rotator": USER_ROT, "web": {"host": "0.0.0.0", "port": 8080}}
        events = Queue()
        rotator = rot

        class _NoSdrCalls:
            def __getattr__(self, name):
                if name.startswith("set_") or name in ("read", "open", "close"):
                    raise AssertionError(f"rotator invoked SDR method {name}")
                raise AttributeError(name)

        src = _NoSdrCalls()
        sdr_state = "OK"
        sdr_restarts = 0
        reader_restarts = 0
        mgc_holds: list[tuple[float, str]] = []
        guard_calls = 0
        guard_active = False
        order: list[str] = []

        def begin_motor_guard(self):
            self.guard_calls += 1
            self.guard_active = True
            self.order.append("guard")

        def hold_auto_mgc(self, seconds, *, reason="operator"):
            self.mgc_holds.append((float(seconds), str(reason)))

        def snapshot(self):
            return {
                "mode": "SWEEP",
                "engine_alive": True,
                "sdr_state": self.sdr_state,
                "sdr_restarts": self.sdr_restarts,
                "reader_restarts": self.reader_restarts,
                "rotator": self.rotator.status(),
            }

        def command(self, *_a, **_k):
            return None

        def _emit(self, *_a, **_k):
            return None

    eng = _Eng()
    original_set_azimuth = rot.set_azimuth

    def watched_set_azimuth(azimuth):
        assert eng.guard_active
        eng.order.append("pwm")
        return original_set_azimuth(azimuth)

    rot.set_azimuth = watched_set_azimuth
    client = TestClient(create_app(eng))
    got = client.get("/api/rotator")
    assert got.status_code == 200
    sdr_before = client.get("/api/state").json()
    body = got.json()
    assert body["gpio_pin"] == 12
    assert body["display_max"] == 180.0
    assert body["center_azimuth"] == 90.0
    put = client.put("/api/rotator", json={"azimuth": 90})
    assert put.status_code == 200
    body = put.json()
    assert body["target_azimuth"] == 90.0
    assert body["pulse_us"] == 1600.0
    step = client.put("/api/rotator", json={"step": -90})
    assert step.status_code == 200
    left = step.json()
    assert left["target_azimuth"] == 0.0
    assert left["pulse_us"] == pulse_us_from_travel(
        0.0, max_range=220.0, min_us=300.0, max_us=2500.0, reverse=True)
    # PWM ramps inside the request so the servo is not slammed; moving
    # may already be false when the HTTP response is built.
    deg = client.put("/api/rotator", json={"azimuth": 37})
    assert deg.status_code == 200
    assert deg.json()["target_azimuth"] == 37.0
    one = client.put("/api/rotator", json={"azimuth": 1})
    assert one.status_code == 200
    assert one.json()["target_azimuth"] == 1.0
    odd = client.put("/api/rotator", json={"azimuth": 91})
    assert odd.status_code == 200
    assert odd.json()["target_azimuth"] == 91.0
    live = client.get("/api/state")
    assert live.status_code == 200
    assert live.json()["rotator"]["target_azimuth"] == 91.0
    nudge = client.put("/api/rotator", json={"step": 1})
    assert nudge.status_code == 200
    assert nudge.json()["target_azimuth"] == 92.0
    back = client.put("/api/rotator", json={"step": -1})
    assert back.status_code == 200
    assert back.json()["target_azimuth"] == 91.0
    grid = client.put("/api/rotator", json={"step": 90})
    assert grid.status_code == 200
    assert grid.json()["target_azimuth"] == 180.0
    bad = client.put("/api/rotator", json={})
    assert bad.status_code == 400
    sdr_after = client.get("/api/state").json()
    assert {
        "sdr_state": sdr_after["sdr_state"],
        "sdr_restarts": sdr_after["sdr_restarts"],
        "reader_restarts": sdr_after["reader_restarts"],
    } == {
        "sdr_state": sdr_before["sdr_state"],
        "sdr_restarts": sdr_before["sdr_restarts"],
        "reader_restarts": sdr_before["reader_restarts"],
    }
    assert eng.guard_calls >= 1
    assert eng.order[:2] == ["guard", "pwm"]
    assert eng.mgc_holds == []


def test_lock_post_force_queues() -> None:
    from fpvscan.web.server import create_app

    class _Eng:
        cmds: list = []
        cfg = {"rotator": USER_ROT, "web": {"host": "0.0.0.0", "port": 8080}}
        events = Queue()
        rotator = AntennaRotator(USER_ROT, sysfs_root=Path("/no-pwm"))

        def snapshot(self):
            raise AssertionError("lock HTTP must not rebuild snapshot")

        def snapshot_json(self):
            return '{"mode":"SWEEP"}'

        def ws_state_json(self):
            return '{"type":"state","data":{"mode":"SWEEP"}}'

        def command(self, name, **kw):
            self.cmds.append((name, dict(kw)))

        def _emit(self, *_a, **_k):
            return None

    eng = _Eng()
    client = TestClient(create_app(eng))
    r = client.post("/api/lock/4988000000?force=1")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert eng.cmds == [("lock", {"freq_hz": 4988000000.0, "force": True})]
    cached = client.get("/api/state")
    assert cached.status_code == 200
    assert cached.json()["mode"] == "SWEEP"


def test_engine_snapshot_reads_cache_not_rebuild() -> None:
    cfg = {**CFG, "sdr": {"gain_db": 30, "auto_gain": True, "bias_tee": False}}
    eng = Engine(_StubSrc(), cfg, Queue())
    n = [0]
    real = eng._build_snapshot

    def counted():
        n[0] += 1
        return real()

    eng._build_snapshot = counted
    got = eng.snapshot()
    assert got["mode"] == "SWEEP"
    assert n[0] == 0
    assert '"mode"' in eng.snapshot_json()
    eng.publish_snapshot()
    assert n[0] == 1


def test_snapshot_overlays_live_rotator() -> None:
    cfg = {
        **CFG,
        "sdr": {"gain_db": 30, "auto_gain": True, "bias_tee": False},
        "rotator": USER_ROT,
    }
    eng = Engine(_StubSrc(), cfg, Queue())
    assert eng.snapshot()["rotator"]["target_azimuth"] == 90.0
    eng.rotator.set_azimuth(91)
    got = eng.snapshot()
    assert got["rotator"]["target_azimuth"] == 91.0
    with eng._pub_lock:
        frozen = eng._pub_snap["rotator"]["target_azimuth"]
    assert frozen == 90.0
    body = eng.snapshot_json()
    assert '"target_azimuth": 91' in body or '"target_azimuth":91' in body


def test_snapshot_overlays_live_video_not_cached_blob() -> None:
    """GET /api/state must not freeze pic_score while LOCK decode updates _last_video."""
    cfg = {**CFG, "sdr": {"gain_db": 30, "auto_gain": True, "bias_tee": False}}
    eng = Engine(_StubSrc(), cfg, Queue())
    assert eng.snapshot().get("video") is None
    eng._last_video = {
        "freq_hz": 3250e6,
        "pic_score": 0.41,
        "line_rate": 15625.0,
        "locked": True,
        "free_run": False,
        "row_corr": 0.32,
    }
    eng._fps_ema = 8.5
    eng._last_frame_ref = "out/photos/now.webp"
    got = eng.snapshot()
    assert got["video"]["pic_score"] == 0.41
    assert got["video"]["line_rate"] == 15625.0
    assert got["video"]["freq_hz"] == 3250e6
    assert got["fps"] == 8.5
    assert got["last_frame_ref"] == "out/photos/now.webp"
    with eng._pub_lock:
        assert eng._pub_snap["video"] is None
    eng.state.mode = "LOCK"
    eng.state.lock_target = 3410e6
    eng.refresh_lock()
    assert eng._last_video is None
    assert eng.snapshot()["video"] is None


def test_concurrent_plus_one_reaches_92() -> None:
    rot = AntennaRotator(USER_ROT, sysfs_root=Path("/no-pwm"))
    rot.set_azimuth(90)
    workers = [threading.Thread(target=rot.nudge, args=(1,)) for _ in range(2)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert rot.status()["target_azimuth"] == 92.0


def test_export_failure_retries(tmp_path: Path, monkeypatch) -> None:
    t = [100.0]
    monkeypatch.setattr("fpvscan.rotator.time.monotonic", lambda: t[0])
    monkeypatch.setattr(
        "fpvscan.rotator.time.sleep", lambda s: t.__setitem__(0, t[0] + s))
    root = tmp_path / "pwm"
    chip = root / "pwmchip0"
    chip.mkdir(parents=True)
    (chip / "npwm").write_text("1\n")
    (chip / "export").write_text("")
    rot = AntennaRotator(USER_ROT, sysfs_root=root)
    st = rot.set_azimuth(91)
    assert st["available"] is False
    pwm = chip / "pwm0"
    pwm.mkdir()
    (pwm / "period").write_text("0\n")
    (pwm / "duty_cycle").write_text("0\n")
    (pwm / "enable").write_text("0\n")
    st = rot.set_azimuth(91)
    assert st["available"] is True
    assert st["target_azimuth"] == 91.0


def test_stop_reader_keeps_alive_thread() -> None:
    cfg = {**CFG, "sdr": {"gain_db": 30, "auto_gain": True, "bias_tee": False}}
    eng = Engine(_StubSrc(), cfg, Queue())
    eng._sdr_reader_join_s = lambda: 0.12
    def hang():
        time.sleep(0.8)
    thread = threading.Thread(target=hang, daemon=True)
    eng._reader_thread = thread
    thread.start()
    try:
        eng._stop_reader()
        raise AssertionError("expected live SDR reader to stay attached")
    except RuntimeError:
        assert eng._reader_thread is thread
    thread.join(timeout=2)
    del eng._sdr_reader_join_s
    assert 4.0 <= eng._sdr_reader_join_s() < 6.0

