from __future__ import annotations

import importlib.util
import io
import json
import uuid
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "knob_sweep", ROOT / "scripts" / "knob_sweep.py",
)
assert _SPEC is not None and _SPEC.loader is not None
knob_sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(knob_sweep)

# Arbitrary Hz for URL / parse tests — not an operational analog channel.
ARB_HIGH_HZ = 5.1e9
ARB_LOW_HZ = 1.1e9
# Mock origin for run() tests that print --base. HttpClient is still mocked.
OPERATOR_BASE = knob_sweep.OPERATOR_BASE_HINT


@pytest.mark.parametrize(
    "raw, want",
    [
        ("1100", 1.1e9),
        ("1100e6", 1.1e9),
        ("1100000000", 1.1e9),
        ("5100.0", 5.1e9),
        ("5.1e9", 5.1e9),
        ("5100000000", 5.1e9),
    ],
)
def test_parse_freq_hz_mhz_vs_hz(raw: str, want: float) -> None:
    assert knob_sweep.parse_freq_hz(raw) == want


def test_parse_freq_hz_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        knob_sweep.parse_freq_hz("0")


def test_ofat_skips_live_value_and_scan_fft() -> None:
    current = {
        "video.capture_ms": 56,
        "video.afc": True,
        "scan.fft_size": 8192,
    }
    grid = {
        "video.capture_ms": [40, 56, 80],
        "video.afc": [True, False],
        "scan.fft_size": [1024, 8192],
        "scan.threshold_min_db": [1.0, 5.0],
        "scan.inspect_ms": [40, 90],
    }
    trials = knob_sweep.ofat_trials(current, grid)
    patches = [t["values"] for t in trials]
    assert patches[0] == {}
    assert {"video.capture_ms": 40} in patches
    assert {"video.capture_ms": 80} in patches
    assert {"video.capture_ms": 56} not in patches
    assert {"video.afc": False} in patches
    assert {"video.afc": True} not in patches
    for patch in patches:
        assert "scan.fft_size" not in patch
        assert "scan.threshold_min_db" not in patch
        assert "scan.inspect_ms" not in patch


def test_extract_metrics_overlays_last_video_when_state_video_empty() -> None:
    """GET /api/state video is engine._last_video; overlay if the key is raw."""
    hz = ARB_HIGH_HZ
    state = {
        "mode": "LOCK",
        "lock_target": hz,
        "tuned_hz": hz,
        "fps": 1.7,
        "afc_hz": 0,
        "video": None,
        "_last_video": {
            "pic_score": 0.41,
            "locked": True,
            "line_rate": 15625.0,
            "row_corr": 0.52,
            "lines": 288,
            "standard": "PAL",
            "freq_hz": hz,
            "free_run": False,
        },
    }
    m = knob_sweep.extract_metrics(state)
    assert m["pic_score"] == 0.41
    assert m["row_corr"] == 0.52
    assert m["line_rate"] == 15625.0
    assert m["locked"] is True
    assert m["free_run"] is False
    assert m["video_freq_hz"] == hz
    assert m["lines"] == 288
    # Live HTTP video wins over a stale _last_video dump.
    mixed = knob_sweep.extract_metrics({
        "mode": "LOCK",
        "lock_target": hz,
        "video": {"pic_score": 0.55, "locked": True, "line_rate": 15620.0,
                  "row_corr": 0.60, "freq_hz": hz},
        "_last_video": {"pic_score": 0.11, "locked": False, "line_rate": 14880.0,
                        "row_corr": 0.04, "freq_hz": hz},
    })
    assert mixed["pic_score"] == 0.55
    assert mixed["row_corr"] == 0.60


def test_extract_metrics_infers_free_run_from_unlocked_video() -> None:
    hz = ARB_HIGH_HZ
    state = {
        "mode": "LOCK",
        "lock_target": hz,
        "tuned_hz": hz,
        "fps": 4.2,
        "afc_hz": -12000,
        "video": {
            "pic_score": 0.11,
            "locked": False,
            "line_rate": 14880.0,
            "row_corr": 0.04,
            "standard": "?",
            "lines": 220,
        },
        "last_frame_ref": "out/photos/shot.webp",
    }
    live = {
        "lock_state": False,
        "lock_target_hz": hz,
        "video_metrics": {"fps": 4.2, "line_rate": 14880.0, "locked": False},
    }
    m = knob_sweep.extract_metrics(state, live)
    assert m["pic_score"] == 0.11
    assert m["locked"] is False
    assert m["free_run"] is True
    assert m["line_rate"] == 14880.0
    assert m["row_corr"] == 0.04
    assert m["fps"] == 4.2
    assert m["lock_target"] == hz

    locked = knob_sweep.extract_metrics({
        "mode": "LOCK",
        "lock_target": hz,
        "tuned_hz": hz,
        "video": {"pic_score": 0.55, "locked": True, "line_rate": 15625.0, "row_corr": 0.4},
    })
    assert locked["free_run"] is False
    assert locked["locked"] is True

    snow_hz = ARB_LOW_HZ
    snow = knob_sweep.extract_metrics({
        "mode": "LOCK",
        "lock_target": snow_hz,
        "tuned_hz": snow_hz,
        "video": {
            "freq_hz": snow_hz,
            "pic_score": 0.09,
            "locked": False,
            "free_run": True,
            "line_rate": 16363.4,
            "row_corr": 0.009,
        },
    })
    assert snow["free_run"] is True
    assert snow["video_freq_hz"] == snow_hz
    assert snow["pic_score"] == 0.09

    blank = knob_sweep.extract_metrics(
        {"mode": "LOCK", "lock_target": snow_hz, "tuned_hz": snow_hz},
    )
    assert blank["pic_score"] is None
    assert blank["free_run"] is None
    assert blank["video_freq_hz"] is None


def test_still_skip_note_does_not_call_locked_analog_snow() -> None:
    assert knob_sweep.still_skip_note(
        {"locked": True, "free_run": False, "row_corr": 0.938}, False,
    ) == "no new still (timeout or copy; lock was analog)"
    assert knob_sweep.still_skip_note(
        {"locked": False, "free_run": True, "row_corr": 0.009}, False,
    ) == "no new still (snow/free_run never writes)"
    assert knob_sweep.still_skip_note({"locked": True}, True) is None
    assert knob_sweep.still_wait_s(2.5, 0.23) >= 6.0
    assert knob_sweep.still_wait_s(2.5, 0.23) >= 2.5 / 0.23


def test_diagnose_state_explains_all_none() -> None:
    hz = ARB_LOW_HZ
    assert "no video key" in (knob_sweep.diagnose_state(
        {"mode": "LOCK", "lock_target": hz, "tuned_hz": hz}, hz,
    ) or "")
    text = knob_sweep.diagnose_state(
        {"mode": "SWEEP", "lock_target": None, "video": None}, hz,
    ) or ""
    assert "not LOCK" in text
    assert "no video key" in text
    assert knob_sweep.diagnose_state({
        "mode": "LOCK",
        "lock_target": hz,
        "video": {"pic_score": 0.4, "line_rate": 15625.0, "locked": True},
    }, hz) is None


def test_metrics_fingerprint_detects_stale_row() -> None:
    a = {"pic_score": 0.092, "line_rate": 16363.4, "locked": False, "row_corr": 0.009,
         "video_freq_hz": ARB_HIGH_HZ}
    b = dict(a)
    assert knob_sweep.metrics_fingerprint(a) == knob_sweep.metrics_fingerprint(b)
    b["pic_score"] = 0.41
    assert knob_sweep.metrics_fingerprint(a) != knob_sweep.metrics_fingerprint(b)


class _FakeResp:
    def __init__(self, status: int, payload: object) -> None:
        self.status = status
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_put_parameters_sends_uuid_and_treats_2xx_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResp(200, {
            "values": {"video.capture_ms": 80},
            "applied_keys": ["video.capture_ms"],
            "pending_keys": [],
        })

    monkeypatch.setattr(knob_sweep, "urlopen", fake_urlopen)
    http = knob_sweep.HttpClient("http://127.0.0.1:8080")
    out = knob_sweep.apply_parameters(http, {"video.capture_ms": 80})
    assert captured["method"] == "PUT"
    assert captured["url"] == "http://127.0.0.1:8080/api/test/parameters"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["values"] == {"video.capture_ms": 80}
    uuid.UUID(str(body["idempotency_key"]))  # valid UUID, new each call
    assert out["http_status"] == 200
    assert out["applied_keys"] == ["video.capture_ms"]


def test_http_2xx_not_raised_as_formatter_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """journalctl KeyError on request_id is not an HTTP failure."""

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        return _FakeResp(200, {"ok": True})

    monkeypatch.setattr(knob_sweep, "urlopen", fake_urlopen)
    http = knob_sweep.HttpClient("http://127.0.0.1:8080")
    status, data = http.request("POST", f"/api/lock/{int(ARB_HIGH_HZ)}?force=1")
    assert status == 200
    assert data == {"ok": True}


def test_http_error_body_still_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        raise HTTPError(
            req.full_url, 422, "unprocessable", hdrs=Message(), fp=io.BytesIO(
                json.dumps({"detail": {"message": "unknown parameter"}}).encode()
            ),
        )

    monkeypatch.setattr(knob_sweep, "urlopen", fake_urlopen)
    http = knob_sweep.HttpClient("http://127.0.0.1:8080")
    with pytest.raises(knob_sweep.SweepHttpError) as ei:
        http.request("PUT", "/api/test/parameters", {"idempotency_key": "x", "values": {}})
    assert ei.value.status == 422


def test_default_lock_grid_has_analog_if_not_fft() -> None:
    keys = set(knob_sweep.DEFAULT_LOCK_GRID)
    assert {
        "video.channel_bw_hz", "video.deviation_hz", "video.capture_ms",
        "video.sample_rate", "video.afc", "video.afc_deadband_hz",
        "video.afc_digital_max_hz",
        "video.h_pll", "video.h_phase_frac", "video.average",
        "video.sharpen", "video.width", "video.crop_left_frac",
        "video.crop_bottom_lines", "video.track_window_margin",
    } <= keys
    assert not any(k.startswith("scan.") for k in keys)
    assert "scan.fft_size" not in keys
    assert "scan.averages" not in keys
    assert "scan.threshold_min_db" not in keys
    assert "scan.dc_notch_hz" not in keys
    assert "scan.inspect_ms" not in keys
    assert "video.h_pll" in knob_sweep.DECODE_GRID
    assert "video.average" in knob_sweep.CVBS_GRID
    help_txt = knob_sweep.build_parser().format_help()
    assert "--group" in help_txt
    assert "decode" in help_txt


def test_skip_prefixes_still_exclude_scan() -> None:
    assert knob_sweep.is_skipped_key("scan.fft_size")
    assert knob_sweep.is_skipped_key("scan.averages")
    assert knob_sweep.is_skipped_key("scan.threshold_min_db")
    assert knob_sweep.is_skipped_key("scan.dc_notch_hz")
    assert knob_sweep.is_skipped_key("scan.inspect_ms")
    assert knob_sweep.is_skipped_key("scan.inspect_decode")
    assert not knob_sweep.is_skipped_key("video.h_pll")
    assert not knob_sweep.is_skipped_key("video.channel_bw_hz")
    for key in knob_sweep.DEFAULT_LOCK_GRID:
        assert not knob_sweep.is_skipped_key(key)


# Typical live IF leftovers (12e6 BW) + yaml picture defaults.
# Live duplicates are skipped so default high-band OFAT is ~30, not 19.
STOCK_LIVE = {
    "video.channel_bw_hz": 12e6,
    "video.deviation_hz": 4.8e6,
    "video.capture_ms": 56,
    "video.sample_rate": 20e6,
    "video.afc": True,
    "video.afc_deadband_hz": 80e3,
    "video.afc_digital_max_hz": 1.5e6,
    "video.h_pll": False,
    "video.h_phase_frac": 0.0,
    "video.average": 5,
    "video.sharpen": 0.5,
    "video.width": 640,
    "video.crop_left_frac": 0.07,
    "video.crop_bottom_lines": 6,
    "video.track_window_margin": 1.45,
    "video.auto_levels": True,
    "video.motion_thresh": 24.0,
}


def _lock_state(freq_hz: float = ARB_HIGH_HZ) -> dict:
    return {
        "mode": "LOCK",
        "lock_target": freq_hz,
        "tuned_hz": freq_hz,
        "fps": 5.0,
        "afc_hz": 0,
        "video": {
            "pic_score": 0.2,
            "locked": True,
            "line_rate": 15625.0,
            "row_corr": 0.3,
            "lines": 288,
            "standard": "PAL",
            "freq_hz": freq_hz,
            "free_run": False,
        },
        "last_frame_ref": "out/photos/x.webp",
    }


LOCK_STATE = _lock_state()


def test_best_trial_patch_picks_highest_row_corr() -> None:
    rows = [
        {"applied_values": "", "row_corr": 0.113, "pic_score": 0.262},
        {"applied_values": '{"video.channel_bw_hz": 16000000.0}',
         "row_corr": 0.649, "pic_score": 0.557},
        {"applied_values": '{"video.deviation_hz": 10000000.0}',
         "row_corr": 0.004, "pic_score": 0.091},
        {"applied_values": '{"video.capture_ms": 40}',
         "row_corr": 0.059, "pic_score": 0.105, "error": "x"},
    ]
    assert knob_sweep.best_trial_patch(rows) == {"video.channel_bw_hz": 16000000.0}
    assert knob_sweep.best_trial_patch([]) is None


@pytest.mark.parametrize("freq_hz", [ARB_LOW_HZ, ARB_HIGH_HZ])
def test_lock_target_drifted_half_mhz(freq_hz: float) -> None:
    assert not knob_sweep.lock_target_drifted(freq_hz, freq_hz)
    assert not knob_sweep.lock_target_drifted(freq_hz + 80e3, freq_hz)
    assert knob_sweep.lock_target_drifted(freq_hz - 1e6, freq_hz)
    assert knob_sweep.lock_target_drifted(freq_hz + 1e6, freq_hz)
    assert not knob_sweep.lock_target_drifted(None, freq_hz)
    assert not knob_sweep.lock_target_drifted("x", freq_hz)


@pytest.mark.parametrize("freq_hz", [ARB_LOW_HZ, ARB_HIGH_HZ])
def test_relock_if_drifted_posts_force(
    monkeypatch: pytest.MonkeyPatch, freq_hz: float,
) -> None:
    captured: list[str] = []

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        captured.append(req.full_url)
        return _FakeResp(200, {"ok": True})

    monkeypatch.setattr(knob_sweep, "urlopen", fake_urlopen)
    http = knob_sweep.HttpClient("http://127.0.0.1:8080")
    assert knob_sweep.relock_if_drifted(http, freq_hz, freq_hz) is None
    assert captured == []
    walked = freq_hz - 1e6
    note = knob_sweep.relock_if_drifted(http, freq_hz, walked)
    assert note is not None
    assert f"re-locked {freq_hz / 1e6:.3f} MHz" in note
    assert f"{walked / 1e6:.3f}" in note
    hz = int(round(freq_hz))
    assert captured == [f"http://127.0.0.1:8080/api/lock/{hz}?force=1"]


@pytest.mark.parametrize("freq_hz", [ARB_LOW_HZ, ARB_HIGH_HZ])
def test_post_lock_appends_force_query(
    monkeypatch: pytest.MonkeyPatch, freq_hz: float,
) -> None:
    captured: dict[str, str] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        return _FakeResp(200, {"ok": True})

    monkeypatch.setattr(knob_sweep, "urlopen", fake_urlopen)
    http = knob_sweep.HttpClient("http://127.0.0.1:8080")
    out = knob_sweep.post_lock(http, freq_hz, force=True)
    hz = int(round(freq_hz))
    assert captured["method"] == "POST"
    assert captured["url"] == f"http://127.0.0.1:8080/api/lock/{hz}?force=1"
    assert out["http_status"] == 200
    knob_sweep.post_lock(http, freq_hz, force=False)
    assert captured["url"] == f"http://127.0.0.1:8080/api/lock/{hz}"


def test_keep_best_flag_dest() -> None:
    ns = knob_sweep.build_parser().parse_args(["--freq", "5.1e9", "--keep-best"])
    assert ns.keep_best is True
    ns2 = knob_sweep.build_parser().parse_args(["--freq", "5.1e9"])
    assert ns2.keep_best is False


def test_save_frames_flag_dest() -> None:
    ns = knob_sweep.build_parser().parse_args(["--freq", "5.1e9", "--save-frames"])
    assert ns.save_frames is True
    ns2 = knob_sweep.build_parser().parse_args(["--freq", "5.1e9"])
    assert ns2.save_frames is False
    ns3 = knob_sweep.build_parser().parse_args(
        ["--freq", "5.1e9", "--out", "~/knob-sweep/", "--save-frames"],
    )
    assert ns3.save_frames is True
    assert ns.base is None
    # Operator paste wrap used to split --save-frames; accept the alias.
    ns4 = knob_sweep.build_parser().parse_args(["--freq", "5.1e9", "--save-frame"])
    assert ns4.save_frames is True


def test_read_web_bind_wildcard_host_and_port(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "sdr:\n  driver: bladerf\nweb:\n  host: 0.0.0.0  # all ifaces\n  port: 8080\n",
        encoding="utf-8",
    )
    host, port = knob_sweep.read_web_bind(cfg)
    assert host == "0.0.0.0"
    assert port == 8080


def test_list_base_candidates_env_config_and_loopback(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("web:\n  host: 0.0.0.0\n  port: 9090\n", encoding="utf-8")
    env_first = knob_sweep.list_base_candidates(
        env_base="http://127.0.0.1:7777",
        config_paths=[cfg],
        discover_local=False,
    )
    assert env_first[0] == "http://127.0.0.1:7777"
    assert "http://127.0.0.1:9090" in env_first
    # Wildcard bind → client uses loopback, not 0.0.0.0
    assert all("0.0.0.0" not in u for u in env_first)

    explicit = knob_sweep.list_base_candidates(
        explicit="http://10.1.2.3:8080",
        env_base="http://127.0.0.1:7777",
        config_paths=[cfg],
        discover_local=False,
    )
    assert explicit[0] == "http://10.1.2.3:8080"

    lan = tmp_path / "lan.yaml"
    lan.write_text("web:\n  host: 10.9.8.7\n  port: 8080\n", encoding="utf-8")
    lan_cands = knob_sweep.list_base_candidates(
        config_paths=[lan], discover_local=False,
    )
    assert lan_cands[0] == "http://127.0.0.1:8080"
    assert "http://10.9.8.7:8080" in lan_cands

    zt = knob_sweep.list_base_candidates(
        config_paths=[cfg],
        local_hosts=["10.252.65.47"],
        discover_local=False,
    )
    assert zt[0] == "http://127.0.0.1:9090"
    assert "http://10.252.65.47:9090" in zt
    assert all("0.0.0.0" not in u for u in zt)


def test_unreachable_message_says_pi_service_not_windows() -> None:
    err = knob_sweep.SweepHttpError(
        0, "/api/health", {"error": "[Errno 111] Connection refused"},
    )
    assert knob_sweep.is_unreachable(err)
    msg = knob_sweep.format_unreachable(["http://127.0.0.1:8080"], err)
    assert "Pi service locally, not to your PC" in msg
    assert "systemctl status fpvscan" in msg
    assert "ss -lntp" in msg
    assert "Windows-localhost" in msg
    assert "errno 111" in msg
    assert "--base http://10.252.65.47:8080" in msg
    assert "Tried: http://127.0.0.1:8080" in msg
    not_conn = knob_sweep.SweepHttpError(404, "/api/health", {"detail": "missing"})
    assert not knob_sweep.is_unreachable(not_conn)


def test_run_connection_refused_is_service_down_not_windows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        raise URLError(OSError(111, "Connection refused"))

    monkeypatch.setattr(knob_sweep, "urlopen", fake_urlopen)
    monkeypatch.delenv("FPVSCAN_BASE", raising=False)
    ns = knob_sweep.build_parser().parse_args([
        "--freq", "5.1e9", "--out", str(tmp_path),
        "--install-root", str(tmp_path),
        "--base", "http://127.0.0.1:8080",
        "--save-frames",
    ])
    assert ns.save_frames is True
    rc = knob_sweep.run(ns)
    assert rc == 2
    err = capsys.readouterr().err
    assert "Pi service locally, not to your PC" in err
    assert "systemctl status fpvscan" in err
    assert "127.0.0.1:8080" in err
    assert "errno 111" in err.lower() or "Errno 111" in err
    assert "--base http://10.252.65.47:8080" in err


def test_default_lock_grid_is_about_thirty_trials() -> None:
    trials = knob_sweep.ofat_trials(STOCK_LIVE, knob_sweep.DEFAULT_LOCK_GRID)
    assert 28 <= len(trials) <= 32
    assert trials[0]["values"] == {}
    keys = {next(iter(t["values"])) for t in trials if t["values"]}
    assert "video.h_pll" in keys
    assert "video.h_phase_frac" in keys
    assert "video.average" in keys
    assert "video.sharpen" in keys
    assert "video.width" in keys
    assert "video.crop_left_frac" in keys
    assert "video.crop_bottom_lines" in keys
    assert "video.track_window_margin" in keys
    assert "video.deviation_hz" in keys
    assert "scan.fft_size" not in keys
    assert "scan.averages" not in keys
    yaml_live = dict(STOCK_LIVE)
    yaml_live["video.channel_bw_hz"] = 16e6
    yaml_n = len(knob_sweep.ofat_trials(yaml_live, knob_sweep.DEFAULT_LOCK_GRID))
    assert 28 <= yaml_n <= 32


# Picture extras off — not a named analog channel. OFAT must still try the on/grid values.
BARE_LIVE = {
    **STOCK_LIVE,
    "video.afc": False,
    "video.h_pll": False,
    "video.average": 0,
    "video.sharpen": 0.0,
    "video.crop_left_frac": 0.0,
    "video.crop_bottom_lines": 0,
}


@pytest.mark.parametrize("freq_hz", [ARB_LOW_HZ, ARB_HIGH_HZ])
def test_ofat_bare_settings_still_sweeps_extras(freq_hz: float) -> None:
    """afc/h_pll/average/sharpen/crop off: skip live zeros, still OFAT the grid."""
    grid = knob_sweep.apply_band_extras(dict(knob_sweep.DEFAULT_LOCK_GRID), freq_hz)
    trials = knob_sweep.ofat_trials(BARE_LIVE, grid)
    patches = [t["values"] for t in trials]
    assert patches[0] == {}
    assert {"video.afc": True} in patches
    assert {"video.afc": False} not in patches
    assert {"video.h_pll": True} in patches
    assert {"video.h_pll": False} not in patches
    assert {"video.average": 0} not in patches
    assert {"video.average": 5} in patches
    assert {"video.average": 12} in patches
    assert {"video.sharpen": 0.0} not in patches
    assert {"video.sharpen": 0.5} in patches
    assert {"video.crop_left_frac": 0.0} not in patches
    assert {"video.crop_left_frac": 0.07} in patches
    assert {"video.crop_bottom_lines": 0} not in patches
    assert {"video.crop_bottom_lines": 6} in patches
    assert not any("scan.fft_size" in p for p in patches)


def test_luma_stats_never_raises_on_pil_or_shadowed_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert knob_sweep.luma_stats(tmp_path / "missing.webp") == (None, None)

    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "logging.py").write_text(
        "raise AttributeError(\"module 'logging' has no attribute 'Filter'\")\n",
        encoding="utf-8",
    )
    pil_dir = shadow / "PIL"
    pil_dir.mkdir()
    (pil_dir / "__init__.py").write_text(
        "import logging\nclass Image:\n    pass\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(knob_sweep.sys, "path", [str(shadow), ""] + list(knob_sweep.sys.path))
    assert knob_sweep.luma_stats(tmp_path / "x.webp") == (None, None)


def _ok_urlopen(req, timeout=None):  # noqa: ANN001
    url = req.full_url
    method = req.get_method()
    if url.endswith("/api/health"):
        return _FakeResp(200, {"ok": True})
    if url.endswith("/api/test/parameters/current"):
        return _FakeResp(200, {"values": dict(STOCK_LIVE)})
    if "/api/lock/" in url:
        return _FakeResp(200, {"ok": True})
    if url.endswith("/api/test/parameters") and method == "PUT":
        return _FakeResp(200, {"applied_keys": [], "pending_keys": [], "values": dict(STOCK_LIVE)})
    if url.endswith("/api/state"):
        return _FakeResp(200, dict(LOCK_STATE))
    if url.endswith("/api/test/live"):
        return _FakeResp(200, {
            "mode": "LOCK",
            "lock_target_hz": ARB_HIGH_HZ,
            "video_metrics": {"fps": 5.0, "locked": True, "line_rate": 15625.0},
        })
    if url.endswith("/api/snapshot"):
        return _FakeResp(200, {"ok": True})
    return _FakeResp(404, {"detail": url})


def test_run_writes_rows_even_if_luma_crashes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    """Operator bug: N trials printed, JSONL had 0 rows after luma/PIL crash."""
    expected = len(knob_sweep.ofat_trials(STOCK_LIVE, knob_sweep.DEFAULT_LOCK_GRID))
    assert 28 <= expected <= 32
    monkeypatch.setattr(knob_sweep, "urlopen", _ok_urlopen)
    monkeypatch.setattr(knob_sweep.time, "sleep", lambda _s: None)
    monkeypatch.delenv("FPVSCAN_BASE", raising=False)

    def boom(_path: Path) -> tuple[float | None, float | None]:
        raise AttributeError("module 'logging' has no attribute 'Filter'")

    monkeypatch.setattr(knob_sweep, "luma_stats", boom)
    monkeypatch.setattr(
        knob_sweep, "maybe_save_frame",
        lambda *_a, **_k: tmp_path / "shot.webp",
    )

    ns = knob_sweep.build_parser().parse_args([
        "--freq", "5.1e9",
        "--out", str(tmp_path / "out"),
        "--install-root", str(tmp_path),
        "--base", OPERATOR_BASE,
        "--save-frames",
        "--dwell-ms", "1",
        "--fresh-ms", "0",
        "--no-stop-on-picture",
    ])
    rc = knob_sweep.run(ns)
    assert rc == 0
    captured = capsys.readouterr()
    jsonl = tmp_path / "out" / "knob_sweep.jsonl"
    csv_path = tmp_path / "out" / "knob_sweep.csv"
    assert jsonl.is_file()
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == expected
    assert csv_path.is_file()
    assert f"{expected} trials" in captured.out
    assert OPERATOR_BASE in captured.out
    assert captured.out.count("[") >= expected
    wrote_at = captured.out.find("wrote")
    restored_at = captured.out.find("restored baseline")
    corr_at = captured.out.find("knob correlation")
    assert wrote_at != -1 and restored_at != -1
    assert wrote_at < restored_at
    assert corr_at != -1
    assert wrote_at < corr_at
    assert f"{expected} rows" in captured.out
    assert "knob correlation" in captured.out
    assert "not applied" in captured.out


def test_run_fails_loud_when_lock_stays_sweep(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        url = req.full_url
        method = req.get_method()
        if url.endswith("/api/health"):
            return _FakeResp(200, {"ok": True})
        if url.endswith("/api/test/parameters/current"):
            return _FakeResp(200, {"values": dict(STOCK_LIVE)})
        if "/api/lock/" in url:
            return _FakeResp(200, {"ok": True})
        if url.endswith("/api/test/parameters") and method == "PUT":
            return _FakeResp(200, {"applied_keys": [], "pending_keys": [], "values": dict(STOCK_LIVE)})
        if url.endswith("/api/state"):
            return _FakeResp(200, {"mode": "SWEEP", "lock_target": None, "tuned_hz": 0, "fps": 0})
        if url.endswith("/api/test/live"):
            return _FakeResp(200, {"mode": "SWEEP", "lock_target_hz": None, "video_metrics": {}})
        return _FakeResp(404, {"detail": url})

    monkeypatch.setattr(knob_sweep, "urlopen", fake_urlopen)
    monkeypatch.setattr(knob_sweep.time, "sleep", lambda _s: None)
    monkeypatch.delenv("FPVSCAN_BASE", raising=False)
    ns = knob_sweep.build_parser().parse_args([
        "--freq", "5.1e9",
        "--out", str(tmp_path / "out"),
        "--install-root", str(tmp_path),
        "--base", "http://127.0.0.1:8080",
        "--dwell-ms", "1",
        "--fresh-ms", "0",
        "--no-restore",
        "--no-stop-on-picture",
    ])
    rc = knob_sweep.run(ns)
    assert rc == 2
    captured = capsys.readouterr()
    err = captured.err
    assert "did not enter LOCK" in err
    assert "5100" in err
    # One failed row still prints the named block (empty ranked).
    assert "knob correlation" in captured.out


def test_run_keep_best_puts_winner_not_baseline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    puts: list[dict] = []
    live = dict(STOCK_LIVE)

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        url = req.full_url
        method = req.get_method()
        if url.endswith("/api/health"):
            return _FakeResp(200, {"ok": True})
        if url.endswith("/api/test/parameters/current"):
            return _FakeResp(200, {"values": dict(STOCK_LIVE)})
        if "/api/lock/" in url:
            assert "?force=1" in url
            assert method == "POST"
            return _FakeResp(200, {"ok": True})
        if url.endswith("/api/test/parameters") and method == "PUT":
            body = json.loads(req.data.decode("utf-8"))
            uuid.UUID(str(body["idempotency_key"]))
            vals = dict(body.get("values") or {})
            puts.append(vals)
            live.update(vals)
            return _FakeResp(200, {
                "applied_keys": list(vals), "pending_keys": [], "values": dict(live),
            })
        video = dict(LOCK_STATE["video"])
        bw = live.get("video.channel_bw_hz", 16e6)
        if abs(float(bw) - 8e6) < 1.0:
            video["row_corr"] = 0.81
            video["pic_score"] = 0.70
        state = dict(LOCK_STATE)
        state["video"] = video
        if url.endswith("/api/state"):
            return _FakeResp(200, state)
        if url.endswith("/api/test/live"):
            return _FakeResp(200, {
                "mode": "LOCK",
                "lock_target_hz": ARB_HIGH_HZ,
                "video_metrics": {"fps": 5.0, "locked": True, "line_rate": 15625.0},
            })
        return _FakeResp(404, {"detail": url})

    monkeypatch.setattr(knob_sweep, "urlopen", fake_urlopen)
    monkeypatch.setattr(knob_sweep.time, "sleep", lambda _s: None)
    monkeypatch.delenv("FPVSCAN_BASE", raising=False)
    ns = knob_sweep.build_parser().parse_args([
        "--freq", "5.1e9",
        "--out", str(tmp_path / "out"),
        "--install-root", str(tmp_path),
        "--base", "http://127.0.0.1:8080",
        "--dwell-ms", "1",
        "--fresh-ms", "0",
        "--keep-best",
        "--no-stop-on-picture",
    ])
    rc = knob_sweep.run(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "kept best trial knobs" in out
    assert "restored baseline knobs" not in out
    assert puts, "expected live PUT /api/test/parameters"
    assert any(abs(float(p.get("video.channel_bw_hz", 0)) - 8e6) < 1.0 for p in puts)
    final = puts[-1]
    assert abs(float(final.get("video.channel_bw_hz", 0)) - 8e6) < 1.0


def test_run_relocks_when_afc_walks_off_freq(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    """AFC digital_max 0.5e6 walks lock_target; decode rows must re-POST --freq."""
    lock_posts: list[str] = []
    live = dict(STOCK_LIVE)
    want_hz = ARB_HIGH_HZ
    lock_hz = want_hz

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        nonlocal lock_hz
        url = req.full_url
        method = req.get_method()
        if url.endswith("/api/health"):
            return _FakeResp(200, {"ok": True})
        if url.endswith("/api/test/parameters/current"):
            return _FakeResp(200, {"values": dict(STOCK_LIVE)})
        if "/api/lock/" in url:
            lock_posts.append(url)
            # Path is /api/lock/<hz> or /api/lock/<hz>?force=1
            tail = url.rsplit("/", 1)[-1].split("?", 1)[0]
            lock_hz = float(tail)
            return _FakeResp(200, {"ok": True})
        if url.endswith("/api/test/parameters") and method == "PUT":
            body = json.loads(req.data.decode("utf-8"))
            vals = dict(body.get("values") or {})
            live.update(vals)
            cap = live.get("video.afc_digital_max_hz", 1.5e6)
            if abs(float(cap) - 0.5e6) < 1.0:
                lock_hz = want_hz - 2e6
            return _FakeResp(200, {
                "applied_keys": list(vals), "pending_keys": [], "values": dict(live),
            })
        video = dict(LOCK_STATE["video"])
        video["freq_hz"] = lock_hz
        state = dict(LOCK_STATE)
        state["lock_target"] = lock_hz
        state["tuned_hz"] = lock_hz
        state["video"] = video
        if url.endswith("/api/state"):
            return _FakeResp(200, state)
        if url.endswith("/api/test/live"):
            return _FakeResp(200, {
                "mode": "LOCK",
                "lock_target_hz": lock_hz,
                "video_metrics": {"fps": 5.0, "locked": True, "line_rate": 15625.0},
            })
        return _FakeResp(404, {"detail": url})

    monkeypatch.setattr(knob_sweep, "urlopen", fake_urlopen)
    monkeypatch.setattr(knob_sweep.time, "sleep", lambda _s: None)
    monkeypatch.delenv("FPVSCAN_BASE", raising=False)
    ns = knob_sweep.build_parser().parse_args([
        "--freq", "5.1e9",
        "--out", str(tmp_path / "out"),
        "--install-root", str(tmp_path),
        "--base", "http://127.0.0.1:8080",
        "--dwell-ms", "1",
        "--fresh-ms", "0",
        "--no-restore",
        "--no-stop-on-picture",
    ])
    rc = knob_sweep.run(ns)
    assert rc == 0
    want = f"http://127.0.0.1:8080/api/lock/{int(round(want_hz))}?force=1"
    assert lock_posts[0] == want
    assert lock_posts.count(want) >= 2
    jsonl = tmp_path / "out" / "knob_sweep.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
    notes = " ".join(str(r.get("note") or "") for r in rows)
    assert f"re-locked {want_hz / 1e6:.3f} MHz after AFC walk" in notes
    out = capsys.readouterr().out
    assert "trials" in out


def test_correlate_knobs_ranks_synthetic_delta() -> None:
    """Given these OFAT patches, rank key by delta — not a named bird."""
    rows = [
        {"applied_values": "", "row_corr": 0.20, "pic_score": 0.30,
         "locked": True, "line_rate": 15625.0},
        {"applied_values": '{"video.deviation_hz": 3500000.0}',
         "row_corr": 0.10, "pic_score": 0.12, "locked": False, "line_rate": 15625.0},
        {"applied_values": '{"video.deviation_hz": 8000000.0}',
         "row_corr": 0.90, "pic_score": 0.80, "locked": True, "line_rate": 15625.0},
        {"applied_values": '{"video.capture_ms": 40}',
         "row_corr": 0.40, "pic_score": 0.35, "locked": True, "line_rate": 15625.0},
        {"applied_values": '{"video.capture_ms": 100}',
         "row_corr": 0.05, "pic_score": 0.09, "locked": False, "line_rate": 15625.0},
    ]
    ranked = knob_sweep.correlate_knobs(rows)
    assert [item["key"] for item in ranked][0] == "video.deviation_hz"
    by_key = {item["key"]: item for item in ranked}
    assert "" not in by_key
    assert "baseline" not in by_key
    assert abs(float(by_key["video.deviation_hz"]["best_value"]) - 8e6) < 1.0
    assert abs(float(by_key["video.deviation_hz"]["worst_value"]) - 3.5e6) < 1.0
    assert by_key["video.deviation_hz"]["delta"] == pytest.approx(0.80)
    assert by_key["video.capture_ms"]["delta"] == pytest.approx(0.35)
    deltas = [item["delta"] for item in ranked]
    assert deltas == sorted(deltas, reverse=True)
    text = knob_sweep.format_correlation(ranked)
    assert "not applied" in text
    assert "video.deviation_hz" in text
    assert knob_sweep.best_trial_patch(rows) == {"video.deviation_hz": 8000000.0}
    empty = knob_sweep.format_correlation([])
    assert "knob correlation" in empty
    assert "none" in empty.lower()


def test_correlate_skips_empty_video_and_multi_key() -> None:
    rows = [
        {"applied_values": "", "row_corr": 0.10, "pic_score": 0.20,
         "locked": False, "line_rate": 15625.0},
        {"applied_values": '{"video.sample_rate": 13000000.0}',
         "row_corr": 0.58, "pic_score": 0.65, "locked": True, "line_rate": 15625.0},
        {"applied_values": '{"video.channel_bw_hz": 10000000.0, '
                          '"video.sample_rate": 13000000.0}',
         "row_corr": 0.99, "pic_score": 0.99, "locked": True, "line_rate": 15625.0},
        {"applied_values": '{"video.afc_deadband_hz": 160000.0}',
         "row_corr": None, "pic_score": None, "line_rate": None,
         "note": "no video after dwell (decode never published)"},
        {"applied_values": '{"video.capture_ms": 100}',
         "row_corr": 0.02, "pic_score": None, "line_rate": None,
         "note": "no video key"},
    ]
    ranked = knob_sweep.correlate_knobs(rows)
    assert [item["key"] for item in ranked] == ["video.sample_rate"]
    # Correlation ignores multi-key; keep-best may still pick that patch.
    assert knob_sweep.best_trial_patch(rows) == {
        "video.channel_bw_hz": 10000000.0, "video.sample_rate": 13000000.0,
    }
    assert knob_sweep.analog_winner_patch(rows) == {
        "video.channel_bw_hz": 10000000.0, "video.sample_rate": 13000000.0,
    }


@pytest.mark.parametrize(
    "freq_hz, expect_narrow",
    [
        (1.1e9, True),
        (1.9e9, True),
        (2.0e9, False),
        (5.1e9, False),
    ],
)
def test_band_extras_use_generic_2ghz_threshold(
    freq_hz: float, expect_narrow: bool,
) -> None:
    grid = knob_sweep.apply_band_extras(
        dict(knob_sweep.DEFAULT_LOCK_GRID), freq_hz,
    )
    has_narrow = any(float(v) < 8e6 for v in grid["video.channel_bw_hz"])
    assert has_narrow is expect_narrow
    assert knob_sweep.is_low_band(freq_hz) is expect_narrow
    if expect_narrow:
        assert 13e6 in grid["video.sample_rate"]
        assert 40 in grid["video.capture_ms"]
        assert 80 in grid["video.capture_ms"]
        assert grid["video.sample_rate"][0] == 13e6


def test_ofat_low_band_has_extra_trials_vs_high() -> None:
    high_grid = knob_sweep.apply_band_extras(
        dict(knob_sweep.DEFAULT_LOCK_GRID), ARB_HIGH_HZ,
    )
    low_grid = knob_sweep.apply_band_extras(
        dict(knob_sweep.DEFAULT_LOCK_GRID), ARB_LOW_HZ,
    )
    high = knob_sweep.ofat_trials(STOCK_LIVE, high_grid)
    low = knob_sweep.ofat_trials(STOCK_LIVE, low_grid)
    assert 28 <= len(high) <= 32
    assert len(low) > len(high)
    low_bw = {t["values"].get("video.channel_bw_hz") for t in low if t["values"]}
    high_bw = {t["values"].get("video.channel_bw_hz") for t in high if t["values"]}
    assert 4e6 in low_bw or 6e6 in low_bw
    assert 4e6 not in high_bw and 6e6 not in high_bw


def test_paired_snow_trial_only_low_band_unlocked() -> None:
    snow = {"locked": False, "row_corr": 0.01, "free_run": True}
    locked = {"locked": True, "row_corr": 0.50, "free_run": False}
    extra = knob_sweep.paired_snow_trial(snow, ARB_LOW_HZ, STOCK_LIVE)
    assert extra is not None
    assert len(extra["values"]) == 2
    assert knob_sweep.paired_snow_trial(snow, ARB_HIGH_HZ, STOCK_LIVE) is None
    assert knob_sweep.paired_snow_trial(locked, ARB_LOW_HZ, STOCK_LIVE) is None


def test_analog_winner_printed_not_auto_kept() -> None:
    rows = [
        {"applied_values": '{"video.sample_rate": 13000000.0}',
         "row_corr": 0.58, "pic_score": 0.65, "locked": True, "line_rate": 15625.0},
        {"applied_values": '{"video.capture_ms": 40}',
         "row_corr": 0.10, "pic_score": 0.20, "locked": False, "line_rate": 15625.0},
    ]
    patch = knob_sweep.analog_winner_patch(rows)
    assert patch == {"video.sample_rate": 13000000.0}
    text = knob_sweep.format_winning_put(patch)
    assert "not applied" in text
    assert "13000000" in text
    assert knob_sweep.analog_winner_patch([
        {"applied_values": '{"video.channel_bw_hz": 8000000.0}',
         "row_corr": 0.15, "pic_score": 0.20, "locked": False, "line_rate": 15625.0},
    ]) is None


def test_wait_settle_notes_empty_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hz = ARB_LOW_HZ
    state = {"mode": "LOCK", "lock_target": hz, "tuned_hz": hz, "video": None}

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        if req.full_url.endswith("/api/state"):
            return _FakeResp(200, state)
        if req.full_url.endswith("/api/test/live"):
            return _FakeResp(200, {
                "mode": "LOCK", "lock_target_hz": hz, "video_metrics": {},
            })
        return _FakeResp(404, {"detail": req.full_url})

    monkeypatch.setattr(knob_sweep, "urlopen", fake_urlopen)
    http = knob_sweep.HttpClient("http://127.0.0.1:8080")
    _st, _live, note = knob_sweep.wait_settle(
        http, freq_hz=hz, dwell_s=0, extra_s=0, fresh_s=0.1, before=None,
        need_video=True, sleep_fn=lambda _s: None,
    )
    assert note is not None
    assert "no video" in note


def test_run_restore_prints_winning_put(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    puts: list[dict] = []
    live = dict(STOCK_LIVE)

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        url = req.full_url
        method = req.get_method()
        if url.endswith("/api/health"):
            return _FakeResp(200, {"ok": True})
        if url.endswith("/api/test/parameters/current"):
            return _FakeResp(200, {"values": dict(STOCK_LIVE)})
        if "/api/lock/" in url:
            return _FakeResp(200, {"ok": True})
        if url.endswith("/api/test/parameters") and method == "PUT":
            body = json.loads(req.data.decode("utf-8"))
            vals = dict(body.get("values") or {})
            puts.append(vals)
            live.update(vals)
            return _FakeResp(200, {
                "applied_keys": list(vals), "pending_keys": [], "values": dict(live),
            })
        video = dict(LOCK_STATE["video"])
        bw = live.get("video.channel_bw_hz", 16e6)
        if abs(float(bw) - 8e6) < 1.0:
            video["row_corr"] = 0.81
            video["pic_score"] = 0.70
        state = dict(LOCK_STATE)
        state["video"] = video
        if url.endswith("/api/state"):
            return _FakeResp(200, state)
        if url.endswith("/api/test/live"):
            return _FakeResp(200, {
                "mode": "LOCK",
                "lock_target_hz": ARB_HIGH_HZ,
                "video_metrics": {"fps": 5.0, "locked": True, "line_rate": 15625.0},
            })
        return _FakeResp(404, {"detail": url})

    monkeypatch.setattr(knob_sweep, "urlopen", fake_urlopen)
    monkeypatch.setattr(knob_sweep.time, "sleep", lambda _s: None)
    monkeypatch.delenv("FPVSCAN_BASE", raising=False)
    ns = knob_sweep.build_parser().parse_args([
        "--freq", "5.1e9",
        "--out", str(tmp_path / "out"),
        "--install-root", str(tmp_path),
        "--base", "http://127.0.0.1:8080",
        "--dwell-ms", "1",
        "--fresh-ms", "0",
        "--no-stop-on-picture",
    ])
    rc = knob_sweep.run(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "restored baseline knobs" in out
    assert "kept best trial knobs" not in out
    assert "winning PUT" in out
    assert "not applied" in out
    assert "8000000" in out
    final = puts[-1]
    assert abs(float(final.get("video.channel_bw_hz", 0)) - 12e6) < 1.0


def test_band_extras_only_below_2ghz_not_a_named_bird() -> None:
    """Low-band IF slice is a 2 GHz threshold, not a hardcoded MHz list."""
    grid = knob_sweep.DEFAULT_LOCK_GRID
    for hz in (2e9, 2.1e9, ARB_HIGH_HZ):
        high = knob_sweep.apply_band_extras(grid, hz)
        assert high["video.channel_bw_hz"] == list(grid["video.channel_bw_hz"])
        assert list(high) == list(grid)

    for hz in (0.9e9, ARB_LOW_HZ, 1.8e9):
        low = knob_sweep.apply_band_extras(grid, hz)
        bw = low["video.channel_bw_hz"]
        assert any(knob_sweep.values_close(x, 4e6) for x in bw)
        assert any(knob_sweep.values_close(x, 6e6) for x in bw)
        rates = low["video.sample_rate"]
        assert knob_sweep.values_close(rates[0], 13e6)
        keys = list(low)
        assert keys.index("video.sample_rate") < keys.index("video.channel_bw_hz")
        assert keys.index("video.capture_ms") < keys.index("video.channel_bw_hz")
        cap = low["video.capture_ms"]
        assert int(cap[0]) == 40
        assert int(cap[1]) == 80


def test_low_band_ofat_keeps_13e6_when_live_is_20e6() -> None:
    grid = knob_sweep.apply_band_extras(knob_sweep.DEFAULT_LOCK_GRID, ARB_LOW_HZ)
    trials = knob_sweep.ofat_trials(STOCK_LIVE, grid)
    patches = [t["values"] for t in trials if t["values"]]
    assert any(
        set(p) == {"video.sample_rate"}
        and knob_sweep.values_close(p["video.sample_rate"], 13e6)
        for p in patches
    )
    assert next(iter(patches[0])) == "video.sample_rate"
    n_high = len(knob_sweep.ofat_trials(STOCK_LIVE, knob_sweep.DEFAULT_LOCK_GRID))
    assert len(trials) > n_high


def test_empty_video_row_not_ranked_as_win() -> None:
    """Empty-video note must not beat a scorable OFAT row."""
    rows = [
        {
            "applied_values": "", "row_corr": 0.139, "pic_score": 0.2,
            "line_rate": 16968, "locked": False,
        },
        {
            "applied_values": '{"video.afc_deadband_hz": 160000.0}',
            "row_corr": 0.90, "pic_score": 0.80, "line_rate": 15625,
            "locked": True,
            "note": "no video after dwell (decode never published)",
        },
        {
            "applied_values": '{"video.sample_rate": 13000000.0}',
            "row_corr": 0.583, "pic_score": 0.41, "line_rate": 15625,
            "locked": True,
        },
    ]
    assert not knob_sweep.row_scorable(rows[1])
    assert knob_sweep.best_trial_patch(rows) == {"video.sample_rate": 13000000.0}
    ranked = knob_sweep.correlate_knobs(rows)
    by_key = {item["key"]: item for item in ranked}
    assert "video.afc_deadband_hz" not in by_key
    assert knob_sweep.analog_winner_patch(rows) == {"video.sample_rate": 13000000.0}


def test_paired_snow_trial_low_band_only() -> None:
    snow = {"locked": False, "row_corr": 0.01}
    analog = {"locked": True, "row_corr": 0.45}
    extra = knob_sweep.paired_snow_trial(snow, ARB_LOW_HZ, STOCK_LIVE)
    assert extra is not None
    assert extra["values"] == knob_sweep.PAIRED_LOW_IF
    assert knob_sweep.paired_snow_trial(analog, ARB_LOW_HZ, STOCK_LIVE) is None
    assert knob_sweep.paired_snow_trial(snow, ARB_HIGH_HZ, STOCK_LIVE) is None
    already = dict(STOCK_LIVE)
    already["video.channel_bw_hz"] = 10e6
    already["video.sample_rate"] = 13e6
    assert knob_sweep.paired_snow_trial(snow, 1.8e9, already) is None
    row = {
        "applied_values": json.dumps(extra["values"]),
        "row_corr": 0.5, "pic_score": 0.4, "line_rate": 15625, "locked": True,
    }
    assert knob_sweep.correlate_knobs([row]) == []
    assert knob_sweep.best_trial_patch([row]) == extra["values"]


def test_format_winning_put_includes_curl() -> None:
    text = knob_sweep.format_winning_put(
        {"video.sample_rate": 13e6}, "http://10.252.65.47:8080",
    )
    assert "not applied" in text
    assert "--keep-best" in text
    assert "video.sample_rate" in text
    assert "curl" in text
    assert "idempotency_key" in text
    assert "/api/test/parameters" in text


def test_wait_settle_retries_state_get_on_error() -> None:
    clock = {"t": 0.0}
    n_state = {"n": 0}
    lock_state = {
        "mode": "LOCK",
        "lock_target": ARB_LOW_HZ,
        "tuned_hz": ARB_LOW_HZ,
        "video": {
            "pic_score": 0.4, "line_rate": 15625.0, "locked": True, "row_corr": 0.5,
        },
    }

    class FakeHttp:
        def request(self, method, path, payload=None, timeout_s=None):  # noqa: ANN001
            if path == knob_sweep.API_LIVE:
                return 200, {"mode": "LOCK", "lock_target_hz": ARB_LOW_HZ, "video_metrics": {}}
            if path == knob_sweep.API_STATE:
                n_state["n"] += 1
                if n_state["n"] <= 2:
                    raise knob_sweep.SweepHttpError(0, path, {"error": "blip"})
                return 200, dict(lock_state)
            raise AssertionError(path)

    def sleep(s: float) -> None:
        clock["t"] += float(s)

    state, _live, note = knob_sweep.wait_settle(
        FakeHttp(),  # type: ignore[arg-type]
        freq_hz=ARB_LOW_HZ, dwell_s=0.01, extra_s=0.0, fresh_s=0.0,
        before=None, need_video=True, poll_s=0.01,
        sleep_fn=sleep, clock_fn=lambda: clock["t"],
    )
    assert not knob_sweep.video_missing(state)
    assert n_state["n"] >= 3
    assert note is None or "no video after dwell" not in note


def test_wait_settle_empty_video_retries_then_notes() -> None:
    clock = {"t": 0.0}
    n_state = {"n": 0}
    empty = {"mode": "LOCK", "lock_target": ARB_LOW_HZ, "tuned_hz": ARB_LOW_HZ, "video": {}}
    filled = {
        "mode": "LOCK",
        "lock_target": ARB_LOW_HZ,
        "tuned_hz": ARB_LOW_HZ,
        "video": {
            "pic_score": 0.4, "line_rate": 15625.0, "locked": True, "row_corr": 0.5,
        },
    }

    class FakeHttp:
        def request(self, method, path, payload=None, timeout_s=None):  # noqa: ANN001
            if path == knob_sweep.API_LIVE:
                return 200, {"mode": "LOCK", "lock_target_hz": ARB_LOW_HZ, "video_metrics": {}}
            if path == knob_sweep.API_STATE:
                n_state["n"] += 1
                if n_state["n"] < 6:
                    return 200, dict(empty)
                return 200, dict(filled)
            raise AssertionError(path)

    def sleep(s: float) -> None:
        clock["t"] += float(s)

    state, _live, note = knob_sweep.wait_settle(
        FakeHttp(),  # type: ignore[arg-type]
        freq_hz=ARB_LOW_HZ, dwell_s=0.01, extra_s=0.0, fresh_s=0.0,
        before=None, need_video=True, poll_s=0.01,
        sleep_fn=sleep, clock_fn=lambda: clock["t"],
    )
    assert not knob_sweep.video_missing(state)
    assert n_state["n"] >= 6
    assert note is None or "no video after dwell" not in note


def test_run_low_band_snow_inserts_paired(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    snow = dict(LOCK_STATE)
    snow["lock_target"] = ARB_LOW_HZ
    snow["tuned_hz"] = ARB_LOW_HZ
    video = dict(LOCK_STATE["video"])
    video.update({
        "locked": False, "row_corr": 0.01, "pic_score": 0.09,
        "free_run": True, "freq_hz": ARB_LOW_HZ, "line_rate": 16968,
    })
    snow["video"] = video

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        url = req.full_url
        method = req.get_method()
        if url.endswith("/api/health"):
            return _FakeResp(200, {"ok": True})
        if url.endswith("/api/test/parameters/current"):
            return _FakeResp(200, {"values": dict(STOCK_LIVE)})
        if "/api/lock/" in url:
            return _FakeResp(200, {"ok": True})
        if url.endswith("/api/test/parameters") and method == "PUT":
            return _FakeResp(200, {
                "applied_keys": [], "pending_keys": [], "values": dict(STOCK_LIVE),
            })
        if url.endswith("/api/state"):
            return _FakeResp(200, dict(snow))
        if url.endswith("/api/test/live"):
            return _FakeResp(200, {
                "mode": "LOCK",
                "lock_target_hz": ARB_LOW_HZ,
                "video_metrics": {"fps": 5.0, "locked": False, "line_rate": 16968},
            })
        return _FakeResp(404, {"detail": url})

    monkeypatch.setattr(knob_sweep, "urlopen", fake_urlopen)
    monkeypatch.setattr(knob_sweep.time, "sleep", lambda _s: None)
    monkeypatch.delenv("FPVSCAN_BASE", raising=False)
    ns = knob_sweep.build_parser().parse_args([
        "--freq", "1.1e9",
        "--out", str(tmp_path / "out"),
        "--install-root", str(tmp_path),
        "--base", "http://127.0.0.1:8080",
        "--dwell-ms", "1",
        "--fresh-ms", "0",
        "--no-restore",
        "--no-stop-on-picture",
    ])
    rc = knob_sweep.run(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "low-band IF slice" in out
    assert "paired" in out.lower()
    jsonl = tmp_path / "out" / "knob_sweep.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
    paired = [
        r for r in rows
        if "video.channel_bw_hz" in (r.get("applied_keys") or "")
        and "video.sample_rate" in (r.get("applied_keys") or "")
    ]
    assert len(paired) == 1
    patch = json.loads(paired[0]["applied_values"])
    assert knob_sweep.values_close(patch["video.channel_bw_hz"], 10e6)
    assert knob_sweep.values_close(patch["video.sample_rate"], 13e6)


def test_defaults_dwell_fresh_shorter_than_old() -> None:
    ns = knob_sweep.build_parser().parse_args(["--freq", "5.1e9"])
    assert ns.dwell_ms < 2500
    assert ns.fresh_ms < 8000
    assert 800 <= ns.dwell_ms <= 1000
    assert 2000 <= ns.fresh_ms <= 2500
    assert ns.stop_on_picture is True
    ns_off = knob_sweep.build_parser().parse_args(
        ["--freq", "5.1e9", "--no-stop-on-picture"],
    )
    assert ns_off.stop_on_picture is False
    # ~30 trials × (dwell+fresh) is ~90–120 s, not 5+ minutes.
    hunt_s = 30 * (ns.dwell_ms + ns.fresh_ms) / 1000.0
    assert 90 <= hunt_s <= 120


def test_drifted_lock_target_row_is_not_winner() -> None:
    hz = ARB_HIGH_HZ
    rows = [
        {
            "trial": 0, "freq_hz": hz, "applied_values": "",
            "row_corr": 0.10, "pic_score": 0.12, "locked": False,
            "lock_target": hz, "tuned_hz": hz, "line_rate": 15625,
        },
        {
            "trial": 1, "freq_hz": hz,
            "applied_values": '{"video.channel_bw_hz": 8000000.0}',
            "row_corr": 0.95, "pic_score": 0.90, "locked": True,
            "lock_target": hz - 2e6, "tuned_hz": hz - 2e6, "line_rate": 15625,
        },
        {
            "trial": 2, "freq_hz": hz,
            "applied_values": '{"video.deviation_hz": 8000000.0}',
            "row_corr": 0.50, "pic_score": 0.45, "locked": True,
            "lock_target": hz, "tuned_hz": hz, "line_rate": 15625,
        },
    ]
    assert knob_sweep.best_trial_patch(rows) == {"video.deviation_hz": 8000000.0}
    assert knob_sweep.analog_winner_patch(rows) == {"video.deviation_hz": 8000000.0}


def test_late_cosmetic_lock_after_snow_not_winner() -> None:
    """TX appearing mid-sweep is not a crop/sharpen win. Arbitrary Hz."""
    hz = ARB_LOW_HZ
    rows = [
        {
            "trial": 0, "freq_hz": hz, "applied_values": "",
            "row_corr": 0.01, "pic_score": 0.09, "locked": False,
            "lock_target": hz, "tuned_hz": hz, "line_rate": 16968,
        },
        {
            "trial": 1, "freq_hz": hz,
            "applied_values": '{"video.channel_bw_hz": 8000000.0}',
            "row_corr": 0.02, "pic_score": 0.10, "locked": False,
            "lock_target": hz, "tuned_hz": hz, "line_rate": 16968,
        },
        {
            "trial": 20, "freq_hz": hz,
            "applied_values": '{"video.crop_bottom_lines": 0}',
            "row_corr": 0.974, "pic_score": 0.90, "locked": True,
            "lock_target": hz, "tuned_hz": hz, "line_rate": 15625,
        },
        {
            "trial": 21, "freq_hz": hz,
            "applied_values": '{"video.sharpen": 0.0}',
            "row_corr": 0.833, "pic_score": 0.80, "locked": True,
            "lock_target": hz, "tuned_hz": hz, "line_rate": 15625,
        },
        {
            "trial": 22, "freq_hz": hz,
            "applied_values": '{"video.track_window_margin": 0.80}',
            "row_corr": 0.80, "pic_score": 0.70, "locked": True,
            "lock_target": hz, "tuned_hz": hz, "line_rate": 15625,
        },
    ]
    assert knob_sweep.best_trial_patch(rows) is None
    assert knob_sweep.analog_winner_patch(rows) is None
    note = knob_sweep.mid_sweep_signal_note(rows) or ""
    assert "mid-sweep" in note
    assert "not attributed" in note


def test_run_stop_on_picture_skips_later_trials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    """First analog lock ends OFAT; later knobs are not PUT."""
    puts: list[dict] = []
    live = dict(STOCK_LIVE)

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        url = req.full_url
        method = req.get_method()
        if url.endswith("/api/health"):
            return _FakeResp(200, {"ok": True})
        if url.endswith("/api/test/parameters/current"):
            return _FakeResp(200, {"values": dict(STOCK_LIVE)})
        if "/api/lock/" in url:
            return _FakeResp(200, {"ok": True})
        if url.endswith("/api/test/parameters") and method == "PUT":
            body = json.loads(req.data.decode("utf-8"))
            vals = dict(body.get("values") or {})
            puts.append(vals)
            live.update(vals)
            return _FakeResp(200, {
                "applied_keys": list(vals), "pending_keys": [], "values": dict(live),
            })
        video = dict(LOCK_STATE["video"])
        video.update({
            "locked": False, "row_corr": 0.01, "pic_score": 0.09,
            "free_run": True, "line_rate": 16968,
        })
        bw = live.get("video.channel_bw_hz", 12e6)
        if abs(float(bw) - 8e6) < 1.0:
            video.update({
                "locked": True, "row_corr": 0.55, "pic_score": 0.50,
                "free_run": False, "line_rate": 15625.0,
            })
        state = dict(LOCK_STATE)
        state["video"] = video
        if url.endswith("/api/state"):
            return _FakeResp(200, state)
        if url.endswith("/api/test/live"):
            return _FakeResp(200, {
                "mode": "LOCK",
                "lock_target_hz": ARB_HIGH_HZ,
                "video_metrics": {
                    "fps": 5.0, "locked": bool(video["locked"]),
                    "line_rate": video["line_rate"],
                },
            })
        return _FakeResp(404, {"detail": url})

    monkeypatch.setattr(knob_sweep, "urlopen", fake_urlopen)
    monkeypatch.setattr(knob_sweep.time, "sleep", lambda _s: None)
    monkeypatch.delenv("FPVSCAN_BASE", raising=False)
    ns = knob_sweep.build_parser().parse_args([
        "--freq", "5.1e9",
        "--out", str(tmp_path / "out"),
        "--install-root", str(tmp_path),
        "--base", "http://127.0.0.1:8080",
        "--dwell-ms", "1",
        "--fresh-ms", "0",
        "--no-restore",
    ])
    assert ns.stop_on_picture is True
    rc = knob_sweep.run(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "stop-on-picture" in out
    jsonl = tmp_path / "out" / "knob_sweep.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
    expected_full = len(knob_sweep.ofat_trials(STOCK_LIVE, knob_sweep.DEFAULT_LOCK_GRID))
    assert 1 < len(rows) < expected_full
    assert any("video.channel_bw_hz" in (r.get("applied_keys") or "") for r in rows)
    assert not any("video.crop_bottom_lines" in (r.get("applied_keys") or "") for r in rows)
    assert not any("video.sharpen" in (r.get("applied_keys") or "") for r in rows)
    assert not any(
        abs(float(p.get("video.crop_bottom_lines", -1)) - 0) < 1e-9
        and set(p) == {"video.crop_bottom_lines"}
        for p in puts
    )


def test_run_baseline_already_analog_stops(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    puts: list[dict] = []

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        url = req.full_url
        method = req.get_method()
        if url.endswith("/api/health"):
            return _FakeResp(200, {"ok": True})
        if url.endswith("/api/test/parameters/current"):
            return _FakeResp(200, {"values": dict(STOCK_LIVE)})
        if "/api/lock/" in url:
            return _FakeResp(200, {"ok": True})
        if url.endswith("/api/test/parameters") and method == "PUT":
            body = json.loads(req.data.decode("utf-8"))
            vals = dict(body.get("values") or {})
            puts.append(vals)
            return _FakeResp(200, {
                "applied_keys": list(vals), "pending_keys": [],
                "values": dict(STOCK_LIVE),
            })
        if url.endswith("/api/state"):
            return _FakeResp(200, dict(LOCK_STATE))
        if url.endswith("/api/test/live"):
            return _FakeResp(200, {
                "mode": "LOCK",
                "lock_target_hz": ARB_HIGH_HZ,
                "video_metrics": {"fps": 5.0, "locked": True, "line_rate": 15625.0},
            })
        return _FakeResp(404, {"detail": url})

    monkeypatch.setattr(knob_sweep, "urlopen", fake_urlopen)
    monkeypatch.setattr(knob_sweep.time, "sleep", lambda _s: None)
    monkeypatch.delenv("FPVSCAN_BASE", raising=False)
    ns = knob_sweep.build_parser().parse_args([
        "--freq", "5.1e9",
        "--out", str(tmp_path / "out"),
        "--install-root", str(tmp_path),
        "--base", "http://127.0.0.1:8080",
        "--dwell-ms", "1",
        "--fresh-ms", "0",
        "--no-restore",
    ])
    rc = knob_sweep.run(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "baseline already analog, stopping" in out
    jsonl = tmp_path / "out" / "knob_sweep.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["applied_values"] == ""
    assert puts == []

