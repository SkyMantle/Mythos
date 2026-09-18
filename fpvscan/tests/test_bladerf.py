from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from fpvscan.sdr.bladerf import BladeRF, BladeRFError, ERR_IO, ERR_TIMEOUT


def _configured_source(lib) -> BladeRF:
    src = BladeRF.__new__(BladeRF)
    src.lib = lib
    src._dev = object()
    src.ch = 0
    src._fc = 1.1e9
    src._fs = 2e6
    src._bw = 1.8e6
    src._raw = None
    src._streaming = True
    src.use_meta = False
    src.timeout_ms = 50
    src.num_buffers = 4
    src.buffer_size = 1024
    src.num_transfers = 2
    src.bandwidth_ratio = 0.9
    src.settle_us = 100
    src.agc = False
    src.gain_db = 20
    src.bias_tee = False
    src.timeouts = 0
    src.stream_restarts = 0
    src.io_errors = 0
    src.overflows = 0
    src._read_cancel_event = threading.Event()
    src._consecutive_rx_errors = 0
    return src


def test_reconfigure_disables_before_rate_bandwidth_config_and_enable(
        monkeypatch) -> None:
    calls: list[str] = []

    class Lib:
        @staticmethod
        def bladerf_enable_module(_dev, _ch, enabled):
            calls.append("enable" if enabled else "disable")
            return 0

        @staticmethod
        def bladerf_set_sample_rate(_dev, _ch, hz, actual):
            calls.append("sample_rate")
            actual._obj.value = hz
            return 0

        @staticmethod
        def bladerf_set_bandwidth(_dev, _ch, hz, actual):
            calls.append("bandwidth")
            actual._obj.value = hz
            return 0

        @staticmethod
        def bladerf_sync_config(*_args):
            calls.append("sync_config")
            return 0

        @staticmethod
        def bladerf_strerror(_rc):
            return b"error"

    src = _configured_source(Lib())
    monkeypatch.setattr("fpvscan.sdr.bladerf.load_lib", lambda _path=None: src.lib)

    src.set_sample_rate(4e6)

    assert calls == [
        "disable", "sample_rate", "bandwidth", "sync_config", "enable"]
    assert src.sample_rate == 4e6
    assert src._stream_state == "OK"


def test_frequency_retune_only_sets_frequency() -> None:
    calls: list[str] = []

    class Lib:
        @staticmethod
        def bladerf_enable_module(_dev, _ch, enabled):
            calls.append("enable" if enabled else "disable")
            return 0

        @staticmethod
        def bladerf_set_frequency(*_args):
            calls.append("frequency")
            return 0

        @staticmethod
        def bladerf_sync_config(*_args):
            calls.append("sync_config")
            return 0

    src = _configured_source(Lib())
    src.set_center_freq(5.1e9)

    assert calls == ["frequency"]
    assert src.stream_restarts == 0


def test_repeated_sweep_retunes_do_not_restart_stream() -> None:
    calls: list[int] = []

    class Lib:
        @staticmethod
        def bladerf_set_frequency(_dev, _ch, hz):
            calls.append(hz)
            return 0

    src = _configured_source(Lib())
    for hz in (1.01e9, 1.03e9, 1.07e9, 1.09e9):
        src.set_center_freq(hz)

    assert calls == [1_010_000_000, 1_030_000_000, 1_070_000_000, 1_090_000_000]
    assert src.stream_restarts == 0
    assert src._streaming is True


def test_gain_update_is_serialized_control_only_without_stream_restart() -> None:
    calls: list[str] = []

    class Lib:
        @staticmethod
        def bladerf_set_gain_mode(*_args):
            calls.append("gain_mode")
            return 0

        @staticmethod
        def bladerf_set_gain(*_args):
            calls.append("gain")
            return 0

        @staticmethod
        def bladerf_enable_module(*_args):
            calls.append("enable_module")
            return 0

        @staticmethod
        def bladerf_sync_config(*_args):
            calls.append("sync_config")
            return 0

    src = _configured_source(Lib())
    src.set_gain(27)
    src.set_gain(27)
    src.set_gain(29)

    assert calls == ["gain_mode", "gain", "gain"]
    assert src.gain_db == 29
    assert src._streaming is True
    assert src.stream_restarts == 0


def test_hardware_mutation_waits_until_sync_rx_returns() -> None:
    receive_entered = threading.Event()
    release_receive = threading.Event()
    mutation_done = threading.Event()
    mutated_while_receiving = False

    class Lib:
        @staticmethod
        def bladerf_sync_rx(*_args):
            receive_entered.set()
            assert release_receive.wait(1)
            return 0

        @staticmethod
        def bladerf_enable_module(*_args):
            return 0

        @staticmethod
        def bladerf_set_sample_rate(_dev, _ch, hz, actual):
            nonlocal mutated_while_receiving
            mutated_while_receiving = not release_receive.is_set()
            actual._obj.value = hz
            return 0

        @staticmethod
        def bladerf_set_bandwidth(_dev, _ch, hz, actual):
            actual._obj.value = hz
            return 0

        @staticmethod
        def bladerf_sync_config(*_args):
            return 0

    src = _configured_source(Lib())

    reader = threading.Thread(target=lambda: src.read(1024))
    changer = threading.Thread(
        target=lambda: (src.set_sample_rate(4e6), mutation_done.set()))
    reader.start()
    assert receive_entered.wait(1)
    changer.start()
    time.sleep(0.02)
    assert not mutation_done.is_set()
    release_receive.set()
    reader.join(1)
    changer.join(1)

    assert mutation_done.is_set()
    assert not mutated_while_receiving


def test_manual_gain_waits_for_bounded_sync_rx_without_rebuild() -> None:
    receive_entered = threading.Event()
    release_receive = threading.Event()
    gain_done = threading.Event()
    calls: list[str] = []

    class Lib:
        @staticmethod
        def bladerf_sync_rx(*_args):
            receive_entered.set()
            assert release_receive.wait(1)
            return 0

        @staticmethod
        def bladerf_set_gain_mode(*_args):
            assert release_receive.is_set()
            calls.append("gain_mode")
            return 0

        @staticmethod
        def bladerf_set_gain(*_args):
            calls.append("gain")
            return 0

    src = _configured_source(Lib())
    reader = threading.Thread(target=lambda: src.read(1024))
    changer = threading.Thread(
        target=lambda: (src.set_gain(31), gain_done.set()))
    reader.start()
    assert receive_entered.wait(1)
    changer.start()
    time.sleep(0.02)
    assert not gain_done.is_set()
    release_receive.set()
    reader.join(1)
    changer.join(1)

    assert gain_done.is_set()
    assert calls == ["gain_mode", "gain"]
    assert src.stream_restarts == 0


def test_disable_device_loss_fails_fast_and_is_surfaced(monkeypatch) -> None:
    disables = 0

    class Lib:
        @staticmethod
        def bladerf_enable_module(_dev, _ch, enabled):
            nonlocal disables
            assert not enabled
            disables += 1
            return ERR_IO

        @staticmethod
        def bladerf_strerror(_rc):
            return b"I/O error"

    src = _configured_source(Lib())
    monkeypatch.setattr("fpvscan.sdr.bladerf.load_lib", lambda _path=None: src.lib)

    with pytest.raises(BladeRFError) as caught:
        src.set_sample_rate(4e6)

    assert caught.value.code == ERR_IO
    assert disables == 1
    assert src._stream_disable_failures == 1
    assert src._streaming is False
    assert not src._dev


def test_first_device_lost_abandons_handle_and_stops_control_sequence() -> None:
    calls: list[str] = []

    class Lib:
        @staticmethod
        def bladerf_enable_module(_dev, _ch, enabled):
            calls.append("enable" if enabled else "disable")
            return 0

        @staticmethod
        def bladerf_set_gain_mode(*_args):
            calls.append("gain_mode")
            return 0

        @staticmethod
        def bladerf_set_gain(*_args):
            calls.append("gain")
            return ERR_IO

        @staticmethod
        def bladerf_set_sample_rate(*_args):
            calls.append("sample_rate")
            return 0

        @staticmethod
        def bladerf_set_bandwidth(*_args):
            calls.append("bandwidth")
            return 0

        @staticmethod
        def bladerf_sync_config(*_args):
            calls.append("sync_config")
            return 0

        @staticmethod
        def bladerf_set_frequency(*_args):
            calls.append("frequency")
            return 0

        @staticmethod
        def bladerf_close(*_args):
            calls.append("close")

        @staticmethod
        def bladerf_strerror(_rc):
            return b"No devices available"

    src = _configured_source(Lib())
    with pytest.raises(BladeRFError) as first:
        src.set_gain(25)
    with pytest.raises(BladeRFError) as later:
        src.set_sample_rate(4e6)
    with pytest.raises(BladeRFError) as retune:
        src.set_center_freq(1.2e9)
    src.close()
    src.close()

    assert first.value.code == ERR_IO
    assert later.value.code == ERR_IO
    assert retune.value.code == ERR_IO
    assert calls == ["gain_mode", "gain", "close"]
    assert not src._dev
    assert src._stream_state == "WAITING_DEVICE"
    assert src._streaming is False


def test_reader_stop_cancels_before_second_long_retry() -> None:
    stop = threading.Event()
    calls = {"rx": 0, "restart": 0}

    class Lib:
        def bladerf_sync_rx(self, *_args):
            calls["rx"] += 1
            if calls["rx"] > 1:
                raise AssertionError("second blocking receive must be skipped")
            return ERR_TIMEOUT

    src = BladeRF.__new__(BladeRF)
    src.lib = Lib()
    src._need_dev = lambda: None
    src._dev = object()
    src._raw = None
    src.use_meta = False
    src.timeout_ms = 3500
    src.timeouts = 0
    src.stream_restarts = 0
    src.overflows = 0
    src._read_cancel_event = stop

    def restart() -> None:
        calls["restart"] += 1
        stop.set()

    src._config_stream = restart

    with pytest.raises(InterruptedError):
        src.read(1024)

    assert calls == {"rx": 1, "restart": 1}
    assert src.timeouts == 1
    assert src.stream_restarts == 1
    assert src._raw.dtype == np.int16


def test_timeout_restart_then_success_resets_driver_error_streak() -> None:
    calls = {"rx": 0, "restart": 0}
    src = BladeRF.__new__(BladeRF)

    class Lib:
        def bladerf_sync_rx(self, *_args):
            calls["rx"] += 1
            if calls["rx"] == 1:
                return ERR_TIMEOUT
            src._raw.fill(0)
            return 0

    src.lib = Lib()
    src._need_dev = lambda: None
    src._dev = object()
    src._raw = None
    src.use_meta = False
    src.timeout_ms = 3500
    src.timeouts = 0
    src.stream_restarts = 0
    src.io_errors = 0
    src.overflows = 0
    src._read_cancel_event = threading.Event()
    src._consecutive_rx_errors = 0

    def restart() -> None:
        calls["restart"] += 1

    src._config_stream = restart
    iq = src.read(1024)

    assert iq.shape == (1024,)
    assert calls == {"rx": 2, "restart": 1}
    assert src.timeouts == 1
    assert src.stream_restarts == 1
    assert src._consecutive_rx_errors == 0


def test_timeout_during_motor_retries_without_stream_restart() -> None:
    calls = {"rx": 0, "restart": 0}
    src = BladeRF.__new__(BladeRF)

    class Lib:
        def bladerf_sync_rx(self, *_args):
            calls["rx"] += 1
            if calls["rx"] == 1:
                return ERR_TIMEOUT
            src._raw.fill(0)
            return 0

    src.lib = Lib()
    src._need_dev = lambda: None
    src._dev = object()
    src._raw = None
    src.use_meta = False
    src.timeout_ms = 3500
    src.timeouts = 0
    src.stream_restarts = 0
    src.io_errors = 0
    src.overflows = 0
    src._read_cancel_event = threading.Event()
    src._consecutive_rx_errors = 0
    src._config_stream = lambda: calls.__setitem__(
        "restart", calls["restart"] + 1)
    src.set_rx_fragile(True)
    iq = src.read(1024)

    assert iq.shape == (1024,)
    assert calls == {"rx": 2, "restart": 0}
    assert src.timeouts == 1
    assert src.stream_restarts == 0
    assert src._consecutive_rx_errors == 0


def test_io_error_escapes_to_engine_without_driver_reopen(monkeypatch) -> None:
    calls = {"recover": 0}

    class Lib:
        @staticmethod
        def bladerf_sync_rx(*_args):
            return ERR_IO

        @staticmethod
        def bladerf_strerror(_rc):
            return b"I/O error"

    src = BladeRF.__new__(BladeRF)
    src.lib = Lib()
    src._need_dev = lambda: None
    src._dev = object()
    src._raw = None
    src.use_meta = False
    src.timeout_ms = 3500
    src.timeouts = 0
    src.stream_restarts = 0
    src.io_errors = 0
    src.overflows = 0
    src._read_cancel_event = threading.Event()
    src._consecutive_rx_errors = 0
    src._recover = lambda: calls.__setitem__("recover", calls["recover"] + 1)
    monkeypatch.setattr("fpvscan.sdr.bladerf.load_lib", lambda _path=None: src.lib)

    with pytest.raises(BladeRFError) as caught:
        src.read(1024)

    assert caught.value.code == ERR_IO
    assert src.io_errors == 1
    assert calls["recover"] == 0
    assert not src._dev
    assert src._stream_state == "WAITING_DEVICE"
