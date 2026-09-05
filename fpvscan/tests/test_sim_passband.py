"""SimSource must not leak emitters that sit outside the current passband."""
from __future__ import annotations

import numpy as np

from fpvscan.sdr.sim import Emitter, SimSource, default_scene


def test_default_scene_has_ntsc_5g8_and_pal_1g2():
    scene = default_scene()
    kinds = {(e.freq_hz, e.kind, round(e.line_rate)) for e in scene}
    assert (5800e6, "fpv", 15734) in kinds
    assert (1280e6, "fpv", 15625) in kinds
    assert any(e.kind == "cw" for e in scene)


def test_out_of_band_emitter_is_not_added():
    src = SimSource(
        emitters=[Emitter(100e6, -10.0, "cw", 1e6, label="far")],
        noise_db=-80, seed=0,
    )
    src.open()
    src.set_sample_rate(20e6)
    src.set_center_freq(5800e6)
    src.set_gain(40)
    iq = src.read(4096)
    # Noise-only: magnitude stays near the -80 dB floor
    assert float(np.mean(np.abs(iq))) < 2e-3


def test_in_band_cw_is_a_tone():
    src = SimSource(
        emitters=[Emitter(5800e6, -12.0, "cw", 1e6, label="tone")],
        noise_db=-90, seed=0,
    )
    src.open()
    src.set_sample_rate(8e6)
    src.set_center_freq(5800e6)
    src.set_gain(40)
    iq = src.read(8192)
    spec = np.abs(np.fft.fftshift(np.fft.fft(iq)))
    peak = int(np.argmax(spec))
    assert peak == len(spec) // 2 or abs(peak - len(spec) // 2) <= 2
    assert float(spec.max()) > 8 * float(np.median(spec))
