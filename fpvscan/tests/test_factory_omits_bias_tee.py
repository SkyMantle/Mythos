"""factory must not enable bias-tee inside BladeRF.open()."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_does_not_forward_bias_tee(monkeypatch):
    """Shipped YAML has `bias_tee: true`. If factory passes that into
    BladeRF(), `open()` turns on 4.5 V and writes the full `gain_db`
    *before* Engine._run applies `bias_tee_gain_offset_db` (15 dB).
    On a stand with an LNA that is a clipped ADC and a striped
    picture that looks like a dead decoder. Bias-tee stays an
    engine-start concern, not a constructor side effect.
    """
    seen: dict = {}

    class Fake:
        name = "bladerf"

        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr("fpvscan.sdr.bladerf.BladeRF", Fake)
    make_source({
        "driver": "bladerf",
        "bias_tee": True,
        "gain_db": 35,
        "bias_tee_gain_offset_db": 15,
    })
    assert "bias_tee" not in seen
    assert seen["gain_db"] == 35.0
