"""INSPECT must keep the 25 ms grab and classifier floors when YAML omits them."""
from __future__ import annotations

from unittest.mock import patch

from fpvscan.dsp.demod import VideoScore
from fpvscan.dsp.spectrum import Occupancy
from tests.helpers import MockSource, make_engine

_REJECT = VideoScore(False, 0.0, "?", 0.0)


def test_inspect_omitted_ms_and_classifier_defaults():
    """#18 / #36 forwarded explicit inspect_ms=4 and classifier knobs,
    and pinned shipped 25 ms. The code fallbacks are inspect_ms=25,
    line_tol_hz=150, line_prominence_db=8, min_confidence=0.45.
    A 0 ms dump fails classify after decimation (<8192). A 0
    confidence floor lists Wi-Fi as analog. helpers.py already
    omits these scan keys — that is the operator-partial-config path.
    """
    src = MockSource()
    grabbed: list[int] = []
    real = src.retune_and_read

    def wrap(hz, n):
        grabbed.append(int(n))
        return real(hz, n)

    src.retune_and_read = wrap
    eng = make_engine(src=src)
    assert "inspect_ms" not in eng.cfg["scan"]
    assert "line_tol_hz" not in eng.cfg["scan"]
    occ = Occupancy(
        center_hz=5800e6, bandwidth_hz=8e6, peak_db=-18.0, snr_db=12.0,
    )
    with patch(
        "fpvscan.engine.demod.classify_video", return_value=_REJECT,
    ) as spy:
        eng._inspect(None, 5800e6, 20e6, occ)

    assert grabbed == [int(20e6 * 0.025)]
    kw = spy.call_args.kwargs
    assert kw["tol_hz"] == 150.0
    assert kw["min_prominence_db"] == 8.0
    assert kw["min_conf"] == 0.45
