"""priority_bands must interleave, not append after a full 400–6 GHz pass."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_priority_bands_interleave_one_for_one():
    """~99% of analog FPV lives in PRIORITY_BANDS. Appending those
    points after the solid sweep means a 5.8 GHz bird waits a full
    400 MHz–6 GHz pass (~tens of seconds on USB) before the extra
    visit. The 1:1 interleave is the whole feature. The code
    default is True — omitting the YAML key must keep it.
    """
    eng = make_engine(src=MockSource())
    eng.cfg["scan"]["priority_bands"] = False
    plain = eng._sweep_plan()
    assert plain, "empty sweep plan"

    eng.cfg["scan"]["priority_bands"] = True
    interleaved = eng._sweep_plan()
    assert interleaved[0::2] == plain
    assert len(interleaved) == 2 * len(plain)
    assert interleaved[1] != interleaved[0]
    solid_5g8 = next(i for i, f in enumerate(plain) if f >= 5100e6)
    extra_5g8 = next(i for i, f in enumerate(interleaved)
                     if i % 2 == 1 and 5100e6 <= f <= 6000e6)
    assert extra_5g8 < 2 * solid_5g8, (
        "5G8 extra visit must land before the solid sweep reaches "
        f"5.1 GHz (slot {extra_5g8} vs solid slot {2 * solid_5g8})"
    )

    del eng.cfg["scan"]["priority_bands"]
    assert len(eng._sweep_plan()) == 2 * len(plain)
