"""nearest_channel's 6 MHz window is inclusive, matching MERGE_TOL."""
from __future__ import annotations

from fpvscan.bands import nearest_channel


def test_channel_tol_includes_exactly_six_mhz():
    """#61 pinned omitted `tol_mhz=6`: 5705 is E1, 5715 (10 MHz) is
    None. The predicate is `best_d <= tol_mhz`, so 5711 MHz (exactly
    6 MHz from E1) is still E1 and 5712 is a gap. A leftover `<`
    blanks a bird sitting on the window edge; a leftover 20 labels
    5712 as E1/A8 and the console first column lies.
    """
    assert nearest_channel(5711e6) == "E1"
    assert nearest_channel(5712e6) is None
    assert nearest_channel(5705e6) == "E1"
