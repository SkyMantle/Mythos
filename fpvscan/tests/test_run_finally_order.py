"""Engine teardown must release the reader before the exclusive handle."""
from __future__ import annotations

import inspect

from fpvscan.engine import Engine


def test_run_finally_stops_reader_then_recorder_then_source():
    """#19 only checked that the three calls exist inside `finally`.
    Closing the bladeRF first leaves `_reader_loop` in `sync_rx` on a
    freed device (NIOS hang / use-after-free). Stopping ffmpeg after
    `src.close()` is fine; stopping it after the reader still holds
    USB is not the issue — the reader-before-close order is.
    """
    src = inspect.getsource(Engine._run)
    fin = src[src.index("finally:"):]
    assert fin.index("self._stop_reader()") < fin.index("self._rec_stop()")
    assert fin.index("self._rec_stop()") < fin.index("self.src.close()")
