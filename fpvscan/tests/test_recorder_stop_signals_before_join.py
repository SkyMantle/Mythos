"""VideoRecorder.stop must flip the pump flag before joining it."""
from __future__ import annotations

import inspect

from fpvscan.recorder import VideoRecorder


def test_stop_sets_flag_before_join():
    """`Engine._rec_stop` / `_run` finally call `stop()`. Joining the
    pump first waits for a thread that is still in `sleep(1/fps)`
    because the loop only exits when `_stop` is set. A hang there
    keeps ffmpeg's stdin open and the next rec_start fails to replace
    the file. #35 only checked the hung-ffmpeg `kill` after wait.
    """
    src = inspect.getsource(VideoRecorder.stop)
    assert src.index("self._stop.set()") < src.index(".join(")
    assert "timeout=" in src
