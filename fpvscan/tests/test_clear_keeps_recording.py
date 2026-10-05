"""Clear is a sidebar reset, not a leave-LOCK / stop-clip command."""
from __future__ import annotations

from tests.helpers import make_engine


class _Rec:
    def __init__(self):
        self.stopped = False
        self.started_at = 0.0

    def stop(self):
        self.stopped = True
        return {"path": "x.mp4", "bytes": 0, "seconds": 0, "kbps": 0}


def test_clear_does_not_stop_an_active_recorder():
    """#37 kept mode/lock_target so Clear would not take the
    lock/sweep branch that already stops the clip. If Clear later
    grows its own `_rec_stop()`, Очистити during a long hold still
    kills the only recording of that VTx — the operator thought they
    only wiped a stale Wi-Fi row.
    """
    eng = make_engine()
    rec = _Rec()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._rec = rec
    eng._handle_command("clear", {})
    assert rec.stopped is False
    assert eng._rec is rec
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
