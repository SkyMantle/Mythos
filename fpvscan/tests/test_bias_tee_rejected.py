"""Board rejecting bias-tee must not silently cut receiver gain."""
from __future__ import annotations

from queue import Empty

from tests.helpers import MockSource, make_engine


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except Empty:
            return out


class _RejectBias(MockSource):
    def set_bias_tee(self, on):
        self.bias_req = bool(on)
        return False


def test_bias_tee_rejected_by_board_does_not_apply_lna_offset():
    """libbladeRF can return 'not supported' on a board without a
    bias-tee rail. Applying the LNA gain offset anyway would drop
    15 dB of RF gain on a live LOCK and the picture would go snow
    with no 'bias-tee failed' cue.
    """
    src = _RejectBias()
    src.gain = 35.0
    eng = make_engine(src=src)
    eng.command("bias_tee", on=True)
    eng._drain_commands()
    assert src.gain == 35.0
    notices = [e for e in _drain(eng.events) if e["type"] == "notice"]
    assert notices
    assert notices[-1]["data"]["level"] == "error"
    assert "не підтримується платою" in notices[-1]["data"]["text"]
