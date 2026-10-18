"""Clear must not convert an auto-peek into a hold, or a hold into a peek."""
from __future__ import annotations

from tests.helpers import make_engine


def test_clear_during_peek_does_not_convert_to_hold():
    """#63 lock() sets auto=False so a leftover peek timer cannot
    steal a click-to-hold. Clear is the other button: it only
    wipes the list (#37). Setting auto=False on Clear would turn
    a 5 s auto-peek into a hold — Очистити would pin the operator
    on a channel they only wanted to glance at, and Sweep would
    never resume. Setting auto=True would expire a manual hold.
    Leave the flag and the deadline as-is.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    eng.state.auto_until = 9e9
    eng.command("clear")
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng.state.auto is True
    assert eng.state.auto_until == 9e9


def test_clear_during_hold_does_not_arm_peek_timer():
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5725e6
    eng.state.auto = False
    eng.state.auto_until = 0.0
    eng.command("clear")
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5725e6
    assert eng.state.auto is False
    assert eng.state.auto_until == 0.0
