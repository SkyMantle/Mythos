"""lock must not join the IQ reader in the command handler."""
from __future__ import annotations

import inspect

from fpvscan.engine import Engine


def test_lock_handler_does_not_stop_reader():
    """Sweep joins in `_handle_command` (#24) so `_do_sweep` owns USB.
    lock only nulls `_lock_tuned` (#21) and lets `_start_reader`
    join (#50) on the next `_do_lock`. Stopping the reader in the
    lock handler races a `_do_lock` that is already in
    `ring.snapshot` — the ring becomes None mid-frame, abs_start
    jumps, and tracking resets on every channel click.
    """
    src = inspect.getsource(Engine._handle_command)
    lock = src[src.index('if name == "lock"'):src.index('elif name == "sweep"')]
    sweep = src[src.index('elif name == "sweep"'):src.index('elif name == "clear"')]
    assert "_stop_reader" not in lock
    assert "_stop_reader" in sweep
    assert "_lock_tuned = None" in lock
