"""File replay must not sleep; captures must not follow the process cwd."""
from __future__ import annotations

from pathlib import Path

from fpvscan import paths
from fpvscan.sdr.file import FileSource
from fpvscan.sdr.sim import SimSource


def test_file_source_does_not_sleep_by_default():
    """realtime=True sleeps n/fs per read (40 ms at 1k samples / 25 ksps).
    File replay is the Windows/CI debug path; a default sleep turns a
    2 s capture into wall-clock minutes and inspect_ms looks like a hang.
    """
    src = FileSource("unused.cf32")
    assert src.realtime is False
    assert src.loop is True


def test_output_dir_is_under_the_project_not_cwd():
    """Photos, mp4s, and .cf32 used to land wherever run.py was launched.
    ROOT is the fpvscan/ tree (parent of the package), OUT is absolute.
    """
    assert paths.OUT == paths.ROOT / "out"
    assert paths.OUT.is_absolute()
    pkg_dir = Path(paths.__file__).resolve().parent
    assert paths.ROOT == pkg_dir.parent
    assert paths.PHOTOS.parent == paths.OUT
    assert paths.VIDEO.parent == paths.OUT
    assert paths.CAPS.parent == paths.OUT


def test_sim_source_context_manager_closes():
    """Scripts use `with BladeRF() as src`. If __exit__ forgets close(),
    the next run.py on a Pi finds the board busy.
    """
    src = SimSource()
    assert src._open is False
    with src as opened:
        assert opened is src
        assert src._open is True
    assert src._open is False
