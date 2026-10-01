"""User paths with a directory stay where the operator put them."""
from __future__ import annotations

from pathlib import Path

from fpvscan import paths


def test_resolve_nested_relative_is_not_moved_into_default_dir(tmp_path: Path):
    """`record.py -o caps/vtx_f4.cf32` and bringup `--save` pass a
    relative path with a parent. resolve() must keep `caps/...` as
    written. Prefixing default_dir (`out/caps/caps/...`) or stripping
    to a bare name (`out/caps/vtx_f4.cf32` while the operator looks
    in ./caps/) is how a 960 MB capture "vanishes" after a successful
    write.
    """
    default_dir = tmp_path / "out" / "caps"
    p = paths.resolve("caps/vtx_f4.cf32", default_dir, "unused.cf32")
    assert p.name == "vtx_f4.cf32"
    assert p.parent.name == "caps"
    assert p.parent != default_dir
    assert default_dir not in p.parents
