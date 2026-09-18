#!/usr/bin/env python3
"""Зібрати інсталятор fpvscan для Raspberry Pi 5 (Ubuntu arm64).

Запуск з Windows або Linux (з каталогу fpvscan або з кореня репо):

    py -3 deploy/build_pi_package.py
    python3 deploy/build_pi_package.py

Результат у dist/:
  fpvscan-pi5-arm64-<ver>.tar.gz   — скопіювати на Pi, розпакувати, sudo bash install.sh
  fpvscan_<ver>_arm64.deb          — dpkg -i (якщо вдалось зібрати ar-архів)

На Windows aarch64-бінарники numpy/scipy/numba не компілюються.
Скрипт тягне готові manylinux wheels під CPython 3.12 aarch64.
libbladeRF / ffmpeg у пакет не входять — їх ставить install.sh через apt.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = Path(__file__).resolve().parent
DEFAULT_VERSION = "0.1.0"
EXCLUDE_DIRS = {
    ".venv", ".git", "__pycache__", "out", "dist", ".pytest_cache",
    "caps", "wheels", "node_modules", ".cursor",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd"}
TEXT_EXEC_NAMES = {
    "install.sh", "install_pi.sh", "postinst", "prerm", "postrm",
}
PLATFORMS = (
    "manylinux_2_17_aarch64",
    "manylinux_2_28_aarch64",
    "manylinux_2_34_aarch64",
    "linux_aarch64",
)


def version() -> str:
    vf = DEPLOY / "VERSION"
    if vf.is_file():
        return vf.read_text(encoding="utf-8").strip() or DEFAULT_VERSION
    return DEFAULT_VERSION


def to_lf_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def is_text_script(path: Path) -> bool:
    if path.name in TEXT_EXEC_NAMES or path.suffix in {".sh", ".rules", ".service"}:
        return True
    if path.parent.name == "debian" and path.name in TEXT_EXEC_NAMES:
        return True
    return False


def should_skip(path: Path, rel: Path) -> bool:
    parts = set(rel.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if rel.suffix in EXCLUDE_SUFFIXES:
        return True
    if rel.parts[:1] == ("tests",):
        return True
    if path.name == "decoded.png" and "scripts" in rel.parts:
        return True
    return False


def iter_payload_files() -> list[tuple[Path, Path]]:
    """(absolute, relative-to-fpvscan-root) files to ship."""
    out: list[tuple[Path, Path]] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        pdir = Path(dirpath)
        rel_dir = pdir.relative_to(ROOT)
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        if any(part in EXCLUDE_DIRS for part in rel_dir.parts):
            continue
        for name in filenames:
            src = pdir / name
            rel = src.relative_to(ROOT)
            if should_skip(src, rel):
                continue
            out.append((src, rel))
    return out


def parse_reqs(path: Path) -> list[str]:
    reqs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


def pip_download(req: str, dest: Path, py_version: str, abi: str, platform: str) -> bool:
    cmd = [
        sys.executable, "-m", "pip", "download",
        req,
        "-d", str(dest),
        "--python-version", py_version,
        "--platform", platform,
        "--implementation", "cp",
        "--abi", abi,
        "--only-binary", ":all:",
        "--disable-pip-version-check",
    ]
    print("  ", " ".join(cmd[3:]))
    r = subprocess.run(cmd, cwd=str(ROOT))
    return r.returncode == 0


def download_wheels(dest: Path, py_version: str) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    abi = f"cp{py_version.replace('.', '')}"
    req_file = DEPLOY / "requirements-runtime.txt"
    reqs = parse_reqs(req_file)
    missing: list[str] = []
    print(f"== wheels CPython {py_version} / {abi} aarch64 -> {dest}")
    for old in dest.glob("*.whl"):
        old.unlink()
    resolved = False
    for plat in PLATFORMS:
        cmd = [
            sys.executable, "-m", "pip", "download",
            "-r", str(req_file),
            "-d", str(dest),
            "--python-version", py_version,
            "--platform", plat,
            "--implementation", "cp",
            "--abi", abi,
            "--only-binary", ":all:",
            "--disable-pip-version-check",
        ]
        print("  ", " ".join(cmd[3:]))
        if subprocess.run(cmd, cwd=str(ROOT)).returncode == 0:
            resolved = True
            break
    if not resolved:
        for req in reqs:
            ok = False
            for plat in PLATFORMS:
                if pip_download(req, dest, py_version, abi, plat):
                    ok = True
                    break
            if not ok:
                print(f"!! немає aarch64 wheel: {req}")
                missing.append(req)
    wheels = sorted(dest.glob("*.whl"))
    print(f"   зібрано {len(wheels)} файлів")
    return missing


def payload_bytes_mode(src: Path) -> tuple[bytes, int]:
    data = src.read_bytes()
    binary = src.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".whl", ".so"}
    if not binary:
        data = to_lf_bytes(data)
    mode = 0o644
    if (not binary) and (
        is_text_script(src) or src.suffix == ".sh" or src.name in {"run.py", "install.sh"}
    ):
        mode = 0o755
    return data, mode


def add_tar_bytes(tar: tarfile.TarFile, arcname: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name=arcname.replace("\\", "/"))
    info.size = len(data)
    info.mtime = int(time.time())
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    tar.addfile(info, io.BytesIO(data))


def add_tar_file(tar: tarfile.TarFile, src: Path, arcname: str) -> None:
    data, mode = payload_bytes_mode(src)
    add_tar_bytes(tar, arcname, data, mode)


def stage_name(ver: str) -> str:
    return f"fpvscan-pi5-arm64-{ver}"


def write_tarball(out_path: Path, ver: str, wheels_dir: Path) -> None:
    prefix = stage_name(ver)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"== tar.gz {out_path}")
    with tarfile.open(out_path, "w:gz", format=tarfile.GNU_FORMAT) as tar:
        install_sh = to_lf_bytes((DEPLOY / "install.sh").read_bytes())
        add_tar_bytes(tar, f"{prefix}/install.sh", install_sh, 0o755)
        readme = to_lf_bytes((DEPLOY / "INSTALL-PI.md").read_bytes())
        add_tar_bytes(tar, f"{prefix}/INSTALL-PI.md", readme, 0o644)
        for src, rel in iter_payload_files():
            add_tar_file(tar, src, f"{prefix}/{rel.as_posix()}")
        if wheels_dir.is_dir():
            for whl in sorted(wheels_dir.glob("*.whl")):
                add_tar_file(tar, whl, f"{prefix}/wheels/{whl.name}")


def gnu_ar(path: Path, members: list[tuple[str, bytes]]) -> None:
    """Write a System V / GNU ar archive (enough for dpkg)."""
    with path.open("wb") as f:
        f.write(b"!<arch>\n")
        for name, data in members:
            if len(name) > 15:
                raise ValueError(f"ar member name too long: {name}")
            header_name = (name + "/").ljust(16).encode("ascii")
            header = (
                header_name
                + f"{int(time.time()):<12}".encode("ascii")
                + b"0     "
                + b"0     "
                + b"100644  "
                + f"{len(data):<10}".encode("ascii")
                + b"`\n"
            )
            f.write(header)
            f.write(data)
            if len(data) % 2 == 1:
                f.write(b"\n")


def tar_gz_from_files(files: list[tuple[str, bytes, int]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.GNU_FORMAT) as tar:
        for name, data, mode in files:
            add_tar_bytes(tar, name, data, mode)
    return buf.getvalue()


def write_deb(out_path: Path, ver: str, wheels_dir: Path) -> None:
    print(f"== deb {out_path}")
    data_files: list[tuple[str, bytes, int]] = []

    def put(arc: str, data: bytes, mode: int = 0o644) -> None:
        data_files.append((arc, data, mode))

    for src, rel in iter_payload_files():
        data, mode = payload_bytes_mode(src)
        put(f"./opt/fpvscan/{rel.as_posix()}", data, mode)

    if wheels_dir.is_dir():
        for whl in sorted(wheels_dir.glob("*.whl")):
            put(f"./opt/fpvscan/wheels/{whl.name}", whl.read_bytes(), 0o644)

    put("./opt/fpvscan/install.sh", to_lf_bytes((DEPLOY / "install.sh").read_bytes()), 0o755)
    put("./opt/fpvscan/INSTALL-PI.md", to_lf_bytes((DEPLOY / "INSTALL-PI.md").read_bytes()), 0o644)

    svc = to_lf_bytes((DEPLOY / "fpvscan.service").read_bytes())
    put("./etc/systemd/system/fpvscan.service", svc, 0o644)
    rules = to_lf_bytes((DEPLOY / "88-nuand-bladerf.rules").read_bytes())
    put("./lib/udev/rules.d/88-nuand-bladerf.rules", rules, 0o644)
    pwm_rules = to_lf_bytes((DEPLOY / "99-pwm-gpio.rules").read_bytes())
    put("./lib/udev/rules.d/99-pwm-gpio.rules", pwm_rules, 0o644)

    data_tar = tar_gz_from_files(data_files)

    control_txt = (DEPLOY / "debian" / "control").read_text(encoding="utf-8")
    control_txt = control_txt.replace("@VERSION@", ver)
    scripts = []
    for name in ("control",):
        scripts.append((f"./{name}", to_lf_bytes(control_txt.encode("utf-8")), 0o644))
    for name in ("postinst", "prerm", "postrm"):
        raw = to_lf_bytes((DEPLOY / "debian" / name).read_bytes())
        scripts.append((f"./{name}", raw, 0o755))
    control_tar = tar_gz_from_files(scripts)

    gnu_ar(out_path, [
        ("debian-binary", b"2.0\n"),
        ("control.tar.gz", control_tar),
        ("data.tar.gz", data_tar),
    ])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Raspberry Pi 5 installer for fpvscan")
    ap.add_argument("--version", default=version())
    ap.add_argument("--python-version", default="3.12",
                    help="CPython ABI for wheels (Ubuntu 24.04 = 3.12)")
    ap.add_argument("--skip-wheels", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    wheels_dir = out / "wheels-aarch64"
    missing: list[str] = []
    if args.skip_wheels:
        print("== пропускаю wheels")
        wheels_dir.mkdir(parents=True, exist_ok=True)
    else:
        missing = download_wheels(wheels_dir, args.python_version)

    ver = args.version
    tarball = out / f"fpvscan-pi5-arm64-{ver}.tar.gz"
    deb = out / f"fpvscan_{ver}_arm64.deb"
    write_tarball(tarball, ver, wheels_dir)
    write_deb(deb, ver, wheels_dir)

    print()
    print("Артефакти:")
    sums = out / "SHA256SUMS"
    lines = []
    for p in (tarball, deb):
        if p.is_file():
            digest = sha256(p)
            lines.append(f"{digest}  {p.name}")
            print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)  sha256 {digest[:16]}…")
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {sums}")
    if missing:
        print()
        print("Wheels, яких немає (Pi доставить з PyPI під час install.sh):")
        for m in missing:
            print(f"  - {m}")
    print()
    print("На Pi 5 (Ubuntu 24.04 arm64):")
    print(f"  tar -xzf {tarball.name}")
    print(f"  cd {stage_name(ver)}")
    print("  sudo bash install.sh")
    print("або:")
    print(f"  sudo dpkg -i {deb.name}")
    print("  sudo apt-get install -f -y")
    return 0 if not missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
