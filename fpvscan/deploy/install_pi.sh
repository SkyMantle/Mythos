#!/usr/bin/env bash
# Розгортання fpvscan на Raspberry Pi 5 / Ubuntu 24.04 arm64.
# Працює з git-клону (./deploy/install_pi.sh) і з архіву (sudo bash install.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${FPVSCAN_SRC:-$(cd "$SCRIPT_DIR/.." && pwd)}"
APP="${FPVSCAN_APP:-/opt/fpvscan}"
FROM_DEB=0
SKIP_COPY="${FPVSCAN_SKIP_COPY:-0}"
SKIP_APT="${FPVSCAN_SKIP_APT:-0}"

for arg in "$@"; do
  case "$arg" in
    --from-deb) FROM_DEB=1; SKIP_COPY=1; SKIP_APT=1 ;;
    --skip-apt) SKIP_APT=1 ;;
    --skip-copy) SKIP_COPY=1 ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "Запусти від root: sudo bash $0"
  exit 1
fi

ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
  echo "!! очікується aarch64 (Pi 5 64-bit), зараз: $ARCH"
  echo "   продовжую, але wheels з архіву можуть не підійти"
fi

export DEBIAN_FRONTEND=noninteractive

apt_bladerf() {
  apt-get install -y libbladerf2 bladerf bladerf-fpga-hostedxa4
}

if [ "$SKIP_APT" != "1" ]; then
  echo "== пакети =="
  apt-get update
  apt-get install -y software-properties-common || true
  add-apt-repository -y universe >/dev/null 2>&1 || true
  apt-get install -y \
    python3 python3-venv python3-dev python3-pip \
    build-essential rsync ffmpeg libusb-1.0-0 adduser \
    udev systemd
  if ! apt_bladerf; then
    echo "!! пакети bladeRF не знайдені у поточних репозиторіях"
    echo "   спроба PPA Nuand..."
    add-apt-repository -y ppa:bladerf/bladerf || true
    apt-get update
    apt_bladerf
  fi
fi

if [ "$FROM_DEB" = 1 ]; then
  echo "== доустановка після dpkg =="
fi

echo "== користувач fpv =="
getent group plugdev >/dev/null || groupadd plugdev
getent group gpio >/dev/null || groupadd gpio
if ! id -u fpv >/dev/null 2>&1; then
  useradd -r -s /usr/sbin/nologin -M -G plugdev,gpio fpv
else
  usermod -aG plugdev fpv || true
fi
usermod -aG plugdev,gpio fpv || true

echo "== udev bladeRF / PWM =="
RULES_SRC="$SRC/deploy/88-nuand-bladerf.rules"
if [ -f "$RULES_SRC" ]; then
  install -m 644 "$RULES_SRC" /etc/udev/rules.d/88-nuand-bladerf.rules
fi
PWM_RULES="$SRC/deploy/99-pwm-gpio.rules"
if [ -f "$PWM_RULES" ]; then
  install -m 644 "$PWM_RULES" /etc/udev/rules.d/99-pwm-gpio.rules
fi
udevadm control --reload-rules 2>/dev/null || true
udevadm trigger 2>/dev/null || true

if [ "$SKIP_COPY" != "1" ]; then
  echo "== файли -> $APP =="
  mkdir -p "$APP"
  KEEP_CFG=""
  if [ -f "$APP/config.yaml" ]; then
    KEEP_CFG="$(mktemp)"
    cp -a "$APP/config.yaml" "$KEEP_CFG"
  fi
  rsync -a \
    --exclude '.venv/' \
    --exclude 'out/' \
    --exclude '.git/' \
    --exclude 'dist/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache/' \
    --exclude 'caps/' \
    "$SRC/" "$APP/"
  if [ -n "$KEEP_CFG" ]; then
    cp -a "$KEEP_CFG" "$APP/config.yaml"
    cp -a "$SRC/config.yaml" "$APP/config.yaml.pkg" || true
    rm -f "$KEEP_CFG"
    echo "   збережено наявний config.yaml (нові типові — config.yaml.pkg)"
  fi
fi

mkdir -p "$APP/out"
chown -R fpv:fpv "$APP"

WHEELS=""
for cand in "$APP/wheels" "$SRC/wheels"; do
  if ls "$cand"/*.whl >/dev/null 2>&1; then
    WHEELS="$cand"
    break
  fi
done

echo "== оточення python =="
as_fpv() {
  runuser -u fpv -- "$@"
}

if [ ! -x "$APP/.venv/bin/python" ]; then
  as_fpv python3 -m venv "$APP/.venv"
fi
as_fpv "$APP/.venv/bin/python" -m pip install -q --upgrade pip

REQ="$APP/deploy/requirements-runtime.txt"
if [ ! -f "$REQ" ]; then
  REQ="$APP/requirements.txt"
fi

if [ -n "$WHEELS" ]; then
  echo "   wheels: $WHEELS"
  if ! as_fpv "$APP/.venv/bin/pip" install --no-index --find-links="$WHEELS" -r "$REQ"; then
    echo "!! не всі wheels підійшли до $(python3 --version), добираю з PyPI"
    as_fpv "$APP/.venv/bin/pip" install --find-links="$WHEELS" -r "$REQ"
  fi
else
  echo "   wheels немає — ставлю з PyPI (потрібен інтернет)"
  as_fpv "$APP/.venv/bin/pip" install -r "$REQ"
fi

echo "== systemd =="
install -m 644 "$APP/deploy/fpvscan.service" /etc/systemd/system/fpvscan.service
systemctl daemon-reload
systemctl enable fpvscan
systemctl restart fpvscan || systemctl start fpvscan
sleep 1
systemctl status fpvscan --no-pager -l | head -25 || true

echo "== перевірка приймача =="
if command -v bladeRF-cli >/dev/null 2>&1; then
  bladeRF-cli -e info || echo "!! плату не видно: USB 3.0, живлення 5 В / 5 А, udev (переткни кабель)"
else
  echo "!! bladeRF-cli немає — пакет bladerf не встановився"
fi

if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -q "Status: active"; then
  echo "== ufw: відкриваю 8080 лише на інтерфейсах ZeroTier =="
  ZT_IFACES="$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^zt' || true)"
  if [ -n "$ZT_IFACES" ]; then
    while IFS= read -r iface; do
      [ -z "$iface" ] && continue
      ufw allow in on "$iface" to any port 8080 proto tcp comment 'fpvscan-zerotier' || true
    done <<< "$ZT_IFACES"
  else
    echo "!! ufw увімкнений, але інтерфейсу zt* ще немає."
    echo "   після ZeroTier: sudo ufw allow in on zt+ to any port 8080 proto tcp"
  fi
fi

ZT_IP="$(ip -4 -o addr show 2>/dev/null | awk '/ zt/{print $4}' | cut -d/ -f1 | head -1 || true)"
echo
echo "Готово. Служба: sudo systemctl status fpvscan"
echo "Логи:        sudo journalctl -u fpvscan -f"
if [ -n "$ZT_IP" ]; then
  echo "Консоль:     http://${ZT_IP}:8080"
else
  echo "Консоль:     http://<адреса-ZeroTier>:8080"
  echo "Адресу ZT:   ip -4 addr show | grep zt"
fi
echo
echo "Сервер слухає 0.0.0.0:8080 (усі інтерфейси, включно з ZeroTier)."
echo "Щоб слухати лише ZT: у $APP/config.yaml постав web.host: <IP-ZeroTier>"
echo "                     і sudo systemctl restart fpvscan"
