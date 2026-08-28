#!/usr/bin/env bash
# Розгортання на Raspberry Pi 5 / Ubuntu 24.04 arm64.
set -euo pipefail
APP=/opt/fpvscan

echo "== пакети =="
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential rsync \
  libbladerf2 bladerf bladerf-fpga-hostedxa4

echo "== користувач і каталог =="
sudo useradd -r -s /usr/sbin/nologin -G plugdev fpv 2>/dev/null || true
sudo mkdir -p $APP
sudo rsync -a --exclude .venv ./ $APP/
sudo chown -R fpv:fpv $APP

echo "== оточення python =="
sudo -u fpv python3 -m venv $APP/.venv
sudo -u fpv $APP/.venv/bin/pip install -q -r $APP/requirements.txt

echo "== перевірка приймача =="
bladeRF-cli -e info || echo "!! плату не видно: перевір кабель USB3 і правила udev"
bladeRF-cli -e "print rxvga1" >/dev/null 2>&1 || true

echo "== служба =="
sudo cp deploy/fpvscan.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fpvscan
sudo systemctl status fpvscan --no-pager -l | head -20

echo
echo "Веб-консоль: http://<адреса-в-zerotier>:8080"
