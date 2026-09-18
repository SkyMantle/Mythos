# fpvscan на Raspberry Pi 5 (Ubuntu 24.04, ARM64)

Цільова машина: Raspberry Pi 5, 64-bit Ubuntu, ZeroTier уже стоїть.

## Що скопіювати

Один файл з `fpvscan/dist/`:

- `fpvscan-pi5-arm64-*.tar.gz` — універсальний шлях
- або `fpvscan_*_arm64.deb` — якщо зручніше через dpkg

Зібрати на Windows (тягне aarch64 wheels з PyPI, не крос-компілює native code):

```powershell
cd D:\projects\Mythos\fpvscan
py -3 deploy\build_pi_package.py
```

На самій Pi (якщо будуєте там):

```bash
cd fpvscan
python3 deploy/build_pi_package.py
```

## Встановлення з tar.gz

```bash
tar -xzf fpvscan-pi5-arm64-0.1.0.tar.gz
cd fpvscan-pi5-arm64-0.1.0
sudo bash install.sh
```

Консоль: `http://<IP-ZeroTier>:8080`

Дізнатись адресу:

```bash
ip -4 addr show | grep zt
```

## Встановлення з .deb

```bash
sudo dpkg -i fpvscan_0.1.0_arm64.deb
sudo apt-get install -f -y
```

Якщо dpkg скаржиться на `libbladerf2` / `bladerf` — спочатку tar.gz (скрипт підключить universe/PPA).

## ZeroTier / bind

Сервер **слухає 0.0.0.0:8080** — усі інтерфейси, включно з ZeroTier.
Окремої проброски портів, nginx і прив’язки до конкретного ZT-IP **немає**.
Це достатньо, щоб відкрити UI з іншого вузла мережі ZeroTier.

Щоб слухати лише ZeroTier:

```yaml
web:
  host: 10.x.x.x   # IP цього вузла в ZeroTier
  port: 8080
```

Потім `sudo systemctl restart fpvscan`.

Якщо увімкнений `ufw`, інсталятор відкриває 8080 лише на інтерфейсах `zt*`.

## Що в пакеті, що з apt

У tar.gz / .deb:

- код fpvscan, `config.yaml`, systemd-юніт, udev-правила bladeRF
- Python wheels під CPython 3.12 aarch64 (numpy, scipy, numba, Pillow, FastAPI, uvicorn, …)

На Pi через `apt` (ставить `install.sh`):

```
python3 python3-venv python3-dev python3-pip
build-essential rsync ffmpeg libusb-1.0-0
libbladerf2 bladerf bladerf-fpga-hostedxa4
```

Не бандляться (системні / USB):

- **libbladeRF** і FPGA-прошивка xA4 — пакети `libbladerf2`, `bladerf`, `bladerf-fpga-hostedxa4`
- **ffmpeg** — запис відео
- **ZeroTier** — уже має бути
- **udev**: інсталятор кладе `88-nuand-bladerf.rules` і `99-pwm-gpio.rules`. Користувач служби `fpv` у `plugdev` і `gpio`. Якщо плата не видно — переткнути USB 3.0.

## Служба

```bash
sudo systemctl status fpvscan
sudo journalctl -u fpvscan -f
sudo systemctl restart fpvscan
```

Конфіг: `/opt/fpvscan/config.yaml` (при повторній установці не затирається).

## Ротатор (серво, GPIO 12)

Type 2 — Linux PWM через sysfs (`/sys/class/pwm/pwmchip0`, канал 0), 50 Гц, 300–2500 µs.

У `/boot/firmware/config.txt`:

```
dtoverlay=pwm,pin=12,func=4
```

Служба `fpv` має бути в `gpio` (`SupplementaryGroups=plugdev gpio`). udev: `deploy/99-pwm-gpio.rules`.

Перезавантажити. Перевірка: `ls /sys/class/pwm/pwmchip0/pwm0` (після `echo 0 > export`). Без overlay канал `pwm0` не з’являється — це не помилка азимута в UI.

## Оновити лише config.yaml

Не пишіть одразу в `/opt/fpvscan` — там `permission denied`. Скопіюйте в домашню теку, потім `sudo`.

З Windows (підставте IP ZeroTier і SSH-користувача, часто `ubuntu`):

```powershell
scp D:\projects\Mythos\fpvscan\config.yaml ubuntu@<ZT-IP>:~/config.yaml
```

На Pi:

```bash
sudo cp ~/config.yaml /opt/fpvscan/config.yaml
sudo chown fpv:fpv /opt/fpvscan/config.yaml
sudo systemctl restart fpvscan
```

Перевірка, що служба взяла новий файл:

```bash
sudo systemctl status fpvscan
grep -E "threshold_mode|threshold_offset_db|threshold_db|hit_filter|accept_energy|confirm_hits|step_hz" /opt/fpvscan/config.yaml
```

Очікуйте `threshold_mode: auto`, `threshold_offset_db: 1.2`, `threshold_db: 1.0`, `step_hz: 12.0e6`, `hit_filter: all`, `accept_energy: false`, `confirm_hits: 1`. Live yaml на Pi цим файлом не перезаписувати; «Публікувати енергію без PAL» лишається повзунком.
