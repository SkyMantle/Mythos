# Mythos

Репозиторій зараз містить один застосунок — **fpvscan**: серверний
сканер аналогового FPV-відео (400 МГц — 6 ГГц) на bladeRF 2.0 micro
xA4. Приймач працює тільки на прийом.

| Документ | Що там |
|----------|--------|
| [fpvscan/README.md](fpvscan/README.md) | Розгортання, архітектура SWEEP / INSPECT / LOCK, консоль, HTTP API, пастки |
| [fpvscan-technical.pdf](fpvscan-technical.pdf) | Розширений технічний опис |
| [fpvscan/config.yaml](fpvscan/config.yaml) | Усі робочі параметри з коментарями |

## Швидкий старт без плати

У Cloud Agent і будь-якій машині без bladeRF:

```bash
bash .cursor/install.sh
cd fpvscan && ./.venv/bin/python run.py --driver sim --port 8080
```

Консоль: http://127.0.0.1:8080

Перевірка тракту без браузера:

```bash
cd fpvscan
./.venv/bin/python scripts/selftest.py
./.venv/bin/python tests/test_engine_lock.py
```

## Залізо (Pi 5)

```bash
git clone <repo> && cd Mythos/fpvscan
./deploy/install_pi.sh
```

Далі — розділ «Розгортання на Pi 5» у [fpvscan/README.md](fpvscan/README.md).
