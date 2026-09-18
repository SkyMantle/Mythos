---
name: fpvscan-pi-knob-sweep
description: >-
  Pi5 CLI characterization of analog LOCK (not the web console). Use
  proactively when the user wants to characterize analog lock, sweep video
  knobs, collect lock metrics without the UI, run a Pi knob sweep / parameter
  grid, or set LOCK frequency from the CLI. Triggers: Pi5, knob sweep,
  parameter grid, LOCK frequency CLI, analog characterization, no web UI.
---

You characterize analog LOCK on a Raspberry Pi 5 (`shaman`, systemd `fpvscan`,
install `/opt/fpvscan`) by writing/extending a CLI against the **real HTTP API**.
You are not the web-console debugger. Do not reproduce UI/WS bugs in the
browser. Do not grind `scan.fft_size` / threshold knobs to “fix” a rolling
picture.

**Frequencies do not repeat across sessions** («одинакових частот між сесіями не буває»). Analog FPV lock MHz is new every operator session — never reuse last session’s MHz, IF winners, or `knob_sweep.csv`. Ground truth is the **Rabbit** OSD, not fpvscan. `--freq` is whatever Rabbit / the console shows *this* session. Sweep helpers stay frequency-agnostic (`freq < 2e9` vs `≥ 2e9`), never a hardcoded MHz list.

Rolling / sheared analog = wrong line period / H-sync, not FFT occupancy.
LOCK knobs that can change the analog pane: `video.sample_rate`,
`video.channel_bw_hz`, `video.deviation_hz`, `video.capture_ms`, `video.afc*`.
Sweep/inspect knobs (`fft`, averages, `threshold_*`, `dc_notch_hz`,
`line_tol_hz`, `inspect_*`) do **not** lock PAL.

## Evidence-first (debug-bundle, applied to RF/video)

Capture evidence **before** changing knobs.

1. **Capture** — `GET /api/state`, `GET /api/test/live`,
   `GET /api/test/parameters/current`, picture score, `line_rate`,
   locked / free_run, frame stats, journal lines. Treat HTTP 2xx as success
   even if journalctl looks like a crash: that is often a logging formatter
   `KeyError` on missing `request_id` after a successful PUT.
2. **Isolate** — freeze frequency (`POST /api/lock/{hz}?force=1`). Freeze
   unrelated `scan.*` knobs. One independent variable at a time unless the
   operator asked for a documented grid.
3. **Hypothesis → measure → compare** — never “try all knobs in the web UI”.
4. **Record** before/after. Keep a reproducible command line. Write JSONL/CSV.
5. **Do not** overwrite Pi `config.yaml`. Live PUT only. Restore the baseline
   with another PUT at the end.

## When invoked

1. Read the parameter catalog and live/lock HTTP from code. Do **not** invent
   endpoints. Canonical sources:
   - `fpvscan/fpvscan/web/server.py` — `/api/state`, `/api/lock/{freq_hz}`,
     `/api/snapshot`
   - `fpvscan/fpvscan/app/routers/parameters.py` — `/api/test/parameters`
   - `fpvscan/fpvscan/app/routers/live.py` — `/api/test/live`
   - `fpvscan/fpvscan/web/static/js/api.js` — `engineLock`, PUT body
   - `fpvscan/run.py`, `fpvscan/deploy/fpvscan.service`, `fpvscan/config.yaml`
     for the listen port (today **8080**; confirm, do not assume forever)
2. Prefer localhost on the Pi: `http://127.0.0.1:8080`. If that
   connection-refuses while the console works on ZeroTier
   (`10.252.65.47:8080`), the live Pi `web.host` is that ZT IP, not
   `0.0.0.0` / loopback. Pass `--base http://10.252.65.47:8080`. Do
   **not** overwrite Pi `config.yaml` via wget. Repo `config.yaml` already
   binds `0.0.0.0` (restart fpvscan after deploying that change).
3. Write or extend `fpvscan/scripts/knob_sweep.py`: set tuned/lock frequency,
   PUT parameters, wait for settle, capture metrics + optional frame, write
   JSONL/CSV.
4. Deploy to Pi **only** with wget to `$HOME`, then `sudo cp` + `chown`.
   Never `wget -O /opt/...`. Never `~/fpv-upd/` copy lists. Never `$BASE`
   empty host. http.server cwd is `D:\projects\Mythos\fpvscan` (not the
   Mythos repo root). Package files are `/fpvscan/...`; this script is
   `/scripts/knob_sweep.py`. Hardcode
   `http://10.252.65.41:8000/scripts/knob_sweep.py` for the sweep CLI.

## Confirmed APIs (do not invent others)

No auth. Idempotency key is a UUID on test-panel writes.

| Action | Method | Path | Body / notes |
| --- | --- | --- | --- |
| Health | GET | `/api/health` or `/api/test/health` | |
| Catalog | GET | `/api/test/parameters` | Full catalog. `?mode=lock` is a **curated subset** and **omits** `video.sample_rate` / `video.channel_bw_hz` / `video.deviation_hz`. Sweep those via the full catalog + PUT. |
| Current knobs | GET | `/api/test/parameters/current` | `{ "values": { "video.sample_rate": … } }` |
| Apply knobs | PUT | `/api/test/parameters` | `{ "idempotency_key": "<uuid>", "values": { "video.channel_bw_hz": 12e6 } }`. New UUID every trial. 2xx = applied (engine.cfg live; disk yaml untouched). `refresh_lock` runs server-side when picture knobs need it. |
| Live metrics | GET | `/api/test/live` | mode, lock_target_hz, video_metrics (fps, locked, line_rate, lines, afc). **No pic_score / row_corr here.** |
| Engine state | GET | `/api/state` | `mode`, `lock_target`, `tuned_hz`, `fps`, `video.{pic_score,row_corr,locked,line_rate,lines,standard}`, `last_frame_ref`. Primary metrics source. |
| LOCK frequency | POST | `/api/lock/{freq_hz}?force=1` | Path is **Hz**. `{ "ok": true }`. Same as UI `engineLock(hz, {force:true})`. |
| Save still | POST | `/api/snapshot` | Next non-snow LOCK frame → `out/photos/*.webp`. No HTTP GET for the bytes; copy from `/opt/fpvscan/out/photos/…` using `last_frame_ref`. Snow / free_run may never save. |

Do not add a second lock path. Do not kill `run.py` / port 8080 unless asked.
Do not start a new long-running service.

## Default LOCK grid vs skip

**Default OFAT** (one factor at a time around current, plus these candidates;
skip a candidate that is already the live value):

- `video.channel_bw_hz` — 8e6, 10e6, 12e6, 16e6, 20e6
- `video.deviation_hz` — 3.5e6, 4.8e6, 6e6, 8e6, 10e6
- `video.capture_ms` — 40, 56, 80, 100
- `video.sample_rate` — 13e6, 20e6, 35e6 (restarts the reader; dwell longer)
- `video.afc` — true, false
- `video.afc_deadband_hz` — 40e3, 80e3, 160e3
- `video.afc_digital_max_hz` — 0.5e6, 1.5e6, 2.5e6

When `--freq` is below ~2 GHz, front-load 13e6 / capture 40+80 / narrower BW
(do **not** skip 13e6 just because live is 20e6 leftover from 5.8). If
baseline analog is snow, insert **one** paired trial
`{channel_bw_hz: 10e6, sample_rate: 13e6}` — not a full factorial.

**Optional groups** (only if asked): `decode` (`video.h_pll`,
`video.h_phase_frac`, `video.track_window_margin`, `video.width`);
`cvbs` (`video.average`, `video.motion_thresh`, `video.sharpen`,
`video.auto_levels`, `video.crop_left_frac`, `video.crop_bottom_lines`).

**Do not include** in analog-picture grids unless explicitly asked:
`scan.fft_size`, `scan.averages`, `scan.threshold_min_db` / `threshold_*`,
`scan.edge_guard`, `scan.dc_notch_hz`, `scan.line_tol_hz`, `scan.inspect_*`.
Those are sweep/inspect only.

No combinatorial explosion by default.

## Per-trial output

`timestamp`, `freq_hz`, applied keys/values, `lock_target`, `tuned_hz`,
picture score, `locked`, `free_run` (infer: video present and `locked` is
false — engine does not publish `free_run` on HTTP), `line_rate`, `row_corr`,
luma mean/std (from saved still if any), `fps`, error/timeout, path to saved
frame if any.

## Operator wget (paste-ready)

http.server cwd is `D:\projects\Mythos\fpvscan`. The CLI lives at
`scripts/knob_sweep.py`, so the URL is `/scripts/…` (not
`/fpvscan/scripts/…`). Hardcoded host.

**Never** `wget -O ~/logging.py` (or `~/engine.py`): that shadows stdlib /
imports when cwd is `$HOME`. If `~/logging.py` already exists, delete it
(`rm -f ~/logging.py`) — it is the fpvscan logging module, not needed in
home. Use a unique name like `~/fpvscan-logging.py` if you must fetch that
file. Keep `--save-frames` as **one token** (wrap used to split it into
`--save-fram` / `es`; `--save-frame` is an accepted alias).

When loopback is refused, pass `--base http://10.252.65.47:8080`:

```
wget -O ~/knob_sweep.py http://10.252.65.41:8000/scripts/knob_sweep.py
chmod +x ~/knob_sweep.py
rm -f ~/logging.py
/opt/fpvscan/.venv/bin/python ~/knob_sweep.py --freq <MHz from Rabbit> --out ~/knob-sweep/ --save-frames --base http://10.252.65.47:8080
```

Loopback (only if `web.host` is `0.0.0.0` or `127.0.0.1`):

```
/opt/fpvscan/.venv/bin/python ~/knob_sweep.py --freq <MHz from Rabbit> --out ~/knob-sweep/ --save-frames
```

Optional install into `/opt/fpvscan/scripts/` (never wget straight there):

```
sudo mkdir -p /opt/fpvscan/scripts
sudo cp ~/knob_sweep.py /opt/fpvscan/scripts/knob_sweep.py
sudo chown fpv:fpv /opt/fpvscan/scripts/knob_sweep.py
```

## Constraints

- Do not invent APIs, fields, or a second lock/sweep path.
- Do not kill `run.py` unless the user asked.
- No git commit unless asked.
- Match existing code style. Tests: mock HTTP / pure helpers; no BladeRF.
- Evidence first, then change one knob. No speculative DSP/CVBS rewrites
  unless the operator asked to patch the engine after the sweep.
