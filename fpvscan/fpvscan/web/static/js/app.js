import {
  TASK_TOOLS, sameValue, displayOf, storedFromDisplay, formatDefault,
  clampToSpec, optValue, optLabel, formatList, displayUnit,
} from "./catalog.js"
import {
  TestClient, engineLock, engineSweep, engineClear, engineSnapshot,
  engineRecord, engineState,
} from "./api.js"

const F0 = 400e6, F1 = 6000e6
const UI_LS = "fpvscan.ui.v1"
const client = new TestClient()

const $ = id => document.getElementById(id)
function on(id, ev, fn) {
  const el = $(id)
  if (!el) return
  el.addEventListener(ev, fn)
}
function bindClick(id, fn) {
  const el = $(id)
  if (!el) return
  el.onclick = fn
}
const live = {
  mode: "",
  freqHz: null,
  locked: false,
  fps: 0,
  standard: "—",
  lineRate: null,
  lines: null,
  afcHz: null,
  freqErrHz: null,
  afcPegged: false,
  clipFrac: 0,
  recording: false,
  lastFrameAt: 0,
  lastWs: 0,
  metrics: {},
  frame: null,
  lastShot: null,
  lastRec: null,
}

const specView = {
  bins: null,
  center_hz: null,
  span_hz: null,
  floor_db: -90,
  bw_hz: null,
  cursor_hz: null,
  peak_db: null,
  nfft: null,
}

const grid = {
  pass_index: 0,
  pass_count: 0,
  current_hz: null,
  start_hz: null,
  stop_hz: null,
  step_hz: null,
  progress_01: 0,
  visiting_hz: [],
  passes_done: 0,
}

const LERP_MS = 160
const SYMPTOM_FIELDS = [
  { key: "picture_jump", label: "Стрибки картинки", type: "bool", required: false },
  { key: "line_tearing", label: "Рядковість / розрив", type: "bool", required: false },
  { key: "spectrum_stutter", label: "Спектр смикається", type: "bool", required: false },
]
const SYMPTOM_KEYS = new Set(SYMPTOM_FIELDS.map(f => f.key))
const autoFill = {
  frequency_mhz: { dirty: false, last: null },
  bandwidth_note: { dirty: false, last: null },
}

let hits = new Map()
let current = null
let sock = null
let fpsHist = []
let values = {}
let defaults = {}
let catalog = []
let debounceTimer = 0
let pendingPatch = {}
let editingObsId = null
let panelTab = "params"
let autoRelock = true
let pendingNow = new Set()
let pendingWhy = {}
let toolsetMode = ""
let toolsetKeys = null

const LOCK_RETUNE = new Set([
  "video.sample_rate", "video.capture_ms", "video.hunt",
  "video.hunt_every", "video.hunt_drop", "sdr.settle_us",
])
const LOCK_HIDE = new Set(["video.sample_rate", "sdr.gain_db", "video.h_pll", "video.pll_enable"])
const PANEL_SKIP = new Set(["scan_grid", "hit_filter", "pll", "picture_jump", "phase_tear", "phase_tear_h", "phase_tear_v"])

const dirty = { spec: false, grid: false, frame: false, hud: true }
const motion = {
  cursorHz: null,
  cursorTarget: null,
  cursorFrom: null,
  cursorFromAt: 0,
  playHz: null,
  playTarget: null,
  playFrom: null,
  playFromAt: 0,
}

let rafStarted = false

function retargetHz(kind, hz) {
  if (hz == null || !Number.isFinite(hz)) return
  const tgtKey = kind + "Target"
  if (motion[tgtKey] === hz) return
  const curKey = kind + "Hz"
  const fromKey = kind + "From"
  const atKey = kind + "FromAt"
  motion[fromKey] = motion[curKey] ?? hz
  motion[atKey] = performance.now()
  motion[tgtKey] = hz
}
let latestFrame = null
let picShownUrl = null
let picPendingUrl = null
let picGen = 0
const specStamps = []
let lastHudAt = 0

const gridCv = $("grid-pass")
const gctx = gridCv ? gridCv.getContext("2d") : null
const fftCv = $("fft")
const fftx = fftCv ? fftCv.getContext("2d") : null

function reduceMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches
}

function markSpectrumMsg() {
  const t = performance.now()
  specStamps.push(t)
  while (specStamps.length && specStamps[0] < t - 1000) specStamps.shift()
}

function specRatePerSec() {
  const t = performance.now()
  while (specStamps.length && specStamps[0] < t - 1000) specStamps.shift()
  return specStamps.length
}

function stepMotion(now) {
  const snap = reduceMotion()
  let moving = false
  const step = (curKey, tgtKey, fromKey, atKey) => {
    const tgt = motion[tgtKey]
    if (tgt == null) return
    if (motion[curKey] == null || snap) {
      motion[curKey] = tgt
      return
    }
    const t0 = motion[atKey] || now
    const u = Math.min(1, (now - t0) / LERP_MS)
    const from = motion[fromKey] ?? motion[curKey]
    motion[curKey] = from + (tgt - from) * u
    if (u < 1) moving = true
    else motion[curKey] = tgt
  }
  step("cursorHz", "cursorTarget", "cursorFrom", "cursorFromAt")
  step("playHz", "playTarget", "playFrom", "playFromAt")
  return moving
}

function b64ToBytes(b64) {
  const bin = atob(b64)
  const n = bin.length
  const out = new Uint8Array(n)
  for (let i = 0; i < n; i++) out[i] = bin.charCodeAt(i)
  return out
}

function frameMime(bytes) {
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return "image/jpeg"
  }
  if (bytes.length >= 12 && bytes[0] === 0x52 && bytes[1] === 0x49
      && bytes[2] === 0x46 && bytes[3] === 0x46) {
    return "image/webp"
  }
  if (bytes.length >= 8 && bytes[0] === 0x89 && bytes[1] === 0x50) {
    return "image/png"
  }
  return "image/jpeg"
}

function revokePics() {
  picGen += 1
  if (picPendingUrl) {
    URL.revokeObjectURL(picPendingUrl)
    picPendingUrl = null
  }
  if (picShownUrl) {
    URL.revokeObjectURL(picShownUrl)
    picShownUrl = null
  }
}

function presentFrame(d) {
  const img = $("pic")
  if (!img || !d || !d.img) return
  let bytes
  try {
    bytes = b64ToBytes(d.img)
  } catch {
    return
  }
  const url = URL.createObjectURL(new Blob([bytes], { type: frameMime(bytes) }))
  if (picPendingUrl) URL.revokeObjectURL(picPendingUrl)
  picPendingUrl = url
  const gen = ++picGen
  img.onload = () => {
    if (gen !== picGen) return
    if (picShownUrl && picShownUrl !== url) URL.revokeObjectURL(picShownUrl)
    picShownUrl = url
    if (picPendingUrl === url) picPendingUrl = null
    img.style.display = "block"
    const hint = $("hint")
    if (hint) hint.hidden = true
  }
  img.src = url
}

function applyFrameHud(d) {
  if (d.freq_hz) {
    current = d.freq_hz
    live.freqHz = d.freq_hz
  }
  if (d.standard) live.standard = d.standard
  if (d.line_rate != null) live.lineRate = d.line_rate
  if (d.lines != null) live.lines = d.lines
  if (d.locked != null) live.locked = !!d.locked
  ingestAfcLimit(d)
  put("v-f", fmt(d.freq_hz || live.freqHz))
  put("v-std", live.standard || "—")
  metric("v-line", live.lineRate, " Гц")
  metric("v-lines", live.lines, "")
  if (live.afcHz != null) put("v-afc", fmtAfc(live.afcHz))
  const lk = $("v-lock")
  if (lk) lk.innerHTML = `<i class="dot ${live.locked ? "ok" : ""}"></i>${live.locked ? "є" : "нема"}`
  fpsHist.push(Date.now())
  while (fpsHist.length && fpsHist[0] < Date.now() - 3000) fpsHist.shift()
  live.fps = fpsHist.length / 3
  metric("v-fps", live.fps.toFixed(1), " /с")
}

function updateWatchStrip() {
  const el = $("watch-strip")
  if (!el) return
  const watching = live.mode === "LOCK"
  const age = live.lastFrameAt ? Date.now() - live.lastFrameAt : null
  const rate = specRatePerSec()
  const fps = live.fps
  put("w-fps", watching ? `кадрів ${fps.toFixed(1)}/с` : "кадрів —/с")
  put("w-lines", live.lines != null ? `рядків ${live.lines}` : "рядків —")
  put("w-sync", `синхро ${live.locked ? "є" : "нема"}`)
  put("w-spec", `спектр ${rate}/с`)
  put("w-age", age == null ? "кадр — мс" : `кадр ${age} мс`)
  const noteEl = $("w-note")
  if (noteEl) noteEl.textContent = ""
  el.classList.remove("lag")
  updateAfcLimitUi()
  applyLockAutofill()
}

function paint(now) {
  try {
    stepMotion(now)
    if (dirty.frame && latestFrame) {
      const d = latestFrame
      latestFrame = null
      dirty.frame = false
      presentFrame(d)
      applyFrameHud(d)
    }
    dirty.spec = false
    drawMiniSpectrum()
    dirty.grid = false
    drawGridStrip()
    if (dirty.hud || now - lastHudAt > 200) {
      dirty.hud = false
      lastHudAt = now
      updateWatchStrip()
      updateHealth()
      updateFftReadout()
      updateMediaPath()
    }
  } catch (err) {
    console.error("малювання:", err)
  }
  requestAnimationFrame(paint)
}

function startPaint() {
  if (rafStarted) return
  rafStarted = true
  requestAnimationFrame(paint)
}

function connect() {
  try {
    sock = new WebSocket((location.protocol === "https:" ? "wss" : "ws") + "://" + location.host + "/ws")
  } catch (err) {
    setWs(false, "нема")
    setTimeout(connect, 1500)
    return
  }
  const ws = sock
  ws.onopen = () => setWs(true, "є")
  ws.onerror = () => setWs(false, "помилка")
  ws.onclose = () => { setWs(false, "нема"); setTimeout(connect, 1500) }
  ws.onmessage = e => {
    try {
      live.lastWs = Date.now()
      const m = JSON.parse(e.data)
      if (m.type === "spectrum") {
        markSpectrumMsg()
        applySpectrumPayload(m.data)
      } else if (m.type === "detection") {
        hits.set(Math.round(m.data.freq_hz / 1e6), m.data)
        renderHits()
        dirty.grid = true
      } else if (m.type === "frame") queueFrame(m.data)
      else if (m.type === "state") {
        const d = m.data || {}
        if (d.spectrum || Array.isArray(d.bins)) markSpectrumMsg()
        applyState(d)
      }
      else if (m.type === "notice") note(m.data.text, m.data.level === "error")
    } catch (err) {
      console.error("обробка повідомлення:", err, String(e.data).slice(0, 200))
      note("помилка в браузері: " + err.message, true)
    }
  }
}

startPaint()
connect()

const fmt = hz => {
  if (hz == null || !Number.isFinite(hz)) return "—"
  return hz >= 1e9 ? (hz / 1e9).toFixed(3) + " ГГц" : (hz / 1e6).toFixed(1) + " МГц"
}
function fmtAfc(hz) {
  const a = Number(hz) || 0
  if (Math.abs(a) >= 1e6) return (a / 1e6).toFixed(2) + " МГц"
  return (a / 1e3).toFixed(0) + " кГц"
}

function ingestAfcLimit(src) {
  if (!src || typeof src !== "object") return
  const vm = src.video_metrics && typeof src.video_metrics === "object" ? src.video_metrics : {}
  const video = src.video && typeof src.video === "object" ? src.video : {}
  const spec = src.spectrum && typeof src.spectrum === "object" ? src.spectrum : {}
  const flag = src.afc_pegged ?? vm.afc_pegged ?? video.afc_pegged ?? spec.afc_pegged
  if (flag != null) live.afcPegged = !!flag
  const err = src.freq_err_hz ?? vm.freq_err_hz ?? video.freq_err_hz ?? spec.freq_err_hz
  if (err != null && Number.isFinite(Number(err))) live.freqErrHz = Number(err)
  const afc = src.afc_hz ?? vm.afc_hz ?? video.afc_hz ?? spec.afc_hz
  if (afc != null && Number.isFinite(Number(afc))) live.afcHz = Number(afc)
}

function afcAtLimit() {
  return !!live.afcPegged
}

function afcLimitText() {
  if (!afcAtLimit()) return ""
  const err = Math.abs(Number(live.freqErrHz) || 0)
  return `AFC на межі · помилка ${(err / 1e6).toFixed(2)} МГц`
}

function updateAfcLimitUi() {
  const text = live.mode === "LOCK" && afcAtLimit() ? afcLimitText() : ""
  const watch = $("w-afc")
  if (watch) {
    watch.textContent = text
    watch.hidden = !text
  }
  const jump = $("afc-limit-note")
  if (jump) {
    jump.textContent = text
    jump.hidden = !text
  }
}

function gridRange() {
  const lo = grid.start_hz || F0
  const hi = grid.stop_hz && grid.stop_hz > lo ? grid.stop_hz : F1
  return [lo, hi]
}

function hzToX(hz, width) {
  const [lo, hi] = gridRange()
  return (hz - lo) / (hi - lo) * width
}

function ingestGrid(src) {
  if (!src || typeof src !== "object") return
  const g = src.grid || src
  if (g.pass_index != null) grid.pass_index = Number(g.pass_index)
  if (g.pass_count != null) grid.pass_count = Number(g.pass_count)
  if (g.progress_01 != null) grid.progress_01 = Number(g.progress_01)
  if (g.current_hz != null) {
    grid.current_hz = Number(g.current_hz)
    retargetHz("play", grid.current_hz)
  }
  if (g.start_hz != null) grid.start_hz = Number(g.start_hz)
  if (g.stop_hz != null) grid.stop_hz = Number(g.stop_hz)
  if (g.step_hz != null) grid.step_hz = Number(g.step_hz)
  if (g.passes_done != null) grid.passes_done = Number(g.passes_done)
  if (Array.isArray(g.visiting_hz)) grid.visiting_hz = g.visiting_hz
  dirty.grid = true
}

function drawGridStrip() {
  if (!gctx) return
  const W = gridCv.width, H = gridCv.height
  gctx.fillStyle = "#0d0f0e"
  gctx.fillRect(0, 0, W, H)

  const p = Math.max(0, Math.min(1, Number(grid.progress_01) || 0))
  if (p > 0) {
    gctx.fillStyle = "rgba(62, 154, 114, 0.28)"
    gctx.fillRect(0, H - 7, Math.round(W * p), 5)
  }

  for (const f of grid.visiting_hz || []) {
    const x = hzToX(f, W)
    gctx.fillStyle = "rgba(62, 154, 114, 0.7)"
    gctx.fillRect(x, 3, 1, H - 6)
  }

  const [lo, hi] = gridRange()
  const marks = [433e6, 915e6, 1280e6, 2440e6, 3300e6, 5800e6]
  gctx.fillStyle = "#6a7370"
  for (const f of marks) {
    if (f < lo || f > hi) continue
    gctx.fillRect(hzToX(f, W), 0, 1, 4)
  }

  for (const d of hits.values()) {
    const x = hzToX(d.freq_hz, W)
    const on = current && Math.abs(current - d.freq_hz) < 2e6
    gctx.fillStyle = on ? "#e6ece9" : "#3e9a72"
    gctx.fillRect(x - 1, 2, 2, H - 4)
  }

  const pos = motion.playHz ?? grid.current_hz
  if (pos) {
    gctx.fillStyle = "#c07a42"
    gctx.fillRect(Math.round(hzToX(pos, W)), 0, 2, H)
  }

  const idx = grid.pass_index || 0
  const cnt = grid.pass_count || 0
  const done = grid.passes_done || 0
  put("grid-pass-label", cnt
    ? `прохід ${idx} / ${cnt} · ${done} ок`
    : "прохід —")
  const span = (grid.start_hz && grid.stop_hz)
    ? `${fmt(grid.start_hz)} – ${fmt(grid.stop_hz)}`
    : ""
  put("grid-pos", [pos ? fmt(pos) : null, span].filter(Boolean).join(" · ") || "—")
  put("cov", "покриття " + Math.round(p * 100) + "%")
}

function displayedFreqHz() {
  return live.freqHz || current || live.metrics.lock_target || live.metrics.frequency_hz || live.metrics.tuned_hz || null
}

function currentBwHz() {
  if (specView.bw_hz) return specView.bw_hz
  return Number(values["video.channel_bw_hz"]) || 10e6
}

function currentSpanHz() {
  return specView.span_hz || Number(values["video.sample_rate"]) || 20e6
}

function applySpectrumPayload(d) {
  if (!d || typeof d !== "object") return
  const src = d.spectrum || d
  if (Array.isArray(src.bins) && src.bins.length) specView.bins = src.bins
  if (src.center_hz != null) specView.center_hz = Number(src.center_hz)
  if (src.span_hz != null) specView.span_hz = Number(src.span_hz)
  if (src.floor_db != null) specView.floor_db = Number(src.floor_db)
  if (src.bw_hz != null) specView.bw_hz = Number(src.bw_hz)
  if (src.cursor_hz != null) {
    specView.cursor_hz = Number(src.cursor_hz)
    retargetHz("cursor", specView.cursor_hz)
    if (live.mode === "LOCK") retargetHz("play", specView.cursor_hz)
  } else if (src.center_hz != null) {
    retargetHz("cursor", Number(src.center_hz))
  }
  if (src.peak_db != null) specView.peak_db = Number(src.peak_db)
  if (src.nfft != null) specView.nfft = Number(src.nfft)
  dirty.spec = true
}

function updateFftReadout() {
  const hz = displayedFreqHz()
  const el = $("fft-num")
  if (!el) return
  el.textContent = hz ? (hz / 1e6).toFixed(1) : "—"
}

function drawMiniSpectrum() {
  if (!fftx || !fftCv) return
  if (fftCv.width < 8) fftCv.width = 320
  if (fftCv.height < 8) fftCv.height = 88
  const W = fftCv.width, H = fftCv.height
  fftx.fillStyle = "#141816"
  fftx.fillRect(0, 0, W, H)

  const center = specView.center_hz || displayedFreqHz()
  const span = currentSpanHz()
  const bw = currentBwHz()
  const cursor = motion.cursorHz ?? specView.cursor_hz ?? displayedFreqHz()
  const floor = specView.floor_db ?? -90
  const thresh = Number(values["scan.threshold_db"])
  const scale = 45 + (Number.isFinite(thresh) ? Math.max(0, 8 - thresh) : 0)

  if (specView.bins && specView.bins.length) {
    const n = specView.bins.length
    const g = fftx.createLinearGradient(0, H, 0, 0)
    g.addColorStop(0, "#1a5558")
    g.addColorStop(0.4, "#2f8a48")
    g.addColorStop(0.72, "#7aad3a")
    g.addColorStop(1, "#d4c84a")
    fftx.fillStyle = g
    const barW = Math.max(1, W / n)
    for (let i = 0; i < n; i++) {
      const u = Math.max(0, Math.min(1, (specView.bins[i] - floor) / scale))
      const h = Math.max(1, u * (H - 4))
      fftx.fillRect(Math.floor(i * barW), H - h, Math.max(1, Math.ceil(barW) - 0.4), h)
    }
  }

  if (center && span && bw) {
    const loHz = center - span / 2
    const x0 = ((center - bw / 2 - loHz) / span) * W
    const x1 = ((center + bw / 2 - loHz) / span) * W
    const left = Math.max(2, Math.min(W - 4, x0))
    const right = Math.max(left + 4, Math.min(W - 2, x1))
    fftx.strokeStyle = "#b85c3a"
    fftx.lineWidth = 1.5
    fftx.beginPath()
    fftx.moveTo(left, 10)
    fftx.lineTo(left, 3)
    fftx.lineTo(right, 3)
    fftx.lineTo(right, 10)
    fftx.stroke()
  }

  if (center && span && cursor) {
    const x = ((cursor - (center - span / 2)) / span) * W
    if (x >= 0 && x <= W) {
      fftx.strokeStyle = "#e6c84a"
      fftx.lineWidth = 1
      fftx.beginPath()
      fftx.moveTo(Math.round(x) + 0.5, 0)
      fftx.lineTo(Math.round(x) + 0.5, H)
      fftx.stroke()
    }
  }
  updateFftReadout()
}

function setWatching(on) {
  const main = document.querySelector("main")
  if (main) main.classList.toggle("watching", !!on)
  const win = $("fft-win")
  if (win) win.hidden = false
  if (on) {
    dirty.spec = true
    dirty.hud = true
  }
}

function renderHits() {
  const el = $("hits")
  if (!el) return
  if (!hits.size) {
    el.innerHTML = "<p class=\"empty\">Поки нічого. Кандидат потрапляє сюди лише після підтвердження рядкової частоти.</p>"
    return
  }
  const list = [...hits.values()].sort((a, b) => b.snr_db - a.snr_db)
  el.innerHTML = list.map(d => `
    <div class="hit ${current && Math.abs(current - d.freq_hz) < 2e6 ? "on" : ""}"
         data-f="${d.freq_hz}" role="button" tabindex="0">
      <div class="f">${fmt(d.freq_hz)}</div>
      <div class="m">
        <span>${d.channel || d.band}</span>
        <span>${d.standard}</span>
        <span>${d.snr_db} дБ</span>
        <span>${(d.bandwidth_hz / 1e6).toFixed(0)} МГц</span>
      </div>
    </div>`).join("")
  el.querySelectorAll(".hit").forEach(n => {
    const go = () => lock(parseFloat(n.dataset.f))
    n.onclick = go
    n.onkeydown = e => { if (e.key === "Enter") go() }
  })
}

function updateMediaPath() {
  const el = $("media-path")
  if (!el) return
  const bits = []
  if (live.lastShot) bits.push("знімок " + live.lastShot)
  if (live.lastRec) bits.push("запис " + live.lastRec)
  el.hidden = !bits.length
  el.textContent = bits.join(" · ")
}

function rememberMediaPath(text) {
  const shot = String(text).match(/^знімок:\s+(\S+)/)
  const recStart = String(text).match(/^запис:\s+(\S+)/)
  const recDone = String(text).match(/^(\S+\.mp4):/)
  if (shot) live.lastShot = "out/photos/" + shot[1]
  if (recStart) live.lastRec = "out/video/" + recStart[1]
  if (recDone) live.lastRec = "out/video/" + recDone[1]
  updateMediaPath()
}

function note(text, err) {
  rememberMediaPath(text)
  if (err) console.warn("fpvscan:", text)
}

function setWs(up, txt) {
  const el = $("s-ws")
  if (!el) return
  el.textContent = txt
  el.className = up ? "up" : "down"
}

function put(id, text) {
  const el = $(id)
  if (el) el.textContent = text
}

function metric(id, num, unit) {
  const el = $(id)
  if (!el) return
  el.innerHTML = `${num}<span class="u">${unit || ""}</span>`
}

function queueFrame(d) {
  if (!d || !d.img) return
  latestFrame = d
  live.frame = d
  live.lastFrameAt = Date.now()
  if (d.freq_hz) {
    current = d.freq_hz
    live.freqHz = d.freq_hz
  }
  if (d.standard) live.standard = d.standard
  if (d.line_rate != null) live.lineRate = d.line_rate
  if (d.lines != null) live.lines = d.lines
  if (d.locked != null) live.locked = !!d.locked
  ingestAfcLimit(d)
  dirty.frame = true
  dirty.hud = true
}

function decodeHealth() {
  if (live.mode !== "LOCK") return { cls: "off", text: "очікування", label: "декод: немає потоку" }
  const age = Date.now() - live.lastFrameAt
  if (!live.lastFrameAt || age > 4000) return { cls: "err", text: "мовчить", label: "декод: потік мовчить" }
  if (live.locked) return { cls: "ok", text: "тримає", label: "декод: синхро тримає" }
  return { cls: "warn", text: "без синхро", label: "декод: кадри є, синхро немає" }
}

function updateHealth() {
  const h = decodeHealth()
  const el = $("s-health")
  if (el) {
    el.innerHTML = `<i class="dot ${h.cls === "warn" ? "" : h.cls}"></i>${h.label}`
  }
  const bar = $("bar-health")
  if (bar) {
    bar.innerHTML = `<i class="dot ${h.cls === "warn" ? "" : h.cls}"></i><span>${h.label}</span>`
  }
}

function applyState(s) {
  const prevMode = live.mode
  live.recording = !!s.recording
  live.mode = s.mode
  live.clipFrac = s.clip_frac || 0
  live.metrics = s
  if (s.lock_target) live.freqHz = s.lock_target
  if (s.tuned_hz) current = current || s.tuned_hz
  const b = $("b-rec")
  if (b) {
    b.classList.toggle("on", live.recording)
    const recLabel = live.recording ? `Стоп запис ${s.rec_seconds | 0}с` : "Запис"
    b.title = recLabel
    b.setAttribute("aria-label", recLabel)
  }
  const biasInp = document.querySelector(".bias-toggle input")
  if (biasInp) biasInp.checked = !!s.bias_tee
  if (s.bias_tee != null) values["sdr.bias_tee"] = !!s.bias_tee
  const srcEl = $("s-src")
  if (srcEl) srcEl.textContent = s.source
  const m = $("s-mode")
  if (m) {
    m.textContent = s.mode
    m.className = "mode-" + s.mode
  }
  put("s-tuned", fmt(s.tuned_hz))
  put("s-sweeps", s.sweeps_done)
  ingestAfcLimit(s)
  if (live.afcHz != null) put("v-afc", fmtAfc(live.afcHz))
  if (s.last_frame_ref) live.lastShot = s.last_frame_ref
  ingestGrid(s)
  applySpectrumPayload(s)
  setWatching(s.mode === "LOCK")
  dirty.hud = true
  updateMediaPath()
  const cf = (s.clip_frac || 0) * 100
  const cl = $("s-clip")
  if (cl) {
    cl.textContent = cf.toFixed(2) + "%"
    cl.className = cf > 0.1 ? "bad" : ""
  }
  if (s.fps != null) metric("v-fps", Number(s.fps).toFixed(1), " /с")
  const barMode = $("bar-mode")
  if (barMode) {
    const label = s.mode === "LOCK" ? "утримання" : s.mode === "INSPECT" ? "перевірка" : "свіп"
    barMode.innerHTML = `<i class="dot ${s.mode === "LOCK" ? "ok" : "info"}"></i><span>режим: ${label}</span>`
  }
  const barF = $("bar-freq")
  if (barF) barF.textContent = fmt(s.lock_target || s.tuned_hz)
  if (Array.isArray(s.detections)) {
    hits.clear()
    for (const d of s.detections) hits.set(Math.round(d.freq_hz / 1e6), d)
    renderHits()
  }
  updateHealth()
  if (s.mode && s.mode !== prevMode) syncParamToolset()
}

async function lock(f) {
  current = f
  live.freqHz = f
  live.mode = "LOCK"
  await engineLock(f)
  put("v-f", fmt(f))
  const mf = $("manual-freq")
  if (mf) mf.value = (f / 1e6).toFixed(1)
  setWatching(true)
  renderHits()
  dirty.grid = true
  dirty.spec = true
  syncParamToolset()
  refreshLiveSpectrum()
}

function clusterPayload(raw) {
  if (raw === "off" || raw === "" || raw == null) return "off"
  const n = Number(raw)
  return Number.isFinite(n) ? n : "off"
}

function syncScanMenus() {
  const cluster = $("sel-cluster")
  if (cluster) {
    const v = values["scan.cluster_step_mhz"]
    if (v === "off" || v === 0 || v === "0") cluster.value = "off"
    else if (v == 12 || v == 8 || v == 4) cluster.value = String(Number(v))
    else cluster.value = "8"
  }
  const filt = $("sel-hit-filter")
  if (filt) {
    const v = values["scan.hit_filter"]
    const allowed = new Set(["all", "hide_weak", "hide_no_video", "hide_near_dup"])
    filt.value = allowed.has(v) ? v : "hide_weak"
  }
}

async function commitScanMenu(key, value) {
  const r = await client.setParameters({ [key]: value })
  values = r.values
  setApiStatus(r.remote)
  syncScanMenus()
  if (r.error) { note(r.error, true); return }
  if (key === "scan.cluster_step_mhz") {
    if (live.mode === "SWEEP") {
      try {
        const res = await engineSweep()
        if (res && res.ok === false) throw new Error("sweep")
        setApplyStatus("застосовано зараз", [key])
        dirty.grid = true
        return
      } catch {
        setApplyStatus("після свіпу", [key])
        return
      }
    }
    setApplyStatus("після свіпу", [key])
    return
  }
  setApplyStatus("застосовано зараз", [key])
  if (key === "scan.hit_filter") {
    engineState().then(s => { if (s) applyState(s) }).catch(() => {})
  }
}

on("sel-cluster", "change", () => {
  const el = $("sel-cluster")
  if (!el) return
  commitScanMenu("scan.cluster_step_mhz", clusterPayload(el.value))
})
on("sel-hit-filter", "change", () => {
  const el = $("sel-hit-filter")
  if (!el) return
  commitScanMenu("scan.hit_filter", el.value)
})

function hideVideo() {
  latestFrame = null
  dirty.frame = false
  revokePics()
  const img = $("pic")
  if (img) {
    img.removeAttribute("src")
    img.style.display = "none"
  }
  const hint = $("hint")
  if (hint) hint.hidden = false
}

function startSweep() {
  current = null
  live.freqHz = null
  live.locked = false
  live.mode = "SWEEP"
  hideVideo()
  setWatching(false)
  renderHits()
  dirty.grid = true
  syncParamToolset()
  return engineSweep()
}

bindClick("b-sweep", () => {
  startSweep().catch(err => note(String(err && err.message || err), true))
})
bindClick("b-clear", async () => {
  await engineClear()
  hits.clear()
  grid.progress_01 = 0
  grid.current_hz = null
  grid.visiting_hz = []
  renderHits()
  dirty.grid = true
})
bindClick("b-shot", () => engineSnapshot())
bindClick("b-rec", async () => { await engineRecord(!live.recording) })
{
  const el = $("b-bias")
  if (el) {
    el.onclick = async () => {
      const inp = document.querySelector(".bias-toggle input")
      if (inp) inp.click()
    }
  }
}

function nudge(deltaHz) {
  const base = displayedFreqHz()
  if (!base) return
  lock(base + deltaHz)
}
function stepLock(mhz) {
  nudge(Number(mhz) * 1e6)
}
function parseMhz(raw) {
  const v = parseFloat(String(raw || "").replace(",", ".").replace(/\s/g, ""))
  return Number.isFinite(v) && v > 0 ? v : NaN
}

function closeFreqPop() {
  const pop = $("freq-pop")
  const btn = $("b-fft-freq")
  if (pop) pop.hidden = true
  if (btn) btn.setAttribute("aria-expanded", "false")
}

function openFreqPop() {
  const pop = $("freq-pop")
  const btn = $("b-fft-freq")
  const inp = $("manual-freq")
  if (!pop) return
  pop.hidden = false
  if (btn) btn.setAttribute("aria-expanded", "true")
  const hz = displayedFreqHz()
  if (inp) {
    inp.value = hz ? (hz / 1e6).toFixed(1).replace(".", ",") : ""
    inp.focus()
    inp.select()
  }
}

function commitManualFreq() {
  const inp = $("manual-freq")
  if (!inp) return
  const v = parseMhz(inp.value)
  if (isNaN(v)) return
  closeFreqPop()
  lock(v * 1e6)
}

document.querySelectorAll("#fft-win [data-mhz]").forEach(btn => {
  btn.addEventListener("click", () => stepLock(btn.dataset.mhz))
})
on("b-fft-freq", "click", e => {
  e.stopPropagation()
  const pop = $("freq-pop")
  if (pop && !pop.hidden) closeFreqPop()
  else openFreqPop()
})
on("b-manual-lock", "click", () => commitManualFreq())
on("manual-freq", "keydown", e => {
  if (e.key === "Enter") {
    e.preventDefault()
    commitManualFreq()
  }
  if (e.key === "Escape") {
    e.preventDefault()
    closeFreqPop()
  }
})
document.addEventListener("mousedown", e => {
  const pop = $("freq-pop")
  if (!pop || pop.hidden) return
  const btn = $("b-fft-freq")
  if (pop.contains(e.target) || (btn && btn.contains(e.target))) return
  closeFreqPop()
})

setInterval(async () => {
  if (Date.now() - live.lastWs < 4000) return
  try {
    const s = await engineState()
    if (s.error) { note("сервер: " + s.error, true); return }
    try { applyState(s) } catch (err) { console.error("стан:", err) }
    setWs(false, "опитування")
  } catch {
    setWs(false, "нема")
  }
}, 1500)

setInterval(() => {
  if (live.mode !== "LOCK") return
  if (Date.now() - live.lastFrameAt < 6000) return
  if (sock) { try { sock.close() } catch { /* ignore */ } }
  dirty.hud = true
}, 2000)

function applyLiveExtra(extra, opts = {}) {
  if (!extra || typeof extra !== "object") return
  const prevMode = live.mode
  if (extra.mode) live.mode = extra.mode
  if (extra.frequency_hz != null) live.freqHz = extra.frequency_hz
  if (extra.lock_target_hz != null) live.freqHz = extra.lock_target_hz
  if (extra.lock_state != null) live.locked = extra.lock_state
  if (extra.video_metrics) {
    const vm = extra.video_metrics
    if (vm.locked != null) live.locked = vm.locked
    if (vm.fps != null && Date.now() - live.lastFrameAt > 1500) live.fps = vm.fps
    if (vm.standard) live.standard = vm.standard
    if (vm.line_rate != null) live.lineRate = vm.line_rate
    if (vm.lines != null) live.lines = vm.lines
    if (vm.clip_frac != null) live.clipFrac = vm.clip_frac
  }
  ingestAfcLimit(extra)
  if (opts.mergeParams && extra.parameters && typeof extra.parameters === "object") {
    Object.assign(values, extra.parameters)
    syncScanMenus()
  }
  live.metrics = { ...live.metrics, ...extra }
  if (extra.last_frame_ref) live.lastShot = extra.last_frame_ref
  ingestGrid(extra)
  applySpectrumPayload(extra)
  setWatching(live.mode === "LOCK")
  dirty.hud = true
  if (live.mode !== prevMode) syncParamToolset()
}

async function refreshLiveSpectrum() {
  const extra = await client.live()
  applyLiveExtra(extra, { mergeParams: true })
}

setInterval(async () => {
  if (Date.now() - live.lastWs < 4000) return
  const extra = await client.live()
  if (!extra) return
  applyLiveExtra(extra)
}, 2500)

/* ---------------- testing panel ---------------- */

function setTab(id) {
  if (id === "observe" || id === "test" || id === "schema") id = "params"
  panelTab = "params"
  for (const btn of document.querySelectorAll(".tabs [role=tab]")) {
    btn.setAttribute("aria-selected", btn.dataset.tab === id ? "true" : "false")
  }
  for (const pane of document.querySelectorAll(".tp-pane")) {
    pane.hidden = pane.id !== "pane-params"
  }
  if (location.hash && location.hash.startsWith("#")) {
    if (location.hash !== "#test") history.replaceState(null, "", "#test")
  }
}

function loadUi() {
  try { return JSON.parse(localStorage.getItem(UI_LS) || "{}") } catch { return {} }
}
function saveUi(patch) {
  const next = { ...loadUi(), ...patch }
  try { localStorage.setItem(UI_LS, JSON.stringify(next)) } catch { /* quota */ }
  return next
}

function applyCollapsed(key, collapsed, btn) {
  const cls = ({ rail: "rail-collapsed", status: "status-collapsed", hits: "hits-collapsed", fft: "fft-collapsed", panel: "panel-collapsed" })[key]
  document.body.classList.toggle(cls, !!collapsed)
  if (btn) btn.setAttribute("aria-expanded", collapsed ? "false" : "true")
  if (key === "rail") {
    const rail = $("b-rail")
    if (rail) {
      const label = collapsed ? "Розгорнути скан" : "Згорнути скан"
      rail.title = label
      rail.setAttribute("aria-label", label)
    }
  }
  if (key === "panel") {
    const hideBtn = $("b-panel-close")
    if (hideBtn) {
      hideBtn.title = "Приховати панель"
      hideBtn.setAttribute("aria-label", "Приховати панель")
      hideBtn.setAttribute("aria-expanded", collapsed ? "false" : "true")
    }
    const openBtn = $("b-panel-open")
    if (openBtn) {
      openBtn.title = "Показати панель"
      openBtn.setAttribute("aria-label", "Показати панель")
      openBtn.setAttribute("aria-expanded", collapsed ? "false" : "true")
    }
  }
}

function toggleUi(key, force) {
  const cls = ({ rail: "rail-collapsed", status: "status-collapsed", hits: "hits-collapsed", fft: "fft-collapsed", panel: "panel-collapsed" })[key]
  const hide = force == null ? !document.body.classList.contains(cls) : !force
  const btn = ({ rail: $("b-rail"), status: $("b-status"), hits: $("b-hits"), fft: $("b-fft"), panel: $("b-panel-close") })[key]
  applyCollapsed(key, hide, btn)
  saveUi({ [key]: hide })
}

function restoreUi() {
  const u = loadUi()
  autoRelock = u.autoRelock !== false
  applyCollapsed("rail", !!u.rail, $("b-rail"))
  applyCollapsed("status", !!u.status, $("b-status"))
  applyCollapsed("hits", !!u.hits, $("b-hits"))
  document.body.classList.remove("fft-collapsed")
  applyCollapsed("panel", false, $("b-panel-close"))
}

function togglePanel(force) {
  toggleUi("panel", force)
}

function specByKey(key) {
  return catalog.find(p => p.key === key)
}

function effectLabel(spec) {
  if (!spec) return ""
  if (spec.affects) return String(spec.affects)
  const why = pendingWhy[spec.key]
  if (why) return why
  if (LOCK_RETUNE.has(spec.key) || (spec.group === "lock" && (spec.key || "").startsWith("video."))) {
    return "картинка · після lock"
  }
  if (spec.group === "sweep" || (spec.key || "").startsWith("scan.")) return "сітка свіпу"
  return "одразу"
}

function markPendingRows(keys, reasons) {
  pendingNow = new Set(keys || [])
  pendingWhy = reasons || {}
  for (const row of document.querySelectorAll(".param[data-key]")) {
    const key = row.dataset.key
    row.classList.toggle("pending", pendingNow.has(key))
    const tag = row.querySelector(".param-effect")
    if (tag) {
      const spec = specByKey(key)
      tag.textContent = pendingWhy[key] || (spec ? effectLabel(spec) : "")
    }
  }
  const hint = $("lock-group-hint")
  if (hint) {
    const waiting = [...pendingNow].some(k => LOCK_RETUNE.has(k) || (k || "").startsWith("video."))
    hint.hidden = !waiting
  }
}

function setApplyStatus(kind, keys) {
  const el = $("param-apply")
  if (!el) return
  const list = (keys || []).filter(Boolean)
  const shown = list.slice(0, 4).join(", ") + (list.length > 4 ? ` · ще ${list.length - 4}` : "")
  el.textContent = list.length ? `${kind}: ${shown}` : kind
  el.classList.toggle("waiting", kind === "після переналаштування" || kind === "після свіпу")
}

async function afterParamsApplied(r, patchKeys) {
  const pending = (r && r.pending_keys) || []
  const reasons = (r && r.pending_reasons) || {}
  const applied = (r && r.applied_keys) || patchKeys || []
  markPendingRows(pending, reasons)
  refreshDirty()
  const freq = displayedFreqHz()
  const inLock = live.mode === "LOCK"
  const canRelock = autoRelock && inLock && freq && pending.length
  if (canRelock) {
    try {
      const res = await engineLock(freq)
      if (res && res.ok === false) throw new Error("lock")
      markPendingRows([], {})
      setApplyStatus("застосовано зараз", pending)
      dirty.spec = true
      dirty.grid = true
      dirty.hud = true
      return
    } catch {
      setApplyStatus("після переналаштування", pending)
      return
    }
  }
  if (pending.length) setApplyStatus("після переналаштування", pending)
  else setApplyStatus("застосовано зараз", applied)
  syncScanMenus()
}

function currentToolset() {
  return live.mode === "LOCK" ? "lock" : "sweep"
}

function specTask(spec) {
  if (!spec) return ""
  if (spec.task) return String(spec.task)
  if (Array.isArray(spec.tasks) && spec.tasks.length) return String(spec.tasks[0])
  return ""
}

function specsForSection(section, toolset) {
  const hide = toolset === "lock" ? LOCK_HIDE : null
  const tags = new Set(section.tasks || [section.id])
  const tagged = catalog.filter(p => tags.has(specTask(p)))
  const seen = new Set()
  const out = []
  for (const p of tagged) {
    if (hide && hide.has(p.key)) continue
    if (seen.has(p.key)) continue
    seen.add(p.key)
    out.push(p)
  }
  for (const k of section.keys || []) {
    if (hide && hide.has(k)) continue
    if (seen.has(k)) continue
    if (toolsetKeys && !toolsetKeys.has(k)) continue
    const spec = specByKey(k)
    if (!spec) continue
    seen.add(k)
    out.push(spec)
  }
  return out
}

function appendGroupHead(root, section, extra) {
  const head = document.createElement("div")
  head.className = "group-head"
  head.dataset.group = section.id
  const title = document.createElement("h3")
  title.textContent = section.label
  head.appendChild(title)
  if (extra) extra(head)
  const reset = document.createElement("button")
  reset.type = "button"
  reset.textContent = "Скинути групу"
  reset.addEventListener("click", () => resetGroup(section.id))
  head.appendChild(reset)
  root.appendChild(head)
}

function pllKey() {
  if (specByKey("video.pll_enable")) return "video.pll_enable"
  return "video.h_pll"
}

function lockCompact() {
  return currentToolset() === "lock" && autoRelock
}

function appendBoolChip(parent, spec, label) {
  const key = spec.key
  const on = !!(values[key] ?? spec.default)
  const lab = document.createElement("label")
  lab.className = "toggle auto-relock"
  lab.innerHTML = `<input type="checkbox"${on ? " checked" : ""}> ${esc(label || spec.label)}`
  lab.querySelector("input").addEventListener("change", e => {
    commitParam(key, e.target.checked, true)
  })
  parent.appendChild(lab)
}

function appendGainBiasRow(root) {
  const gain = specByKey("sdr.gain_db")
  if (!gain) return
  const wrap = paramRow(gain)
  wrap.classList.add("gain-bias")
  const bias = specByKey("sdr.bias_tee") || {
    key: "sdr.bias_tee", type: "bool", label: "Bias-T", default: true,
  }
  const ctrl = wrap.querySelector(".param-ctrl")
  const lab = document.createElement("label")
  lab.className = "toggle bias-toggle"
  const on = !!(values[bias.key] ?? bias.default)
  lab.innerHTML = `<input type="checkbox"${on ? " checked" : ""}> Bias-T`
  lab.title = "Bias-T · живлення LNA"
  lab.querySelector("input").addEventListener("change", e => {
    commitParam(bias.key, e.target.checked, true)
  })
  ctrl.appendChild(lab)
  root.appendChild(wrap)
}

function appendLockCore(root) {
  const bar = document.createElement("div")
  bar.className = "group-head toolset-bar"
  const tog = document.createElement("label")
  tog.className = "auto-relock"
  tog.innerHTML = `<input type="checkbox"${autoRelock ? " checked" : ""}> авто lock`
  tog.querySelector("input").addEventListener("change", e => {
    autoRelock = e.target.checked
    saveUi({ autoRelock })
    applyToolsetEnabled()
  })
  bar.appendChild(tog)
  const pll = specByKey(pllKey()) || { key: pllKey(), type: "bool", label: "PLL", default: false }
  appendBoolChip(bar, pll, "PLL")
  root.appendChild(bar)
  appendGainBiasRow(root)
}

function appendTaskSections(root, mode) {
  for (const section of TASK_TOOLS[mode] || []) {
    if (PANEL_SKIP.has(section.id)) continue
    if (section.children) {
      appendGroupHead(root, section)
      for (const child of section.children) {
        const items = specsForSection(child, mode)
        if (!items.length) continue
        const sub = document.createElement("h4")
        sub.className = "sub-head"
        sub.textContent = child.label
        root.appendChild(sub)
        for (const spec of items) root.appendChild(paramRow(spec))
      }
      continue
    }
    const items = specsForSection(section, mode)
    if (!items.length) continue
    appendGroupHead(root, section)
    for (const spec of items) root.appendChild(paramRow(spec))
  }
}

function applyToolsetEnabled() {
  const root = $("param-groups")
  if (!root) return
  const mode = currentToolset()
  for (const fs of root.querySelectorAll("fieldset.toolset")) {
    const on = fs.dataset.toolset === mode
    fs.disabled = !on
    fs.classList.toggle("off", !on)
  }
  const extra = root.querySelector("fieldset.toolset-extra")
  if (extra) {
    const extraOn = mode === "lock" && !autoRelock
    extra.disabled = !extraOn
    extra.classList.toggle("off", !extraOn)
  }
}

function renderParams() {
  const root = $("param-groups")
  if (!root) return
  root.innerHTML = ""
  const sweepBox = document.createElement("fieldset")
  sweepBox.className = "toolset"
  sweepBox.dataset.toolset = "sweep"
  appendTaskSections(sweepBox, "sweep")
  root.appendChild(sweepBox)
  const lockBox = document.createElement("fieldset")
  lockBox.className = "toolset"
  lockBox.dataset.toolset = "lock"
  appendLockCore(lockBox)
  root.appendChild(lockBox)
  applyToolsetEnabled()
  markPendingRows([...pendingNow], pendingWhy)
  updateAfcLimitUi()
}

async function syncParamToolset(force) {
  const next = currentToolset()
  if (!force && toolsetKeys) {
    toolsetMode = next
    applyToolsetEnabled()
    return
  }
  toolsetMode = next
  const lists = await Promise.all([
    client.listParameters("sweep"),
    client.listParameters("lock"),
  ])
  const items = lists.flatMap(list => list || [])
  if (items.length) {
    toolsetKeys = new Set(items.map(p => p.key))
    const byKey = new Map(catalog.map(p => [p.key, p]))
    for (const p of items) byKey.set(p.key, p)
    catalog = [...byKey.values()]
  } else if (force) {
    toolsetKeys = null
  }
  renderParams()
}

function paramRow(spec) {
  const wrap = document.createElement("div")
  wrap.className = "param"
  wrap.dataset.key = spec.key
  const stored = values[spec.key] ?? spec.default
  const dirty = !sameValue(stored, defaults[spec.key] ?? spec.default)
  if (dirty) wrap.classList.add("dirty")
  const help = spec.key !== "sdr.gain_db" && spec.help
    ? `<p class="param-help">${esc(spec.help)}</p>`
    : ""
  wrap.innerHTML = `
    <div class="param-head">
      <label for="p-${cssId(spec.key)}">${esc(spec.label)}</label>
      <span class="param-def">типово ${esc(formatDefault({ ...spec, default: defaults[spec.key] ?? spec.default }))}</span>
      <button type="button" class="param-reset ghost" data-reset="${esc(spec.key)}">скинути</button>
    </div>
    ${help}
    <div class="param-ctrl ${spec.type}"></div>`
  wrap.querySelector("[data-reset]").addEventListener("click", () => {
    commitParam(spec.key, defaults[spec.key] ?? spec.default, true)
  })
  const ctrl = wrap.querySelector(".param-ctrl")
  if (spec.type === "bool") {
    ctrl.innerHTML = `<label class="toggle"><input type="checkbox" id="p-${cssId(spec.key)}" ${stored ? "checked" : ""}> ${stored ? "увімкнено" : "вимкнено"}</label>`
    const inp = ctrl.querySelector("input")
    inp.addEventListener("change", () => {
      ctrl.querySelector(".toggle").lastChild.textContent = " " + (inp.checked ? "увімкнено" : "вимкнено")
      commitParam(spec.key, inp.checked, true)
    })
  } else if (spec.type === "enum") {
    const opts = (spec.options || []).map(o => {
      const v = optValue(o)
      const sel = String(v) === String(stored) ? " selected" : ""
      return `<option value="${esc(String(v))}"${sel}>${esc(String(optLabel(o)))}</option>`
    }).join("")
    ctrl.innerHTML = `<select id="p-${cssId(spec.key)}">${opts}</select>`
    ctrl.querySelector("select").addEventListener("change", e => {
      const raw = e.target.value
      const asNum = Number(raw)
      const numeric = typeof spec.default === "number" || (spec.options || []).some(o => typeof optValue(o) === "number")
      commitParam(spec.key, numeric && raw !== "" && Number.isFinite(asNum) ? asNum : raw, true)
    })
  } else if (spec.type === "string") {
    const shown = Array.isArray(stored) ? formatList(stored) : (stored ?? "")
    ctrl.innerHTML = `<input id="p-${cssId(spec.key)}" type="text" value="${esc(String(shown))}" autocomplete="off">`
    const inp = ctrl.querySelector("input")
    inp.addEventListener("change", () => commitParam(spec.key, storedFromDisplay(spec, inp.value), true))
  } else {
    const shown = displayOf(spec, stored)
    const min = spec.min != null ? spec.min / (spec.factor || 1) : undefined
    const max = spec.max != null ? spec.max / (spec.factor || 1) : undefined
    const step = spec.step != null ? spec.step / (spec.factor || 1) : 1
    ctrl.innerHTML = `
      <input type="range" ${min != null ? `min="${min}"` : ""} ${max != null ? `max="${max}"` : ""} step="${step}" value="${shown}">
      <input id="p-${cssId(spec.key)}" type="number" ${min != null ? `min="${min}"` : ""} ${max != null ? `max="${max}"` : ""} step="${step}" value="${shown}">
      <span class="unit">${esc(displayUnit(spec))}</span>`
    const range = ctrl.querySelector("input[type=range]")
    const num = ctrl.querySelector("input[type=number]")
    const sync = (src, immediate) => {
      const other = src === range ? num : range
      other.value = src.value
      commitParam(spec.key, storedFromDisplay(spec, src.value), immediate)
    }
    range.addEventListener("input", () => sync(range, false))
    range.addEventListener("change", () => sync(range, true))
    num.addEventListener("change", () => sync(num, true))
  }
  return wrap
}

function cssId(key) {
  return key.replace(/[^a-z0-9]+/gi, "-")
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]))
}

function refreshDirty() {
  for (const row of document.querySelectorAll(".param[data-key]")) {
    const spec = specByKey(row.dataset.key)
    if (!spec) continue
    row.classList.toggle("dirty", !sameValue(values[spec.key], defaults[spec.key] ?? spec.default))
  }
}

function commitParam(key, raw, immediate) {
  const spec = specByKey(key)
  const next = spec ? clampToSpec(spec, raw) : raw
  values[key] = next
  pendingPatch[key] = next
  refreshDirty()
  const send = () => {
    const patch = pendingPatch
    pendingPatch = {}
    client.setParameters(patch).then(r => {
      values = r.values
      setApiStatus(r.remote)
      if (r.error) note(r.error, true)
      else afterParamsApplied(r, Object.keys(patch))
    })
  }
  if (immediate) {
    clearTimeout(debounceTimer)
    send()
  } else {
    clearTimeout(debounceTimer)
    debounceTimer = setTimeout(send, 280)
  }
}

function sectionById(gid) {
  for (const mode of Object.keys(TASK_TOOLS)) {
    for (const section of TASK_TOOLS[mode]) {
      if (section.id === gid) return section
      for (const child of section.children || []) {
        if (child.id === gid) return child
      }
    }
  }
  return null
}

function sectionToolset(section) {
  if (!section) return currentToolset()
  for (const [mode, sections] of Object.entries(TASK_TOOLS)) {
    for (const s of sections) {
      if (s.id === section.id) return mode
      if ((s.children || []).some(c => c.id === section.id)) return mode
    }
  }
  return currentToolset()
}

async function resetGroup(gid) {
  const section = sectionById(gid)
  const mode = sectionToolset(section)
  const items = section
    ? [...specsForSection(section, mode), ...(section.children || []).flatMap(c => specsForSection(c, mode))]
    : catalog.filter(p => p.group === gid)
  const patch = {}
  for (const spec of items) {
    if (spec) patch[spec.key] = defaults[spec.key] ?? spec.default
  }
  const r = await client.setParameters(patch)
  values = r.values
  setApiStatus(r.remote)
  renderParams()
  if (r.error) note(r.error, true)
  else afterParamsApplied(r, Object.keys(patch))
}

function setApiStatus(remote) {
  put("tp-api", remote
    ? "параметри на сервері"
    : "локальний каталог · /api/test ще не відповідає")
}

function activeSession() {
  const list = (client.db.sessions || [])
    .filter(s => s.status === "active" || s.status === "paused")
    .sort((a, b) => String(b.updated_at || b.created_at || "").localeCompare(String(a.updated_at || a.created_at || "")))
  return list[0] || null
}

function renderSessionChrome() {
  if (!$("observe-dock") && !$("quick-form") && !$("session-bar")) return
  const s = activeSession()
  if (s && !s.schema_snapshot && !s._schemaTried) {
    s._schemaTried = true
    client.getSession(s.id).then(full => {
      if (!full) return
      const i = client.db.sessions.findIndex(x => x.id === s.id)
      if (i >= 0) client.db.sessions[i] = { ...client.db.sessions[i], ...full }
      renderSessionChrome()
    })
  }
  document.body.classList.toggle("session-on", !!(s && s.status === "active"))
  const label = $("session-label")
  const hide = (id, on) => { const el = $(id); if (el) el.hidden = on }
  if (!s) {
    if (label) label.textContent = "немає активної сесії"
    hide("b-sess-start", false)
    hide("b-sess-pause", true)
    hide("b-sess-resume", true)
    hide("b-sess-done", true)
  } else {
    const st = s.status === "paused" ? "пауза" : s.status === "completed" ? "завершено" : "записує"
    if (label) label.textContent = `${s.title} · ${st}`
    hide("b-sess-start", s.status === "active" || s.status === "paused")
    hide("b-sess-pause", s.status !== "active")
    hide("b-sess-resume", s.status !== "paused")
    hide("b-sess-done", s.status === "completed")
  }
  renderObsForm($("quick-form"), true)
  renderTimeline()
}

function schemaFields() {
  return (client.db.schema && client.db.schema.fields) || []
}

function renderObsForm(root, compact) {
  if (!root) return
  const s = activeSession()
  const allowed = s ? sessionFieldKeys(s) : null
  const fields = schemaFields().filter(f => !allowed || allowed.has(fieldKey(f)))
  if (!s || (compact && s.status !== "active")) {
    root.innerHTML = compact ? "" : "<p class=\"empty\">Почніть сесію, щоб збирати спостереження під відео.</p>"
    return
  }
  const symptoms = fields.filter(f => SYMPTOM_KEYS.has(fieldKey(f)))
  const rest = fields.filter(f => !SYMPTOM_KEYS.has(fieldKey(f)))
  if (compact) {
    autoFill.frequency_mhz.dirty = false
    autoFill.bandwidth_note.dirty = false
    root.innerHTML =
      `<div class="ql-grid">${rest.map(f => fieldHtml(f, true)).join("")}</div>` +
      `<div class="ql-symptoms">${symptoms.map(f => fieldHtml(f, true)).join("")}</div>` +
      `<button type="submit" class="primary" id="ql-submit">Записати</button>`
    bindAutoFillGuards(root)
    applyLockAutofill()
    return
  }
  root.innerHTML = fields.map(f => fieldHtml(f, false)).join("") +
    `<button type="submit" class="primary">${editingObsId ? "Зберегти" : "Записати спостереження"}</button>`
}

function fieldKey(f) {
  return f.key || f.id
}

function fieldHtml(f, compact) {
  const id = (compact ? "q-" : "o-") + fieldKey(f)
  const req = f.required ? " required" : ""
  const name = esc(fieldKey(f))
  const key = fieldKey(f)
  if (compact && f.type === "bool") {
    return `<label class="ql-check"><input type="checkbox" id="${id}" name="${name}"> ${esc(f.label)}</label>`
  }
  let ctrl = ""
  if (f.type === "bool") {
    ctrl = `<label class="toggle"><input type="checkbox" id="${id}" name="${name}"> так</label>`
  } else if (f.type === "enum") {
    const opts = (f.enum_values || f.options || []).map(o => `<option value="${esc(String(o))}">${esc(String(o))}</option>`).join("")
    ctrl = `<select id="${id}" name="${name}"${req}><option value="">—</option>${opts}</select>`
  } else if (f.type === "number" || f.type === "rating" || f.type === "scale") {
    const min = f.type === "rating" || f.type === "scale" ? (f.min ?? 1) : (f.min ?? "")
    const max = f.type === "rating" || f.type === "scale" ? (f.max ?? 5) : (f.max ?? "")
    const step = key === "frequency_mhz" ? 0.001 : (f.step || 1)
    ctrl = `<input id="${id}" name="${name}" type="number" min="${min}" max="${max}" step="${step}"${req}>`
  } else {
    ctrl = compact
      ? `<input id="${id}" name="${name}" type="text" autocomplete="off"${req}>`
      : `<textarea id="${id}" name="${name}" rows="2"${req}></textarea>`
  }
  return `<div class="${compact ? "ql-field" : "field"}"><label for="${id}">${esc(f.label)}${f.required ? " *" : ""}</label>${ctrl}</div>`
}

function readForm(root, compact) {
  const out = {}
  if (!root) return out
  for (const f of schemaFields()) {
    const el = root.querySelector(`#${CSS.escape((compact ? "q-" : "o-") + fieldKey(f))}`)
    if (!el) continue
    const k = fieldKey(f)
    if (f.type === "bool") out[k] = el.checked
    else if (f.type === "number" || f.type === "rating" || f.type === "scale") {
      out[k] = el.value === "" ? null : Number(el.value)
    } else if (f.type === "tag_list") {
      out[k] = String(el.value || "").split(",").map(s => s.trim()).filter(Boolean)
    } else out[k] = el.value
  }
  return out
}

function compactFields(data) {
  const out = {}
  for (const [k, v] of Object.entries(data)) {
    if (v === "" || v == null) continue
    if (typeof v === "number" && !Number.isFinite(v)) continue
    if (Array.isArray(v) && !v.length) continue
    out[k] = v
  }
  return out
}

function sessionFieldKeys(session) {
  const snap = session && (session.schema_snapshot || (session.schema && session.schema.fields))
  if (Array.isArray(snap) && snap.length) {
    return new Set(snap.map(f => f.key || f.id).filter(Boolean))
  }
  return new Set(schemaFields().map(fieldKey).filter(k => !SYMPTOM_KEYS.has(k)))
}

function fieldsForSession(data, session) {
  const allowed = sessionFieldKeys(session)
  const out = compactFields(data)
  for (const k of Object.keys(out)) {
    if (!allowed.has(k)) delete out[k]
  }
  if (out.frequency_mhz === "" || out.frequency_mhz == null) delete out.frequency_mhz
  return out
}

function bindAutoFillGuards(root) {
  for (const key of Object.keys(autoFill)) {
    const el = root.querySelector("#q-" + key)
    if (!el) continue
    el.addEventListener("input", () => { autoFill[key].dirty = true })
  }
}

function applyLockAutofill() {
  if (live.mode !== "LOCK") return
  const form = $("quick-form")
  if (!form) return
  const hz = live.freqHz || current || live.metrics.lock_target || live.metrics.frequency_hz
  if (hz && !autoFill.frequency_mhz.dirty) {
    const mhz = Number((Number(hz) / 1e6).toFixed(3))
    const el = form.querySelector("#q-frequency_mhz")
    if (el && autoFill.frequency_mhz.last !== mhz) {
      el.value = String(mhz)
      autoFill.frequency_mhz.last = mhz
    }
  }
  if (!autoFill.bandwidth_note.dirty) {
    const bw = specView.bw_hz || Number(values["video.channel_bw_hz"]) || null
    if (bw && Number.isFinite(bw)) {
      const text = bw >= 1e6 ? `${Math.round(bw / 1e6)} MHz` : `${Math.round(bw / 1e3)} kHz`
      const el = form.querySelector("#q-bandwidth_note")
      if (el && autoFill.bandwidth_note.last !== text) {
        el.value = text
        autoFill.bandwidth_note.last = text
      }
    }
  }
}

function fillForm(root, data, compact) {
  if (!root || !data) return
  for (const f of schemaFields()) {
    const el = root.querySelector(`#${CSS.escape((compact ? "q-" : "o-") + fieldKey(f))}`)
    if (!el || data[fieldKey(f)] == null) continue
    const v = data[fieldKey(f)]
    if (f.type === "bool") el.checked = !!v
    else if (Array.isArray(v)) el.value = v.join(", ")
    else el.value = v
  }
}

function validateFields(data) {
  for (const f of schemaFields()) {
    if (!f.required) continue
    const v = data[fieldKey(f)]
    if (v == null || v === "") return `заповніть поле «${f.label}»`
    if (Array.isArray(v) && !v.length) return `заповніть поле «${f.label}»`
  }
  return null
}

function currentFrameRef() {
  return live.lastShot
    || live.lastRec
    || (live.metrics && live.metrics.last_frame_ref)
    || null
}

function captureSnapshot() {
  const hit = current && [...hits.values()].find(d => Math.abs(d.freq_hz - current) < 2e6)
  const vm = live.metrics && live.metrics.video_metrics
  return {
    timestamp_ms: Date.now(),
    frequency_hz: live.freqHz || current || live.metrics.tuned_hz || live.metrics.frequency_hz || null,
    rssi: live.metrics.rssi ?? null,
    snr: hit ? hit.snr_db : (live.metrics.snr ?? null),
    frame_ref: currentFrameRef(),
    video_metrics: {
      locked: live.locked,
      fps: live.fps || live.metrics.fps || (vm && vm.fps),
      standard: live.standard || (vm && vm.standard),
      line_rate: live.lineRate ?? (vm && vm.line_rate),
      lines: live.lines ?? (vm && vm.lines),
      afc_hz: live.afcHz ?? (vm && vm.afc_hz),
      freq_err_hz: live.freqErrHz ?? (vm && vm.freq_err_hz),
      afc_pegged: live.afcPegged,
      clip_frac: live.clipFrac,
      decode: decodeHealth().text,
    },
    parameter_snapshot: { ...values },
  }
}

async function submitObservation(fromQuick) {
  const s = activeSession()
  if (!s || s.status !== "active") {
    note("немає активної сесії аналізу", true)
    return
  }
  const root = $("quick-form")
  if (!root) return
  const fields = fieldsForSession(readForm(root, true), s)
  const err = validateFields(fields)
  if (err) { note(err, true); return }
  const snap = captureSnapshot()
  const payload = {
    timestamp_ms: snap.timestamp_ms,
    frequency_hz: snap.frequency_hz,
    rssi: snap.rssi,
    snr: snap.snr,
    video_metrics: snap.video_metrics,
    parameter_snapshot: snap.parameter_snapshot,
    fields,
    frame_ref: snap.frame_ref || currentFrameRef(),
  }
  if (editingObsId) {
    await client.patchObservation(s.id, editingObsId, payload)
    editingObsId = null
  } else {
    const saved = await client.addObservation(s.id, payload)
    if (saved && saved.error) note(saved.error, true)
  }
  renderSessionChrome()
  note("спостереження записано")
}

async function renderTimeline() {
  const s = activeSession()
  const ul = $("timeline")
  if (!ul) return
  if (!s) { ul.innerHTML = ""; return }
  const list = await client.listObservations(s.id)
  if (!list.length) {
    ul.innerHTML = "<li class=\"empty\">Поки порожньо. Space — записати з форми під відео.</li>"
    return
  }
  ul.innerHTML = [...list].reverse().map(o => {
    const when = formatWhen(o.created_at || o.timestamp || o.timestamp_ms)
    const freq = fmt(o.frequency_hz ?? o.freq_hz)
    const bits = Object.entries(o.fields || {}).filter(([, v]) => v !== "" && v != null)
      .map(([k, v]) => `${k}: ${v === true ? "так" : v === false ? "ні" : v}`).join(" · ")
    return `<li data-id="${esc(o.id)}">
      <div class="when">${esc(when)} · <span class="freq">${esc(freq)}</span></div>
      <div class="sum">${esc(bits || "без полів")}</div>
      <div class="row-acts">
        <button type="button" data-edit="${esc(o.id)}">Редагувати</button>
        <button type="button" class="danger" data-del="${esc(o.id)}">Видалити</button>
      </div>
    </li>`
  }).join("")
  ul.querySelectorAll("[data-edit]").forEach(b => b.onclick = () => editObs(s.id, b.dataset.edit, list))
  ul.querySelectorAll("[data-del]").forEach(b => b.onclick = () => delObs(s.id, b.dataset.del))
}

function formatWhen(iso) {
  if (iso == null || iso === "") return "—"
  if (typeof iso === "number") {
    const d = new Date(iso > 1e12 ? iso : iso * 1000)
    return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
  }
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return String(iso)
  return d.toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
}

function editObs(sessionId, id, list) {
  const o = list.find(x => x.id === id)
  if (!o) return
  editingObsId = id
  renderObsForm($("quick-form"), true)
  fillForm($("quick-form"), o.fields || {}, true)
  note("редагування — Space або Записати")
}

async function delObs(sessionId, id) {
  if (!confirm("Видалити спостереження?")) return
  await client.deleteObservation(sessionId, id)
  if (editingObsId === id) editingObsId = null
  renderSessionChrome()
}

function renderSchemaEditor() {
  const root = $("schema-list")
  if (!root) return
  const fields = schemaFields()
  root.innerHTML = fields.map((f, i) => `
    <div class="schema-row" data-i="${i}">
      <input type="text" data-k="key" value="${esc(fieldKey(f))}" aria-label="ключ" pattern="[A-Za-z][A-Za-z0-9_]*">
      <input type="text" data-k="label" value="${esc(f.label)}" aria-label="назва">
      <select data-k="type" aria-label="тип">
        ${["rating", "bool", "number", "text", "enum", "tag_list"].map(t =>
          `<option value="${t}"${f.type === t ? " selected" : ""}>${t}</option>`).join("")}
      </select>
      <label class="toggle" title="обов'язкове"><input type="checkbox" data-k="required"${f.required ? " checked" : ""}></label>
      <button type="button" class="ghost" data-rm="${i}" aria-label="прибрати">×</button>
    </div>
    <div class="schema-row opts" data-i="${i}" ${f.type === "enum" ? "" : "hidden"}>
      <input type="text" data-k="options" value="${esc((f.enum_values || f.options || []).join(", "))}" placeholder="варіанти через кому">
    </div>`).join("")
  root.querySelectorAll("[data-rm]").forEach(b => b.onclick = () => {
    const next = schemaFields().filter((_, i) => i !== Number(b.dataset.rm))
    client.db.schema.fields = next
    renderSchemaEditor()
  })
  root.querySelectorAll("[data-k=type]").forEach(sel => {
    sel.onchange = () => {
      const row = sel.closest(".schema-row")
      const opts = root.querySelector(`.schema-row.opts[data-i="${row.dataset.i}"]`)
      if (opts) opts.hidden = sel.value !== "enum"
    }
  })
}

function readSchemaEditor() {
  const root = $("schema-list")
  if (!root) return []
  const rows = [...root.querySelectorAll(".schema-row[data-i]:not(.opts)")]
  return rows.map(row => {
    const i = row.dataset.i
    const get = k => row.querySelector(`[data-k="${k}"]`)
    const type = get("type").value
    const optsRow = root.querySelector(`.schema-row.opts[data-i="${i}"]`)
    const field = {
      key: (get("key").value.trim() || ("field_" + i)).replace(/[^A-Za-z0-9_]/g, "_"),
      label: get("label").value.trim() || get("key").value.trim(),
      type,
      required: get("required").checked,
    }
    if (type === "enum") {
      const raw = optsRow.querySelector("[data-k=options]").value
      field.enum_values = raw.split(",").map(s => s.trim()).filter(Boolean)
    }
    if (type === "rating") { field.min = 1; field.max = 5 }
    return field
  }).filter(f => f.key)
}

on("quick-form", "submit", e => {
  e.preventDefault()
  submitObservation(true)
})

on("b-sess-start", "click", async () => {
  const titleEl = $("sess-title")
  const modeEl = $("sess-mode")
  if (!titleEl || !modeEl) return
  const title = titleEl.value.trim()
  const mode = modeEl.value
  await client.createSession({ title, mode })
  renderSessionChrome()
})
on("b-sess-pause", "click", async () => {
  const s = activeSession(); if (!s) return
  await client.patchSession(s.id, { status: "paused" })
  renderSessionChrome()
})
on("b-sess-resume", "click", async () => {
  const s = activeSession(); if (!s) return
  await client.patchSession(s.id, { status: "active" })
  renderSessionChrome()
})
on("b-sess-done", "click", async () => {
  const s = activeSession(); if (!s) return
  await client.completeSession(s.id)
  renderSessionChrome()
})

bindClick("b-schema-add", () => {
  const fields = schemaFields()
  fields.push({ key: "field_" + (fields.length + 1), label: "Нове поле", type: "text", required: false })
  client.db.schema.fields = fields
  renderSchemaEditor()
})
bindClick("b-schema-save", async () => {
  const fields = readSchemaEditor()
  const r = await client.setSchema(fields)
  setApiStatus(r.remote || client.remote)
  renderSessionChrome()
  if (r.error) note(r.error, true)
  else note("схему збережено")
})
bindClick("b-hide", () => toggleUi("rail"))
bindClick("b-panel-close", () => togglePanel(false))
bindClick("b-panel-open", () => togglePanel(true))
bindClick("b-rail", () => toggleUi("rail"))
bindClick("b-status", () => toggleUi("status"))
bindClick("b-hits", () => toggleUi("hits"))

function isTyping(el) {
  if (!el) return false
  const tag = el.tagName
  if (tag === "TEXTAREA" || tag === "SELECT") return true
  if (tag === "INPUT") {
    const t = el.type
    return t !== "checkbox" && t !== "range" && t !== "button" && t !== "submit"
  }
  return el.isContentEditable
}

document.addEventListener("keydown", e => {
  if (e.key === "F2") {
    e.preventDefault()
    togglePanel(document.body.classList.contains("panel-collapsed"))
    return
  }
  if (isTyping(e.target)) return
  if (e.key === "[" || e.key === "ArrowLeft") { e.preventDefault(); nudge(e.shiftKey ? -1e6 : -0.1e6) }
  if (e.key === "]" || e.key === "ArrowRight") { e.preventDefault(); nudge(e.shiftKey ? 1e6 : 0.1e6) }
  if (e.key === "1") { e.preventDefault(); toggleUi("status") }
  if (e.key === "2") { e.preventDefault(); toggleUi("rail") }
  if (e.key === "s" || e.key === "S") {
    e.preventDefault()
    const sweep = $("b-sweep")
    if (sweep) sweep.click()
    else startSweep().catch(err => note(String(err && err.message || err), true))
  }
  if (e.key === "-" || e.key === "_") tweak("scan.threshold_db", -0.5)
  if (e.key === "=" || e.key === "+") tweak("scan.threshold_db", 0.5)
  if (e.key === "," ) tweak("sdr.gain_db", -1)
  if (e.key === ".") tweak("sdr.gain_db", 1)
})

function tweak(key, delta) {
  const spec = specByKey(key)
  if (!spec) return
  const next = clampToSpec(spec, (Number(values[key]) || 0) + delta * (spec.factor || 1))
  values[key] = next
  commitParam(key, next, true)
  const row = document.querySelector(`.param[data-key="${key}"]`)
  if (row) {
    const num = row.querySelector("input[type=number], input[type=range]")
    const shown = displayOf(spec, next)
    row.querySelectorAll("input[type=number], input[type=range]").forEach(inp => { inp.value = shown })
    void num
  }
  note(`${spec.label}: ${displayOf(spec, next)} ${spec.unit || ""}`)
}

function applyHash() {
  const h = (location.hash || "").slice(1)
  if (h === "observe" || h === "schema" || h === "params" || h === "test") {
    togglePanel(true)
    setTab(h === "schema" ? "schema" : "params")
  }
}

on("pic", "error", () => {
  if (latestFrame) return
  if (picPendingUrl) {
    URL.revokeObjectURL(picPendingUrl)
    picPendingUrl = null
  }
  if (picShownUrl) return
  const img = $("pic")
  if (img) {
    img.removeAttribute("src")
    img.style.display = "none"
  }
  const hint = $("hint")
  if (hint) {
    hint.hidden = false
    hint.textContent = "Кадр не відкрився. Потік декодера ще не дав зображення."
  }
})

async function ensureSymptomFields() {
  const fields = schemaFields()
  const have = new Set(fields.map(fieldKey))
  const extra = SYMPTOM_FIELDS.filter(f => !have.has(f.key))
  if (!extra.length) return
  await client.setSchema(fields.concat(extra))
  renderSchemaEditor()
  renderSessionChrome()
}

async function boot() {
  const probed = await client.probe()
  catalog = probed.catalog
  defaults = probed.defaults
  values = probed.values
  syncScanMenus()
  await client.getSchema()
  await ensureSymptomFields()
  await client.listSessions()
  setApiStatus(probed.remote)
  await syncParamToolset(true)
  renderSchemaEditor()
  renderSessionChrome()
  restoreUi()
  applyHash()
  dirty.grid = true
  dirty.spec = true
  dirty.hud = true
}

window.addEventListener("hashchange", applyHash)
boot().catch(err => console.error("завантаження:", err))
