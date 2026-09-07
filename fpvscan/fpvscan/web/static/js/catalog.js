/**
 * Каталог керованих параметрів SWEEP / LOCK / shared.
 * Дзеркало config.yaml + того, що читає engine.py.
 * GET /api/test/parameters має повернути той самий каркас;
 * цей модуль — запасне джерело правди, поки ендпоінт ще не піднято.
 *
 * Значення зберігаються в нативних одиницях конфіга (Гц, дБ, мс…).
 * factor — лише для відображення (поділити stored / factor).
 */
export const GROUPS = [
  { id: "sweep", label: "Свіп" },
  { id: "lock", label: "Утримання" },
  { id: "shared", label: "Спільні" },
]

/** Task-focused toolsets. Prefer GET ?mode= + item.task; keys are curated fallback only. */
export const TASK_TOOLS = {
  sweep: [
    {
      id: "scan_width",
      label: "Смуга сканування",
      tasks: ["scan_width"],
      keys: ["scan.start_hz", "scan.stop_hz", "scan.channel_bw_hz"],
    },
    {
      id: "scan_grid",
      label: "Сітка свіпу",
      tasks: ["scan_grid"],
      keys: ["scan.cluster_step_mhz"],
    },
    {
      id: "hit_filter",
      label: "Список знахідок",
      tasks: ["hit_filter"],
      keys: ["scan.hit_filter"],
    },
  ],
  lock: [
    {
      id: "picture_jump",
      label: "Стрибки картинки",
      tasks: ["picture_jump"],
      keys: [
        "video.afc", "video.afc_gain", "video.afc_deadband_hz",
        "video.afc_max_step_hz", "video.afc_digital_max_hz",
        "video.hunt", "video.hunt_every", "video.hunt_drop",
        "sdr.gain_db", "sdr.bias_tee", "sdr.settle_us", "video.capture_ms",
      ],
    },
    {
      id: "pll",
      label: "ФАПЧ рядка",
      tasks: ["pll"],
      keys: ["video.h_pll"],
    },
    {
      id: "phase_tear",
      label: "Фазові розриви",
      children: [
        {
          id: "phase_tear_h",
          label: "Горизонтальні",
          tasks: ["phase_tear_h"],
          keys: ["video.track_window_margin", "video.h_phase_frac"],
        },
        {
          id: "phase_tear_v",
          label: "Вертикальні",
          tasks: ["phase_tear_v"],
          keys: ["video.average", "video.motion_thresh"],
        },
      ],
    },
  ],
}

export const DEFAULT_SCHEMA = [
  { key: "signal_quality", label: "Якість сигналу", type: "rating", required: false, min: 1, max: 5 },
  { key: "picture_lock", label: "Синхро картинки", type: "bool", required: false },
  { key: "picture_jump", label: "Стрибки картинки", type: "bool", required: false },
  { key: "line_tearing", label: "Рядковість / розрив", type: "bool", required: false },
  { key: "spectrum_stutter", label: "Спектр смикається", type: "bool", required: false },
  { key: "analog_noise", label: "Аналоговий шум", type: "rating", required: false, min: 1, max: 5 },
  { key: "osd_present", label: "OSD на кадрі", type: "bool", required: false },
  { key: "frequency_mhz", label: "Частота (оператор)", type: "number", required: false, min: 100, max: 6200 },
  { key: "bandwidth_note", label: "Нотатка про смугу", type: "text", required: false },
  { key: "aircraft_type", label: "Тип борту", type: "text", required: false },
  { key: "notes", label: "Нотатка", type: "text", required: false },
  { key: "tags", label: "Теги", type: "tag_list", required: false },
]

function n(p) {
  return { type: "number", factor: 1, step: 1, ...p }
}
function i(p) {
  return { type: "integer", factor: 1, step: 1, ...p }
}
function b(p) {
  return { type: "bool", ...p }
}
function e(p) {
  return { type: "enum", ...p }
}
function s(p) {
  return { type: "string", ...p }
}

const MHZ = 1e6
const KHZ = 1e3

/** Повний набір knobs, які рушій читає під час свіпу й утримання. */
export const PARAMETER_CATALOG = [
  // ---- shared: SDR, впливає на обидва режими ----
  n({
    key: "sdr.gain_db", group: "shared", label: "Підсилення",
    unit: "дБ", min: 0, max: 60, step: 1, default: 35,
  }),
  n({
    key: "sdr.bias_tee_gain_offset_db", group: "shared", label: "Компенсація LNA",
    help: "На скільки зрізати gain, коли bias-tee увімкнено.",
    unit: "дБ", min: 0, max: 40, step: 1, default: 15,
  }),
  i({
    key: "sdr.settle_us", group: "shared", label: "Установлення ФАПЧ",
    help: "Пауза після перебудови.",
    unit: "мкс", min: 0, max: 5000, step: 50, default: 500,
  }),
  e({
    key: "sdr.rx_channel", group: "shared", label: "Канал RX",
    default: 0,
    options: [{ value: 0, label: "RX1" }, { value: 1, label: "RX2" }],
  }),
  b({
    key: "sdr.bias_tee", group: "shared", label: "Bias-T",
    help: "4.5 В на RX для активної антени. Частина рядка gain, не окрема кнопка.",
    default: true,
  }),
  b({
    key: "sdr.agc", group: "shared", label: "АРП",
    help: "Зручна для пошуку, шкідлива для декодування.",
    default: false,
  }),
  b({
    key: "sdr.quick_tune", group: "shared", label: "Швидка перебудова",
    help: "Профілі libbladeRF; звір через bench_retune.py.",
    default: false,
  }),
  i({
    key: "sdr.num_buffers", group: "shared", label: "USB-буфери",
    min: 4, max: 128, step: 4, default: 32,
  }),
  i({
    key: "sdr.buffer_size", group: "shared", label: "Розмір буфера",
    help: "Кратний 1024.",
    min: 4096, max: 131072, step: 1024, default: 32768,
  }),
  i({
    key: "sdr.num_transfers", group: "shared", label: "USB-трансфери",
    min: 4, max: 64, step: 1, default: 16,
  }),

  // ---- sweep: scan.* ----
  n({
    key: "scan.start_hz", group: "sweep", label: "Початок діапазону",
    unit: "МГц", factor: MHZ, min: 50e6, max: 6000e6, step: 1e6, default: 400e6,
  }),
  n({
    key: "scan.stop_hz", group: "sweep", label: "Кінець діапазону",
    unit: "МГц", factor: MHZ, min: 100e6, max: 6000e6, step: 1e6, default: 6000e6,
  }),
  n({
    key: "scan.sample_rate", group: "sweep", label: "Дискретизація свіпу",
    unit: "МГц", factor: MHZ, min: 5e6, max: 61.44e6, step: 0.5e6, default: 35e6,
  }),
  n({
    key: "scan.channel_bw_hz", group: "sweep", label: "Очікувана ширина каналу",
    help: "Визначає крок свіпу, якщо step_hz = 0.",
    unit: "МГц", factor: MHZ, min: 1e6, max: 40e6, step: 0.5e6, default: 10e6,
  }),
  n({
    key: "scan.step_hz", group: "sweep", label: "Крок свіпу",
    help: "0 — порахувати автоматично зі смуги і ширини каналу.",
    unit: "МГц", factor: MHZ, min: 0, max: 40e6, step: 0.5e6, default: 0,
  }),
  e({
    key: "scan.cluster_step_mhz", group: "sweep", label: "Крок кластера",
    help: "Extras навколо хіта. Вимк. — лише грубий крок.",
    default: "8",
    options: [
      { value: "off", label: "вимк. (грубо)" },
      { value: "12", label: "12 МГц" },
      { value: "8", label: "8 МГц" },
      { value: "4", label: "4 МГц" },
    ],
  }),
  e({
    key: "scan.hit_filter", group: "sweep", label: "Фільтр знахідок",
    help: "Ховає сміття в списку. Сирі піки в рушії лишаються.",
    default: "hide_weak",
    options: [
      { value: "all", label: "усі" },
      { value: "hide_weak", label: "ховати слабкі" },
      { value: "hide_no_video", label: "ховати без картинки" },
      { value: "hide_near_dup", label: "ховати сусідів" },
    ],
  }),
  e({
    key: "scan.fft_size", group: "sweep", label: "Розмір FFT",
    default: 8192,
    options: [1024, 2048, 4096, 8192, 16384, 32768],
  }),
  i({
    key: "scan.averages", group: "sweep", label: "Усереднень спектра",
    min: 1, max: 32, step: 1, default: 8,
  }),
  n({
    key: "scan.threshold_db", group: "sweep", label: "Поріг над підлогою",
    unit: "дБ", min: 0, max: 40, step: 0.5, default: 5,
  }),
  n({
    key: "scan.min_bw_hz", group: "sweep", label: "Мін. ширина зайнятості",
    unit: "МГц", factor: MHZ, min: 0.5e6, max: 40e6, step: 0.5e6, default: 5e6,
  }),
  n({
    key: "scan.max_bw_hz", group: "sweep", label: "Макс. ширина зайнятості",
    unit: "МГц", factor: MHZ, min: 2e6, max: 60e6, step: 1e6, default: 25e6,
  }),
  n({
    key: "scan.inspect_ms", group: "sweep", label: "Вікно класифікації",
    unit: "мс", min: 5, max: 200, step: 1, default: 40,
  }),
  n({
    key: "scan.inspect_bw_hz", group: "sweep", label: "Смуга класифікації",
    unit: "МГц", factor: MHZ, min: 2e6, max: 40e6, step: 0.5e6, default: 12e6,
  }),
  n({
    key: "scan.dc_notch_hz", group: "sweep", label: "Виріз гетеродина",
    unit: "кГц", factor: KHZ, min: 0, max: 2e6, step: 10e3, default: 200e3,
  }),
  i({
    key: "scan.confirm_hits", group: "sweep", label: "Підтверджень",
    min: 1, max: 8, step: 1, default: 2,
  }),
  n({
    key: "scan.line_tol_hz", group: "sweep", label: "Допуск рядкової",
    unit: "Гц", min: 10, max: 500, step: 5, default: 150,
  }),
  n({
    key: "scan.line_prominence_db", group: "sweep", label: "Висота рядкової лінії",
    unit: "дБ", min: 1, max: 40, step: 0.5, default: 10,
  }),
  n({
    key: "scan.min_confidence", group: "sweep", label: "Мін. впевненість",
    min: 0, max: 1, step: 0.05, default: 0.45,
  }),
  i({
    key: "scan.min_harmonics", group: "sweep", label: "Мін. гармонік рядка",
    min: 0, max: 8, step: 1, default: 1,
  }),
  b({
    key: "scan.inspect_decode", group: "sweep", label: "Димовий decode",
    help: "Кадр на зсуві або спектральний PAL/NTSC.",
    default: true,
  }),
  n({
    key: "scan.inspect_min_row_corr", group: "sweep", label: "Мін. кореляція рядків",
    min: 0, max: 1, step: 0.01, default: 0.12,
  }),
  b({
    key: "scan.inspect_require_lock", group: "sweep", label: "Вимагати кадрову на inspect",
    help: "40 мс часто не ловить кадрову; фільтр — кореляція.",
    default: false,
  }),
  i({
    key: "scan.inspect_min_lines", group: "sweep", label: "Мін. рядків димового кадру",
    min: 16, max: 625, step: 1, default: 80,
  }),
  s({
    key: "scan.inspect_offsets_mhz", group: "sweep", label: "Зсуви inspect",
    help: "МГц через кому. Центр зайнятості ≠ центр відео.",
    default: [1.0, 2.0],
  }),
  n({
    key: "scan.inspect_conf_bypass", group: "sweep", label: "Обхід decode за впевненістю",
    min: 0, max: 1, step: 0.05, default: 0.7,
  }),
  n({
    key: "scan.merge_smear_hz", group: "sweep", label: "Злиття ЧМ-плями",
    unit: "МГц", factor: MHZ, min: 1e6, max: 80e6, step: 1e6, default: 32e6,
  }),
  b({
    key: "scan.priority_bands", group: "sweep", label: "Пріоритетні діапазони",
    help: "Частіше обходити 433/900/1G2/2G4/3G3/5G8.",
    default: false,
  }),
  b({
    key: "scan.debug_candidates", group: "sweep", label: "Дебаг кандидатів",
    default: false,
  }),
  b({
    key: "scan.auto_peek", group: "sweep", label: "Автоперегляд знахідки",
    default: true,
  }),
  n({
    key: "scan.auto_peek_secs", group: "sweep", label: "Тривалість автоперегляду",
    unit: "с", min: 1, max: 30, step: 1, default: 5,
  }),
  n({
    key: "scan.auto_peek_min_conf", group: "sweep", label: "Мін. впевненість peek",
    min: 0, max: 1, step: 0.05, default: 0.6,
  }),
  n({
    key: "scan.auto_peek_cooldown_s", group: "sweep", label: "Пауза повторного peek",
    unit: "с", min: 0, max: 600, step: 5, default: 60,
  }),

  // ---- lock: video.* ----
  n({
    key: "video.sample_rate", group: "lock", label: "Дискретизація LOCK",
    unit: "МГц", factor: MHZ, min: 5e6, max: 61.44e6, step: 0.5e6, default: 20e6,
  }),
  n({
    key: "video.deviation_hz", group: "lock", label: "Девіація ЧМ",
    unit: "МГц", factor: MHZ, min: 1e6, max: 30e6, step: 0.5e6, default: 10e6,
  }),
  n({
    key: "video.channel_bw_hz", group: "lock", label: "Смуга каналу",
    unit: "МГц", factor: MHZ, min: 2e6, max: 40e6, step: 0.5e6, default: 10e6,
  }),
  n({
    key: "video.lo_offset_hz", group: "lock", label: "Зміщення ФАПЧ",
    help: "Для 5.8 ГГц при широкій дискретизації.",
    unit: "МГц", factor: MHZ, min: 0, max: 20e6, step: 0.5e6, default: 0,
  }),
  b({
    key: "video.afc", group: "lock", label: "АФС",
    help: "Цифровий зсув каналайзера, не крутити ФАПЧ.",
    default: true,
  }),
  n({
    key: "video.afc_gain", group: "lock", label: "Коефіцієнт АФС",
    min: 0, max: 1, step: 0.05, default: 0.5,
  }),
  n({
    key: "video.afc_limit_hz", group: "lock", label: "Стеля |АФС|",
    unit: "МГц", factor: MHZ, min: 0.1e6, max: 40e6, step: 0.5e6, default: 20e6,
  }),
  n({
    key: "video.afc_deadband_hz", group: "lock", label: "Мертва зона АФС",
    unit: "кГц", factor: KHZ, min: 0, max: 500e3, step: 1e3, default: 80e3,
  }),
  n({
    key: "video.afc_max_step_hz", group: "lock", label: "Крок АФС",
    unit: "кГц", factor: KHZ, min: 1e3, max: 500e3, step: 5e3, default: 0.25e6,
  }),
  n({
    key: "video.afc_digital_max_hz", group: "lock", label: "Макс. цифрової АФС",
    unit: "кГц", factor: KHZ, min: 50e3, max: 3e6, step: 50e3, default: 1.5e6,
  }),
  n({
    key: "video.afc_digital_headroom_hz", group: "lock", label: "Запас Найквіста АФС",
    unit: "МГц", factor: MHZ, min: 0, max: 8e6, step: 0.1e6, default: 1.5e6,
  }),
  n({
    key: "video.capture_ms", group: "lock", label: "Захоплення на кадр",
    unit: "мс", min: 10, max: 200, step: 1, default: 56,
  }),
  i({
    key: "video.stream_quality", group: "lock", label: "Якість WebP",
    min: 10, max: 100, step: 1, default: 82,
  }),
  e({
    key: "video.stream_method", group: "lock", label: "Метод WebP",
    help: "0 найшвидший, 4 найякісніший.",
    default: 0,
    options: [0, 1, 2, 3, 4],
  }),
  i({
    key: "video.rec_fps", group: "lock", label: "FPS запису",
    unit: "к/с", min: 1, max: 30, step: 1, default: 24,
  }),
  e({
    key: "video.rec_height", group: "lock", label: "Висота запису",
    default: 288,
    options: [144, 288, 480, 576],
  }),
  i({
    key: "video.rec_crf", group: "lock", label: "CRF H.264",
    help: "Менше число — краще і важче.",
    min: 16, max: 36, step: 1, default: 24,
  }),
  e({
    key: "video.rec_preset", group: "lock", label: "Пресет ffmpeg",
    default: "veryfast",
    options: ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"],
  }),
  n({
    key: "video.idle_ms", group: "lock", label: "Пауза між кадрами",
    help: "0 — без штучної стелі fps.",
    unit: "мс", min: 0, max: 200, step: 1, default: 0,
  }),
  n({
    key: "video.track_window_margin", group: "lock", label: "Запас вікна трекінгу",
    min: 1, max: 3, step: 0.05, default: 1.25,
  }),
  n({
    key: "video.h_phase_frac", group: "lock", label: "Зсув H-фази · 0 = авто",
    help: "0 — авто-unwrap. Невеликий ненульовий зсув рухає гасіння; залиш 0, якщо смуга не посередині кадру. Великі значення ігноруються: смуга ховається обрізкою, не зсувом.",
    min: -0.5, max: 0.5, step: 0.01, default: 0,
  }),
  b({
    key: "video.h_pll", group: "lock", label: "PLL",
    help: "Повільно уточнює період рядка в DecodeState. Вимк — період заморожений після сліпого захоплення.",
    default: false,
  }),
  e({
    key: "video.width", group: "lock", label: "Ширина кадру",
    default: 640,
    options: [320, 480, 640, 720],
  }),
  i({
    key: "video.spectrum_every", group: "lock", label: "Спектр раз на N кадрів",
    min: 1, max: 64, step: 1, default: 16,
  }),
  b({
    key: "video.hunt", group: "lock", label: "Цифровий hunt",
    help: "Дрібний пошук, де картинка краща.",
    default: true,
  }),
  i({
    key: "video.hunt_every", group: "lock", label: "Hunt раз на N кадрів",
    min: 1, max: 120, step: 1, default: 20,
  }),
  i({
    key: "video.hunt_after_lock", group: "lock", label: "Кадрів до першого hunt",
    min: 0, max: 120, step: 1, default: 24,
  }),
  s({
    key: "video.hunt_offsets_mhz", group: "lock", label: "Зсуви hunt",
    help: "МГц через кому. lock_target не рухаємо.",
    default: [0.25],
  }),
  n({
    key: "video.hunt_min_gain", group: "lock", label: "Мін. приріст оцінки hunt",
    min: 0, max: 1, step: 0.01, default: 0.1,
  }),
  n({
    key: "video.hunt_skip_if_score", group: "lock", label: "Не шукати, якщо оцінка ≥",
    min: 0, max: 1, step: 0.05, default: 0.7,
  }),
  n({
    key: "video.hunt_drop", group: "lock", label: "Hunt при спаді оцінки",
    min: 0, max: 1, step: 0.01, default: 0.12,
  }),
  s({
    key: "video.ffmpeg_path", group: "lock", label: "Шлях ffmpeg",
    default: "",
  }),
  i({
    key: "video.average", group: "lock", label: "Часове усереднення",
    min: 1, max: 15, step: 1, default: 5,
  }),
  n({
    key: "video.motion_thresh", group: "lock", label: "Поріг руху",
    help: "Рівні 0…255: нижче — сильніше усереднення.",
    min: 0, max: 64, step: 1, default: 24,
  }),
  b({
    key: "video.auto_levels", group: "lock", label: "Автоконтраст",
    default: true,
  }),
  n({
    key: "video.sharpen", group: "lock", label: "Апертурна корекція",
    help: "0 — вимк., 0.4–0.8 помірно.",
    min: 0, max: 2, step: 0.05, default: 0.5,
  }),
  n({
    key: "video.ring_seconds", group: "lock", label: "Кільце IQ",
    unit: "с", min: 0.1, max: 3, step: 0.1, default: 0.5,
  }),
]

export function defaultsFrom(list) {
  const out = {}
  for (const p of list) out[p.key] = cloneValue(p.default)
  return out
}

export function cloneValue(v) {
  if (Array.isArray(v)) return v.slice()
  return v
}

export function sameValue(a, b) {
  if (Array.isArray(a) || Array.isArray(b)) {
    return JSON.stringify(a ?? null) === JSON.stringify(b ?? null)
  }
  if (typeof a === "number" && typeof b === "number") {
    return Math.abs(a - b) < 1e-9
  }
  return a === b
}

export function parseList(raw) {
  if (Array.isArray(raw)) return raw.map(Number).filter(x => Number.isFinite(x))
  const s = String(raw ?? "").trim()
  if (!s) return []
  return s.split(/[,\s]+/).map(Number).filter(x => Number.isFinite(x))
}

export function formatList(v) {
  if (Array.isArray(v)) return v.join(", ")
  return String(v ?? "")
}

export function clampToSpec(spec, stored) {
  if (spec.type === "bool") return !!stored
  if (spec.type === "string") {
    if (Array.isArray(spec.default) || Array.isArray(stored)) {
      return parseList(stored)
    }
    return String(stored ?? "")
  }
  if (spec.type === "enum") {
    const opts = (spec.options || []).map(optValue)
    if (opts.length && !opts.some(o => o === stored || String(o) === String(stored))) {
      return spec.default
    }
    if (typeof spec.default === "number") return Number(stored)
    return stored
  }
  let x = Number(stored)
  if (!Number.isFinite(x)) x = Number(spec.default)
  if (spec.min != null) x = Math.max(spec.min, x)
  if (spec.max != null) x = Math.min(spec.max, x)
  const step = spec.step || (spec.type === "integer" ? 1 : 0)
  if (step > 0) {
    const base = spec.min != null ? spec.min : 0
    x = base + Math.round((x - base) / step) * step
  }
  if (spec.type === "integer") x = Math.round(x)
  return x
}

export function optValue(o) {
  return (o && typeof o === "object" && "value" in o) ? o.value : o
}

export function optLabel(o) {
  return (o && typeof o === "object" && "label" in o) ? o.label : String(o)
}

export function displayOf(spec, stored) {
  const f = spec.factor || 1
  if (spec.type === "string" && (Array.isArray(spec.default) || Array.isArray(stored))) {
    return formatList(stored)
  }
  if (typeof stored !== "number") return stored
  return stored / f
}

export function storedFromDisplay(spec, shown) {
  const f = spec.factor || 1
  if (spec.type === "string") {
    if (Array.isArray(spec.default)) return parseList(shown)
    return String(shown ?? "")
  }
  const x = Number(shown)
  if (!Number.isFinite(x)) return spec.default
  return clampToSpec(spec, x * f)
}

export function displayUnit(spec) {
  const f = spec.factor || 1
  const u = spec.unit || ""
  if (f === 1e6 && /^Hz$/i.test(u)) return "МГц"
  if (f === 1e3 && /^Hz$/i.test(u)) return "кГц"
  if (f === 1e6 && !u) return "МГц"
  return u
}

export function formatDefault(spec) {
  const v = displayOf(spec, spec.default)
  if (spec.type === "bool") return spec.default ? "увімк." : "вимк."
  if (Array.isArray(v) || spec.type === "string") return formatList(spec.default) || "—"
  if (typeof v === "number") {
    const t = Number.isInteger(v) ? String(v) : String(Math.round(v * 1000) / 1000)
    const u = displayUnit(spec)
    return u ? `${t} ${u}` : t
  }
  return String(v)
}

export const CATALOG_DEFAULTS = defaultsFrom(PARAMETER_CATALOG)
