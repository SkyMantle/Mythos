/**
 * Клієнт /api/test/* з локальним запасом.
 * Керування свіпом/утриманням/відео лишається на існуючих /api/lock, /api/sweep, /ws.
 */
import {
  PARAMETER_CATALOG, CATALOG_DEFAULTS, DEFAULT_SCHEMA,
  clampToSpec, cloneValue,
} from "./catalog.js"

const UK_HINT = Object.fromEntries(PARAMETER_CATALOG.map(p => [p.key, p]))

export const TEST_PREFIX = "/api/test"
const LS_KEY = "fpvscan.test.v1"

function uuid() {
  return crypto.randomUUID()
}

function loadLocal() {
  try {
    const raw = localStorage.getItem(LS_KEY)
    if (raw) return JSON.parse(raw)
  } catch { /* ignore */ }
  return {
    values: { ...CATALOG_DEFAULTS },
    schema: { fields: DEFAULT_SCHEMA.map(f => ({ ...f })) },
    sessions: [],
    observations: {},
  }
}

function saveLocal(db) {
  try { localStorage.setItem(LS_KEY, JSON.stringify(db)) } catch { /* quota */ }
}

function nowIso() {
  return new Date().toISOString()
}

async function req(method, path, body, timeoutMs = 4000) {
  const ctrl = new AbortController()
  const t = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const r = await fetch(TEST_PREFIX + path, {
      method,
      headers: body == null ? {} : { "Content-Type": "application/json" },
      body: body == null ? undefined : JSON.stringify(body),
      signal: ctrl.signal,
    })
    const text = await r.text()
    let data = null
    if (text) {
      try { data = JSON.parse(text) } catch { data = { raw: text } }
    }
    return { ok: r.ok, status: r.status, data }
  } catch (err) {
    return { ok: false, status: 0, data: null, error: err }
  } finally {
    clearTimeout(t)
  }
}

function asParamList(data) {
  if (!data) return null
  if (Array.isArray(data)) return data
  if (Array.isArray(data.parameters)) return data.parameters
  if (Array.isArray(data.items)) return data.items
  if (Array.isArray(data.fields)) return data.fields
  return null
}

function flatten(obj, prefix = "") {
  const out = {}
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return out
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k
    if (v && typeof v === "object" && !Array.isArray(v)) Object.assign(out, flatten(v, key))
    else out[key] = v
  }
  return out
}

function asValues(data) {
  if (!data || typeof data !== "object") return null
  const src = (data.values && typeof data.values === "object" && !Array.isArray(data.values))
    ? data.values
    : (data.current && typeof data.current === "object" ? data.current : null)
  if (src) {
    const flat = flatten(src)
    if (Object.keys(flat).length) return flat
  }
  if (data.scan || data.video || data.sdr) {
    return flatten({ scan: data.scan, video: data.video, sdr: data.sdr })
  }
  const keys = Object.keys(data).filter(k => k.includes("."))
  if (keys.length) {
    const out = {}
    for (const k of keys) out[k] = data[k]
    return out
  }
  return null
}

function asFields(data) {
  if (!data) return null
  if (Array.isArray(data)) return data
  if (Array.isArray(data.fields)) return data.fields
  if (data.schema && Array.isArray(data.schema.fields)) return data.schema.fields
  return null
}

function asSessionList(data) {
  if (!data) return []
  if (Array.isArray(data)) return data
  if (Array.isArray(data.sessions)) return data.sessions
  if (Array.isArray(data.items)) return data.items
  return []
}

function asObsList(data) {
  if (!data) return []
  if (Array.isArray(data)) return data
  if (Array.isArray(data.observations)) return data.observations
  if (Array.isArray(data.items)) return data.items
  return []
}

export class TestClient {
  constructor() {
    this.remote = false
    this.db = loadLocal()
    this.catalog = PARAMETER_CATALOG
    this.defaults = { ...CATALOG_DEFAULTS }
  }

  async probe() {
    const r = await req("GET", "/parameters")
    const list = asParamList(r.data)
    if (r.ok && list && list.length) {
      this.remote = true
      this.catalog = list.map(normalizeParam)
      this.defaults = {}
      const currents = {}
      for (const p of this.catalog) {
        this.defaults[p.key] = cloneValue(p.default)
        if (p.current !== undefined) currents[p.key] = cloneValue(p.current)
      }
      if (r.data.defaults && typeof r.data.defaults === "object") {
        Object.assign(this.defaults, r.data.defaults)
      }
      const cur = await this.currentParameters()
      this.db.values = { ...this.defaults, ...currents, ...cur }
    } else {
      this.remote = false
      this.catalog = PARAMETER_CATALOG
      this.defaults = { ...CATALOG_DEFAULTS }
      this.db.values = { ...this.defaults, ...this.db.values }
    }
    saveLocal(this.db)
    return { remote: this.remote, catalog: this.catalog, defaults: this.defaults, values: this.db.values }
  }

  spec(key) {
    return this.catalog.find(p => p.key === key)
  }

  async listParameters(mode) {
    const q = mode ? `?mode=${encodeURIComponent(mode)}` : ""
    const r = await req("GET", "/parameters" + q)
    const list = asParamList(r.data)
    if (r.ok && list && list.length) return list.map(normalizeParam)
    return null
  }

  async currentParameters() {
    const r = await req("GET", "/parameters/current")
    if (r.ok) {
      const v = asValues(r.data)
      if (v) return v
    }
    return { ...this.db.values }
  }

  async setParameters(patch) {
    const next = { ...this.db.values }
    for (const [k, v] of Object.entries(patch)) {
      const spec = this.spec(k)
      next[k] = spec ? clampToSpec(spec, v) : v
    }
    this.db.values = next
    saveLocal(this.db)
    const body = { idempotency_key: uuid(), values: patch }
    const r = await req("PUT", "/parameters", body)
    if (r.ok && r.data && r.data.values) Object.assign(next, r.data.values)
    this.db.values = next
    saveLocal(this.db)
    return {
      values: next,
      remote: r.ok,
      status: r.status,
      error: r.ok ? null : errText(r),
      pending_keys: r.ok ? (r.data && r.data.pending_keys) || [] : [],
      pending_reasons: r.ok ? (r.data && r.data.pending_reasons) || {} : {},
      applied_keys: r.ok ? (r.data && r.data.applied_keys) || Object.keys(patch) : [],
    }
  }

  async live() {
    const r = await req("GET", "/live", undefined, 2500)
    if (r.ok && r.data && typeof r.data === "object") return r.data
    return null
  }

  async getSchema() {
    const r = await req("GET", "/observation-schema")
    if (r.ok) {
      const fields = asFields(r.data)
      if (fields) {
        this.db.schema = { fields: fields.map(normalizeField) }
        saveLocal(this.db)
        return this.db.schema
      }
    }
    this.db.schema = { fields: (this.db.schema.fields || DEFAULT_SCHEMA).map(normalizeField) }
    return this.db.schema
  }

  async setSchema(fields) {
    const clean = fields.map(fieldForApi)
    this.db.schema = { fields: clean.map(normalizeField) }
    saveLocal(this.db)
    const r = await req("PUT", "/observation-schema", { idempotency_key: uuid(), fields: clean })
    return { schema: this.db.schema, remote: r.ok, error: r.ok ? null : errText(r) }
  }

  async listSessions() {
    const r = await req("GET", "/sessions")
    if (r.ok) {
      const list = asSessionList(r.data)
      this.db.sessions = list
      saveLocal(this.db)
      return list
    }
    return this.db.sessions
  }

  async getSession(id) {
    const r = await req("GET", `/sessions/${id}`)
    if (r.ok && r.data) return r.data
    return this.db.sessions.find(s => s.id === id) || null
  }

  async createSession({ title, mode }) {
    const local = {
      id: uuid(),
      title: title || untitled(mode),
      mode: mode || "lock",
      status: "active",
      created_at: nowIso(),
      updated_at: nowIso(),
    }
    this.db.sessions.unshift(local)
    this.db.observations[local.id] = []
    saveLocal(this.db)
    const r = await req("POST", "/sessions", {
      idempotency_key: uuid(),
      title: local.title,
      mode: ["sweep", "lock", "manual"].includes(local.mode) ? local.mode : "lock",
      parameter_snapshot: { ...this.db.values },
    })
    if (r.ok && r.data && r.data.id) {
      const remote = normalizeSession(r.data, local)
      this.db.sessions = this.db.sessions.map(s => s.id === local.id ? remote : s)
      this.db.observations[remote.id] = this.db.observations[local.id] || []
      if (remote.id !== local.id) delete this.db.observations[local.id]
      saveLocal(this.db)
      return remote
    }
    return local
  }

  async patchSession(id, patch) {
    const i = this.db.sessions.findIndex(s => s.id === id)
    if (i >= 0) {
      this.db.sessions[i] = { ...this.db.sessions[i], ...patch, updated_at: nowIso() }
      saveLocal(this.db)
    }
    const r = await req("PATCH", `/sessions/${id}`, {
      title: patch.title,
      notes: patch.notes,
      status: patch.status,
    })
    if (r.ok && r.data) {
      if (i >= 0) this.db.sessions[i] = normalizeSession(r.data, this.db.sessions[i])
      saveLocal(this.db)
      return this.db.sessions[i]
    }
    return i >= 0 ? this.db.sessions[i] : null
  }

  async completeSession(id) {
    const r = await req("POST", `/sessions/${id}/complete`, { idempotency_key: uuid() })
    const i = this.db.sessions.findIndex(s => s.id === id)
    const extra = (r.ok && r.data && typeof r.data === "object") ? r.data : {}
    if (i >= 0) {
      this.db.sessions[i] = {
        ...this.db.sessions[i],
        ...extra,
        status: extra.status || "completed",
        updated_at: extra.updated_at || nowIso(),
      }
      saveLocal(this.db)
      return this.db.sessions[i]
    }
    return { id, status: "completed", ...extra }
  }

  async listObservations(sessionId) {
    const r = await req("GET", `/sessions/${sessionId}/observations`)
    if (r.ok) {
      const list = asObsList(r.data)
      this.db.observations[sessionId] = list
      saveLocal(this.db)
      return list
    }
    return this.db.observations[sessionId] || []
  }

  async addObservation(sessionId, payload) {
    const body = observationBody(payload)
    const local = {
      id: uuid(),
      session_id: sessionId,
      created_at: nowIso(),
      updated_at: nowIso(),
      timestamp_ms: body.timestamp_ms,
      ...body,
    }
    const list = this.db.observations[sessionId] || []
    list.push(local)
    this.db.observations[sessionId] = list
    saveLocal(this.db)
    const r = await req("POST", `/sessions/${sessionId}/observations`, {
      idempotency_key: uuid(),
      ...body,
    })
    if (r.ok && r.data && r.data.id) {
      const remote = { ...local, ...r.data }
      this.db.observations[sessionId] = list.map(o => o.id === local.id ? remote : o)
      saveLocal(this.db)
      return remote
    }
    return { ...local, error: errText(r) }
  }

  async patchObservation(sessionId, obsId, patch) {
    const body = observationBody(patch, true)
    const list = this.db.observations[sessionId] || []
    const i = list.findIndex(o => o.id === obsId)
    if (i >= 0) {
      list[i] = { ...list[i], ...body, updated_at: nowIso() }
      saveLocal(this.db)
    }
    const r = await req("PATCH", `/sessions/${sessionId}/observations/${obsId}`, body)
    if (r.ok && r.data && i >= 0) {
      list[i] = { ...list[i], ...r.data }
      saveLocal(this.db)
    }
    return i >= 0 ? list[i] : null
  }

  async deleteObservation(sessionId, obsId) {
    const list = (this.db.observations[sessionId] || []).filter(o => o.id !== obsId)
    this.db.observations[sessionId] = list
    saveLocal(this.db)
    let r = await req("DELETE", `/sessions/${sessionId}/observations/${obsId}`)
    if (!r.ok) await req("DELETE", `/observations/${obsId}`)
    return true
  }
}

function untitled(mode) {
  const t = new Date()
  const pad = n => String(n).padStart(2, "0")
  const stamp = `${pad(t.getHours())}:${pad(t.getMinutes())}`
  const m = mode === "sweep" ? "свіп" : mode === "manual" ? "вручну" : "утримання"
  return `Сесія ${m} ${stamp}`
}

function normalizeSession(data, fallback) {
  return {
    id: data.id || fallback.id,
    title: data.title || fallback.title,
    mode: data.mode || fallback.mode,
    status: data.status || fallback.status,
    created_at: data.created_at || fallback.created_at,
    updated_at: data.updated_at || fallback.updated_at,
    ...data,
  }
}

function inferFactor(p) {
  const unit = String(p.unit || "")
  const max = Number(p.max)
  const min = Number(p.min)
  const span = Number.isFinite(max)
    ? Math.max(Math.abs(max), Number.isFinite(min) ? Math.abs(min) : 0)
    : NaN
  if (/кГц|kHz/i.test(unit)) return Number.isFinite(span) && span < 1000 ? 1 : 1e3
  if (/МГц|MHz/i.test(unit)) {
    if (Number.isFinite(span) && span < 100) return 1
    if (Number.isFinite(span) && span < 5e6) return 1e3
    return 1e6
  }
  if (/^Hz$/i.test(unit) || String(p.key || "").endsWith("_hz")) {
    if (!Number.isFinite(span) || span <= 5000) return 1
    if (span < 5e6) return 1e3
    return 1e6
  }
  return 1
}

function controlType(t) {
  if (t === "float" || t === "number") return "number"
  if (t === "int" || t === "integer") return "integer"
  if (t === "bool") return "bool"
  if (t === "enum") return "enum"
  return "string"
}

function normalizeParam(p) {
  const hint = UK_HINT[p.key] || {}
  const type = controlType(p.type)
  const labels = p.enum_labels || hint.enum_labels || {}
  const options = p.options || (p.enum_values || hint.options || []).map(v => {
    const n = Number(v)
    const value = (typeof p.default === "number" && Number.isFinite(n) && String(n) === String(v)) ? n : v
    const label = labels[String(v)]
    return label ? { value, label } : value
  })
  const apiFactor = p.factor != null && Number(p.factor) > 0 ? Number(p.factor) : null
  const merged = {
    ...p,
    help: p.help || p.description || hint.help || "",
    unit: p.unit || hint.unit,
    min: p.min ?? hint.min,
    max: p.max ?? hint.max,
    step: p.step ?? hint.step,
  }
  const factor = apiFactor != null ? apiFactor : inferFactor(merged)
  return {
    ...merged,
    type,
    options,
    factor,
    step: merged.step ?? (type === "integer" ? 1 : 1),
    group: p.group || hint.group || "shared",
    label: hint.label || p.label || p.key,
    applies_to: p.applies_to || hint.applies_to || [],
    affects: p.affects || hint.affects || null,
    task: p.task || p.task_id || null,
    tasks: p.tasks || p.task_tags || [],
    modes: p.modes || p.applies_to || [],
  }
}

function normalizeField(f) {
  const key = f.key || f.id
  return {
    key,
    id: key,
    label: f.label || key,
    type: f.type === "scale" ? "rating" : f.type,
    required: !!f.required,
    enum_values: f.enum_values || f.options || null,
    options: f.enum_values || f.options || null,
    min: f.min,
    max: f.max,
  }
}

function fieldForApi(f) {
  const type = f.type === "scale" ? "rating" : f.type
  return {
    key: f.key || f.id,
    label: f.label || f.key || f.id,
    type,
    required: !!f.required,
    enum_values: type === "enum" ? (f.enum_values || f.options || []) : null,
    min: f.min ?? null,
    max: f.max ?? null,
  }
}

function observationBody(payload, partial = false) {
  const body = {}
  if (payload.timestamp_ms != null) body.timestamp_ms = Math.round(Number(payload.timestamp_ms))
  else if (!partial) body.timestamp_ms = Date.now()
  if (payload.frequency_hz != null && Number.isFinite(Number(payload.frequency_hz))) {
    body.frequency_hz = Number(payload.frequency_hz)
  }
  if (payload.rssi != null && Number.isFinite(Number(payload.rssi))) body.rssi = Number(payload.rssi)
  if (payload.snr != null && Number.isFinite(Number(payload.snr))) body.snr = Number(payload.snr)
  if (payload.video_metrics && typeof payload.video_metrics === "object") {
    body.video_metrics = payload.video_metrics
  }
  if (payload.parameter_snapshot && typeof payload.parameter_snapshot === "object") {
    body.parameter_snapshot = payload.parameter_snapshot
  }
  body.fields = sanitizeObsFields(payload.fields)
  if (partial && payload.fields == null) delete body.fields
  if (payload.frame_ref) body.frame_ref = String(payload.frame_ref)
  return body
}

function sanitizeObsFields(fields) {
  const out = {}
  if (!fields || typeof fields !== "object") return out
  for (const [k, v] of Object.entries(fields)) {
    if (v === "" || v == null) continue
    if (typeof v === "number" && !Number.isFinite(v)) continue
    if (Array.isArray(v) && !v.length) continue
    out[k] = v
  }
  return out
}

function errText(r) {
  if (!r || r.ok) return null
  const d = r.data
  if (!d) return `HTTP ${r.status}`
  if (typeof d.detail === "string") return d.detail
  if (Array.isArray(d.detail)) return d.detail.map(x => x.msg || JSON.stringify(x)).join("; ")
  if (d.error) return String(d.error)
  if (d.message) return String(d.message)
  return `HTTP ${r.status}`
}

export function engineLock(freqHz) {
  return fetch("/api/lock/" + freqHz, { method: "POST" })
}
export function engineSweep() {
  return fetch("/api/sweep", { method: "POST" })
}
export function engineClear() {
  return fetch("/api/clear", { method: "POST" })
}
export function engineSnapshot() {
  return fetch("/api/snapshot", { method: "POST" })
}
export function engineRecord(on) {
  return fetch("/api/record/" + (on ? "start" : "stop"), { method: "POST" })
}
export function engineBias(on) {
  return fetch("/api/bias_tee/" + (on ? "on" : "off"), { method: "POST" })
}
export function engineState() {
  return fetch("/api/state").then(r => r.json())
}
