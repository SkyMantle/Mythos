#!/usr/bin/env node
/**
 * Incomplete Pi delivery + repeat-patch guard.
 *
 * State: $CURSOR_PROJECT_DIR/.cursor/hooks/state/delivery-<conversation_id>.json
 * Flags: $CURSOR_PROJECT_DIR/.cursor/hooks/state/user_repeat_flags.jsonl
 * Ledger: .cursor/hooks/delivery_ledger.json
 */
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");

const WATCHDOG = "delivery-watchdog";

const USER_REPEAT_RE =
  /повторюєш|повторюєтесь|такий фікс уже|фікс уже був|знову те саме|вже було\)|ти знову|не стало\b|не дали жодного приросту|останній раз кажу|начарта поламав|баг після деплою|cannot import name|unexpected keyword argument|inspect_wall_s|is_disk_full_error|знову падає|знову не зламати|знову не зламати мотор|you(?:['’]re| are) repeating|same fix already|again the same/i;

const DELIVERY_RE =
  /wget\s+-O|sudo\s+cp\s+|на Pi\b|скачай|деплой|deploy(?:ing)?\b|chown\s+fpv/i;

const SYSTEM_PROMPT_RE =
  /perform any necessary follow-up|briefly inform the user about the task result|manually_attached_skills|dynamic_tools/i;

function nowMs() {
  return Date.now();
}

function eventName(payload) {
  return String(payload.hook_event_name || payload.event_name || "").trim();
}

function conversationId(payload) {
  return String(
    payload.conversation_id ||
      payload.parent_conversation_id ||
      payload.session_id ||
      "unknown",
  );
}

function generationId(payload) {
  return String(payload.generation_id || "");
}

function projectDir(payload) {
  const fromEnv = process.env.CURSOR_PROJECT_DIR || process.env.CLAUDE_PROJECT_DIR;
  if (fromEnv) return fromEnv;
  const roots = payload.workspace_roots;
  if (Array.isArray(roots) && roots[0]) return roots[0];
  return process.cwd();
}

function stateDir(payload) {
  if (process.env.DELIVERY_GUARD_STATE_DIR) return process.env.DELIVERY_GUARD_STATE_DIR;
  return path.join(projectDir(payload), ".cursor", "hooks", "state");
}

function ledgerPath(payload) {
  if (process.env.DELIVERY_LEDGER_PATH) return process.env.DELIVERY_LEDGER_PATH;
  return path.join(projectDir(payload), ".cursor", "hooks", "delivery_ledger.json");
}

function safeId(payload) {
  return conversationId(payload).replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 80) || "unknown";
}

function statePath(payload) {
  return path.join(stateDir(payload), `delivery-${safeId(payload)}.json`);
}

function flagsPath(payload) {
  return path.join(stateDir(payload), "user_repeat_flags.jsonl");
}

function readJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return fallback;
  }
}

function emptyState(payload) {
  return {
    conversation_id: conversationId(payload),
    updated_at: new Date().toISOString(),
    edited: [],
    deliveries: [],
    user_flags: [],
    notices: [],
    stop_followups: [],
    last_text: "",
  };
}

function loadLedger(payload) {
  const loaded = readJson(ledgerPath(payload), null);
  if (!loaded || typeof loaded !== "object") {
    return {
      never_auto_deploy: ["rotator.py", "config.yaml"],
      banned_wget_home: ["logging.py"],
      companions: {},
      failed_approaches: [],
    };
  }
  loaded.never_auto_deploy = Array.isArray(loaded.never_auto_deploy)
    ? loaded.never_auto_deploy
    : ["rotator.py", "config.yaml"];
  loaded.banned_wget_home = Array.isArray(loaded.banned_wget_home)
    ? loaded.banned_wget_home
    : ["logging.py"];
  loaded.companions = loaded.companions && typeof loaded.companions === "object"
    ? loaded.companions
    : {};
  loaded.failed_approaches = Array.isArray(loaded.failed_approaches)
    ? loaded.failed_approaches
    : [];
  return loaded;
}

function withLock(dir, fn) {
  fs.mkdirSync(dir, { recursive: true });
  const lock = path.join(dir, ".delivery.lock");
  const started = nowMs();
  while (true) {
    try {
      fs.writeFileSync(lock, String(process.pid), { flag: "wx" });
      break;
    } catch (err) {
      if (err && err.code !== "EEXIST") throw err;
      if (nowMs() - started > 800) {
        try {
          fs.unlinkSync(lock);
        } catch {
          /* ignore */
        }
      } else {
        const until = nowMs() + 25;
        while (nowMs() < until) {
          /* spin */
        }
      }
    }
  }
  try {
    return fn();
  } finally {
    try {
      fs.unlinkSync(lock);
    } catch {
      /* ignore */
    }
  }
}

function loadState(payload) {
  const loaded = readJson(statePath(payload), null);
  if (!loaded || typeof loaded !== "object") return emptyState(payload);
  for (const key of ["edited", "deliveries", "user_flags", "notices", "stop_followups"]) {
    loaded[key] = Array.isArray(loaded[key]) ? loaded[key] : [];
  }
  loaded.last_text = typeof loaded.last_text === "string" ? loaded.last_text : "";
  return loaded;
}

function trimState(state) {
  const cap = 200;
  for (const key of ["edited", "deliveries", "user_flags", "notices", "stop_followups"]) {
    if (state[key].length > cap) state[key] = state[key].slice(-cap);
  }
  if (state.last_text && state.last_text.length > 20000) {
    state.last_text = state.last_text.slice(-20000);
  }
  state.updated_at = new Date().toISOString();
  return state;
}

function saveState(payload, state) {
  const dir = stateDir(payload);
  fs.mkdirSync(dir, { recursive: true });
  const file = statePath(payload);
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(trimState(state), null, 2), "utf8");
  fs.renameSync(tmp, file);
}

function mutate(payload, fn) {
  return withLock(stateDir(payload), () => {
    const state = loadState(payload);
    const result = fn(state);
    saveState(payload, state);
    return result;
  });
}

function posix(p) {
  return String(p || "").replace(/\\/g, "/");
}

function toolName(payload) {
  return String(payload.tool_name || payload.tool || "");
}

function toolInput(payload) {
  const raw = payload.tool_input;
  if (raw && typeof raw === "object") return raw;
  if (typeof raw === "string") {
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object") return parsed;
    } catch {
      return { _raw: raw };
    }
  }
  return {};
}

function collectText(value, depth) {
  if (depth > 4 || value == null) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value.map((item) => collectText(item, depth + 1)).filter(Boolean).join("\n");
  }
  if (typeof value === "object") {
    if (typeof value.text === "string") return value.text;
    if (typeof value.content === "string") return value.content;
    return ["text", "content", "prompt", "message", "response"]
      .map((key) => collectText(value[key], depth + 1))
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

function promptText(payload) {
  return collectText(
    payload.prompt ||
      payload.user_prompt ||
      payload.text ||
      payload.content ||
      payload.message ||
      "",
    0,
  );
}

function responseText(payload) {
  return collectText(
    payload.text ||
      payload.response ||
      payload.assistant_message ||
      payload.message ||
      payload.content ||
      payload.last_assistant_message ||
      "",
    0,
  );
}

function commandFrom(payload) {
  const input = toolInput(payload);
  return String(payload.command || input.command || "");
}

function fileFromPayload(payload) {
  const input = toolInput(payload);
  const raw =
    payload.file_path ||
    payload.path ||
    input.path ||
    input.file_path ||
    input.target_notebook ||
    "";
  return String(raw);
}

function packageRel(absPath) {
  const n = posix(absPath);
  if (!n) return "";
  let m = n.match(/\/fpvscan\/fpvscan\/(.+)$/i);
  if (m) return m[1];
  m = n.match(/\/fpvscan\/scripts\/(.+)$/i);
  if (m) return `scripts/${m[1]}`;
  return "";
}

function isRuntimeRel(rel) {
  if (!rel) return false;
  const n = posix(rel);
  if (/(^|\/)tests\//i.test(n)) return false;
  if (/(^|\/)\.cursor\//i.test(n)) return false;
  return /\.(py|js|html|css)$/i.test(n);
}

function basenameOf(rel) {
  const n = posix(rel);
  const i = n.lastIndexOf("/");
  return i >= 0 ? n.slice(i + 1) : n;
}

function neverAuto(ledger, rel) {
  const base = basenameOf(rel);
  return ledger.never_auto_deploy.some(
    (name) => posix(name) === posix(rel) || name === base,
  );
}

function unique(items) {
  const seen = new Set();
  const out = [];
  for (const item of items) {
    const key = posix(item);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(key);
  }
  return out;
}

function companionsFor(ledger, rels) {
  const extra = [];
  for (const rel of rels) {
    const base = basenameOf(rel);
    const mapped = ledger.companions[rel] || ledger.companions[base] || [];
    extra.push(...mapped);
  }
  return unique(extra).filter((rel) => !neverAuto(ledger, rel));
}

function extractDeployedRels(text) {
  const src = String(text || "");
  const found = [];
  const urlRe = /https?:\/\/\S+\/(?:fpvscan|scripts)\/([\w./-]+\.(?:py|js|html|css))/gi;
  let match;
  while ((match = urlRe.exec(src))) {
    const rest = match[1];
    if (match[0].includes("/scripts/")) found.push(`scripts/${basenameOf(rest)}`);
    else found.push(rest);
  }
  const optRe = /\/opt\/fpvscan\/fpvscan\/([\w./-]+\.(?:py|js|html|css))/gi;
  while ((match = optRe.exec(src))) found.push(match[1]);
  const wgetRe = /wget\s+-O\s+\S+\s+\S+\/([\w./-]+\.(?:py|js|html|css))/gi;
  while ((match = wgetRe.exec(src))) {
    const rest = match[1];
    if (/^scripts\//.test(rest) || /\/scripts\//.test(match[0])) {
      found.push(rest.startsWith("scripts/") ? rest : `scripts/${basenameOf(rest)}`);
    } else if (!/^fpvscan\//.test(rest)) {
      found.push(rest);
    }
  }
  return unique(found);
}


function isNegatedLine(line) {
  return /(?:never|do not|don't|do\s+not|ніколи|не\s+(?:використовуй|качай|роби|давай)|заборонен)/i.test(
    String(line || ""),
  );
}

function commandLines(text) {
  const out = [];
  for (const raw of String(text || "").split(/\r?\n/)) {
    const line = raw.trim().replace(/^[#$%>]+\s*/, "");
    if (!line || isNegatedLine(line)) continue;
    if (/^(?:wget|curl|sudo\s+(?:cp|chown|systemctl)|chmod)\b/i.test(line)) {
      out.push(line);
      continue;
    }
    if (/\bwget\s+-O\b/i.test(line) || /\bsudo\s+cp\b/i.test(line)) out.push(line);
  }
  return out;
}

function commandish(text, opts) {
  if (opts && opts.shell) return String(text || "");
  return commandLines(text).join("\n");
}

function bannedWgetHits(text, ledger) {
  const src = String(text || "");
  const hits = [];
  if (/wget\s+-O\s+~\/logging\.py/i.test(src)) hits.push("wget -O ~/logging.py");
  if (/~\/fpv-upd\//i.test(src) || /\/fpv-upd\//i.test(src)) hits.push("~/fpv-upd/");
  if (/wget\s+-O\s+\/opt\//i.test(src)) hits.push("wget -O /opt/...");
  if (/wget\s+-O\s+~\/engine\.py\b/i.test(src)) hits.push("wget -O ~/engine.py");
  for (const name of ledger.banned_wget_home || []) {
    const re = new RegExp(`wget\\s+-O\\s+~/${name.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&")}`, "i");
    if (re.test(src) && !hits.includes(`wget -O ~/${name}`)) hits.push(`wget -O ~/${name}`);
  }
  if (/\$BASE\b/.test(src) && /wget/i.test(src) && !/10\.252\.65\.41/.test(src)) {
    hits.push("wget with $BASE and no hardcoded host");
  }
  return hits;
}

function bannedApproachHits(text, ledger, opts) {
  const wgetSrc = commandish(text, opts);
  const codeSrc = opts && opts.shell
    ? String(text || "")
    : String(text || "")
        .split(/\r?\n/)
        .filter((line) => !isNegatedLine(line))
        .join("\n");
  const hits = [];
  if (/fpv-upd\//i.test(wgetSrc)) hits.push("fpv-upd-copy-list");
  if (/wget\s+-O\s+~\/logging\.py/i.test(wgetSrc)) hits.push("stdlib-shadow-logging");
  if (/skip free_run|predicted t0.{0,40}no h|без h-фронт.{0,40}free_run/i.test(codeSrc)) {
    hits.push("sticky-no-hedge");
  }
  if (/always\s+free_run|завжди\s+.*free_run/i.test(codeSrc)) {
    hits.push("always-free-run-revert");
  }
  if (/snapshot_json\(\)|ws_state_json\(\)/i.test(codeSrc)) {
    hits.push("gil-on-event-loop");
  }
  return unique(hits);
}

function missingForDelivery(state, ledger, text, gen, opts) {
  const deployed = extractDeployedRels(text);
  const editedGen = unique(
    state.edited
      .filter((row) => !gen || row.generation_id === gen)
      .map((row) => row.rel)
      .filter((rel) => isRuntimeRel(rel) && !neverAuto(ledger, rel)),
  );
  const editedConv = unique(
    state.edited
      .map((row) => row.rel)
      .filter((rel) => isRuntimeRel(rel) && !neverAuto(ledger, rel)),
  );
  const required = unique([
    ...editedGen,
    ...companionsFor(ledger, unique([...deployed, ...editedGen])),
  ]);
  const missing = required.filter((rel) => {
    if (!deployed.length) return true;
    return !deployed.some(
      (got) => posix(got) === posix(rel) || basenameOf(got) === basenameOf(rel),
    );
  });
  const olderMissing = editedConv.filter((rel) => {
    if (editedGen.includes(rel)) return false;
    if (!deployed.length) return false;
    return !deployed.some(
      (got) => posix(got) === posix(rel) || basenameOf(got) === basenameOf(rel),
    );
  });
  return {
    deployed,
    editedGen,
    editedConv,
    missing,
    olderMissing,
    banned: bannedWgetHits(commandish(text, opts), ledger),
    approaches: bannedApproachHits(text, ledger, opts),
  };
}

function recordEdit(state, payload, absPath) {
  const rel = packageRel(absPath);
  if (!isRuntimeRel(rel)) return;
  const gen = generationId(payload);
  const existing = state.edited.find(
    (row) => row.rel === rel && row.generation_id === gen,
  );
  if (existing) {
    existing.ts = nowMs();
    existing.path = absPath;
    return;
  }
  state.edited.push({
    ts: nowMs(),
    generation_id: gen,
    path: absPath,
    rel,
  });
}

function stripUserQuery(text) {
  const src = String(text || "");
  const m = src.match(/<user_query>\s*([\s\S]*?)\s*<\/user_query>/i);
  return (m ? m[1] : src).trim();
}

function appendFlag(payload, rec) {
  const line = JSON.stringify({
    ts: new Date().toISOString(),
    conversation_id: conversationId(payload),
    generation_id: generationId(payload),
    ...rec,
  });
  fs.mkdirSync(stateDir(payload), { recursive: true });
  fs.appendFileSync(flagsPath(payload), `${line}\n`, "utf8");
}

function contextFromState(state, ledger) {
  const lastFlags = state.user_flags.slice(-3);
  const lines = [
    "Delivery guard is active. Before Pi wget, include every edited runtime file plus companions from .cursor/hooks/delivery_ledger.json.",
    "Never wget -O ~/logging.py, never ~/fpv-upd/, never empty $BASE, never wget -O /opt/.",
    "Do not repeat sticky-t0-without-H-edge, always-free_run revert, GIL snapshot on the asyncio loop, or last-session MHz as architecture.",
    `You may invoke ${WATCHDOG} once this turn if wget is incomplete or the operator flagged a repeat — never in a loop.`,
  ];
  if (lastFlags.length) {
    lines.push(
      "Operator repeat/missing-file flags: " +
        lastFlags.map((row) => `"${String(row.quote || "").slice(0, 120)}"`).join(" | "),
    );
  }
  const failed = (ledger.failed_approaches || []).slice(0, 6);
  if (failed.length) {
    lines.push(
      "Banned prior approaches: " + failed.map((row) => row.id).join(", ") + ".",
    );
  }
  return lines.join(" ");
}

function verdictMessage(check, kind) {
  const bits = [];
  if (check.missing.length) {
    bits.push(`недодані файли: ${check.missing.join(", ")}`);
  }
  if (check.olderMissing.length) {
    bits.push(`також змінювались у цьому чаті: ${check.olderMissing.join(", ")}`);
  }
  if (check.banned.length) {
    bits.push(`заборонений wget: ${check.banned.join(", ")}`);
  }
  if (check.approaches.length) {
    bits.push(`застарілий підхід: ${check.approaches.join(", ")}`);
  }
  if (!bits.length) return "";
  const head =
    kind === "followup"
      ? "Delivery guard: список wget неповний або повторює зламаний підхід."
      : "Delivery guard:";
  return [
    head,
    bits.join(". ") + ".",
    "Додай відсутні файли унікальними ~/fpv-* іменами, sudo cp + chown fpv:fpv.",
    `Invoke ${WATCHDOG} at most once, then stop repeating the previous patch.`,
  ].join(" ");
}

function deny(userMessage, agentMessage) {
  const out = { permission: "deny", user_message: userMessage };
  if (agentMessage) out.agent_message = agentMessage;
  return out;
}


function escapeRe(text) {
  return String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function allow(extra) {
  return Object.assign({ permission: "allow" }, extra || {});
}

function onSessionStart(payload) {
  const ledger = loadLedger(payload);
  return mutate(payload, (state) => {
    state.conversation_id = conversationId(payload);
    return { additional_context: contextFromState(state, ledger) };
  });
}

function onBeforeSubmit(payload) {
  const raw = stripUserQuery(promptText(payload));
  if (!raw || SYSTEM_PROMPT_RE.test(raw)) return {};
  if (!USER_REPEAT_RE.test(raw) && !/ImportError|TypeError|AttributeError/.test(raw)) {
    return {};
  }
  const quote = raw.replace(/\s+/g, " ").slice(0, 280);
  const ledger = loadLedger(payload);
  return mutate(payload, (state) => {
    state.user_flags.push({
      ts: nowMs(),
      generation_id: generationId(payload),
      quote,
    });
    appendFlag(payload, { kind: "user_repeat", quote });
    return {
      additional_context: [
        "Operator flagged a repeat, a missing-file deploy, or the same failure again.",
        `Quote: "${quote}".`,
        "Do not ship the previous patch. Check edited files vs wget companions.",
        `Invoke ${WATCHDOG} once this turn, then change approach.`,
        contextFromState(state, ledger),
      ].join(" "),
      user_message: "Зафіксовано скаргу на повтор або недоданий файл. Агент не повинен віддавати той самий wget/патч.",
    };
  });
}

function onAfterFileEdit(payload) {
  const abs = fileFromPayload(payload);
  if (!abs) return {};
  mutate(payload, (state) => {
    recordEdit(state, payload, abs);
  });
  return {};
}

function onPostToolUse(payload) {
  const name = toolName(payload).toLowerCase();
  if (name === "write" || name === "strreplace" || name.includes("edit")) {
    const abs = fileFromPayload(payload);
    if (abs) {
      mutate(payload, (state) => {
        recordEdit(state, payload, abs);
      });
    }
  }
  return {};
}

function onPreToolUse() {
  return {};
}

function onBeforeShell(payload) {
  const cmd = commandFrom(payload);
  const ledger = loadLedger(payload);
  const banned = bannedWgetHits(cmd, ledger);
  if (!banned.length) return {};
  return deny(
    `Хук зупинив небезпечну wget/cp команду: ${banned.join(", ")}.`,
    [
      `Delivery guard blocked shell: ${banned.join(", ")}.`,
      "Hardcode http://10.252.65.41:8000, wget to ~/fpv-*.py, then sudo cp + chown. Never ~/logging.py, never /opt wget.",
    ].join(" "),
  );
}

function onAfterAgentResponse(payload) {
  const text = responseText(payload);
  if (!text) return {};
  const ledger = loadLedger(payload);
  return mutate(payload, (state) => {
    state.last_text = text;
    if (!DELIVERY_RE.test(text)) return {};
    const check = missingForDelivery(state, ledger, text, generationId(payload), {});
    state.deliveries.push({
      ts: nowMs(),
      generation_id: generationId(payload),
      deployed: check.deployed,
      missing: check.missing,
    });
    const notice = verdictMessage(check, "notice");
    if (!notice) return {};
    state.notices.push({ ts: nowMs(), detail: notice });
    return { additional_context: notice };
  });
}

function onStop(payload) {
  const ledger = loadLedger(payload);
  const gen = generationId(payload);
  return mutate(payload, (state) => {
    const already = state.stop_followups.some((row) => row.generation_id === gen && gen);
    if (already) return {};
    const text = responseText(payload) || state.last_text || "";
    if (!DELIVERY_RE.test(text)) return {};
    const check = missingForDelivery(state, ledger, text, gen, {});
    const notice = verdictMessage(check, "followup");
    if (!notice) return {};
    state.stop_followups.push({ ts: nowMs(), generation_id: gen });
    return { followup_message: notice };
  });
}

function handle(payload) {
  switch (eventName(payload)) {
    case "sessionStart":
      return onSessionStart(payload);
    case "beforeSubmitPrompt":
      return onBeforeSubmit(payload);
    case "afterFileEdit":
      return onAfterFileEdit(payload);
    case "preToolUse":
      return onPreToolUse(payload);
    case "postToolUse":
      return onPostToolUse(payload);
    case "beforeShellExecution":
      return onBeforeShell(payload);
    case "afterAgentResponse":
    case "afterAgentThought":
      return onAfterAgentResponse(payload);
    case "stop":
      return onStop(payload);
    default:
      return {};
  }
}

function readStdin() {
  try {
    return fs.readFileSync(0, "utf8");
  } catch {
    return "";
  }
}

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

function selfTest() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "delivery-guard-"));
  process.env.DELIVERY_GUARD_STATE_DIR = dir;
  const ledgerFile = path.join(dir, "delivery_ledger.json");
  fs.writeFileSync(
    ledgerFile,
    fs.readFileSync(path.join(__dirname, "delivery_ledger.json"), "utf8"),
    "utf8",
  );
  process.env.DELIVERY_LEDGER_PATH = ledgerFile;
  const conv = { conversation_id: "c-del", generation_id: "g1", workspace_roots: [dir] };

  handle({
    ...conv,
    hook_event_name: "afterFileEdit",
    file_path: "D:/projects/Mythos/fpvscan/fpvscan/engine.py",
  });
  handle({
    ...conv,
    hook_event_name: "postToolUse",
    tool_name: "Write",
    tool_input: { path: "D:/projects/Mythos/fpvscan/fpvscan/dsp/streaming.py" },
  });

  const prompt = handle({
    ...conv,
    hook_event_name: "beforeSubmitPrompt",
    prompt: "Такий фікс уже був) Ти повторюєшся?",
  });
  assert(typeof prompt.additional_context === "string", "repeat prompt should inject context");
  assert(/повторюєшся/.test(prompt.additional_context), "quote should be stored in context");

  const badShell = handle({
    ...conv,
    hook_event_name: "beforeShellExecution",
    command: "wget -O ~/logging.py http://10.252.65.41:8000/fpvscan/app/core/logging.py",
  });
  assert(badShell.permission === "deny", "logging.py wget should deny");

  const okShell = handle({
    ...conv,
    hook_event_name: "beforeShellExecution",
    command: "wget -O ~/fpv-engine.py http://10.252.65.41:8000/fpvscan/engine.py",
  });
  assert(okShell.permission !== "deny", "unique wget name should not deny");

  const partial = handle({
    ...conv,
    hook_event_name: "afterAgentResponse",
    text: [
      "На Pi:",
      "wget -O ~/fpv-engine.py http://10.252.65.41:8000/fpvscan/engine.py",
      "sudo cp ~/fpv-engine.py /opt/fpvscan/fpvscan/engine.py",
      "sudo chown fpv:fpv /opt/fpvscan/fpvscan/engine.py",
    ].join("\n"),
  });
  assert(typeof partial.additional_context === "string", "partial wget should warn");
  assert(/streaming\.py|cvbs\.py|file\.py|scan_gate/.test(partial.additional_context), "companions must be listed");

  const stop = handle({
    ...conv,
    hook_event_name: "stop",
  });
  assert(typeof stop.followup_message === "string", "stop should follow up once");

  const stop2 = handle({
    ...conv,
    hook_event_name: "stop",
  });
  assert(!stop2.followup_message, "second stop follow-up this generation should be empty");

  const start = handle({
    ...conv,
    hook_event_name: "sessionStart",
    session_id: "c-del",
  });
  assert(typeof start.additional_context === "string", "sessionStart should inject ledger");

  const docs = handle({
    ...conv,
    generation_id: "g-docs",
    hook_event_name: "afterAgentResponse",
    text: [
      "Never use home copy lists from a missing update folder.",
      "Never write wget output onto the install prefix.",
      "Never empty host variables.",
      "Delivery guard is active.",
    ].join("\n"),
  });
  assert(!docs.additional_context, "documentation of banned wget must not warn");

  const docsStop = handle({
    ...conv,
    generation_id: "g-docs",
    hook_event_name: "stop",
  });
  assert(!docsStop.followup_message, "documentation must not trigger stop follow-up");

  const docsReal = handle({
    ...conv,
    generation_id: "g-docs2",
    hook_event_name: "afterAgentResponse",
    text: [
      "Never use ~/" + "fpv-upd/",
      "Never wget -O /opt/.",
      "Never empty $BASE.",
    ].join("\n"),
  });
  assert(!docsReal.additional_context, "negated banned phrases must not warn");

  fs.rmSync(dir, { recursive: true, force: true });
  process.stderr.write("delivery_guard self-test passed\n");
}

function main() {
  if (process.argv.includes("--self-test")) {
    selfTest();
    return;
  }
  let payload = {};
  const raw = readStdin().trim();
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch (err) {
      process.stderr.write(`delivery_guard: invalid JSON on stdin: ${err.message}\n`);
      process.stdout.write("{}\n");
      process.exit(0);
      return;
    }
  }
  let out = {};
  try {
    out = handle(payload) || {};
  } catch (err) {
    process.stderr.write(`delivery_guard: ${err.stack || err.message}\n`);
    out = {};
  }
  process.stdout.write(`${JSON.stringify(out)}\n`);
}

if (require.main === module) {
  main();
}

module.exports = { handle, extractDeployedRels, packageRel, WATCHDOG };
