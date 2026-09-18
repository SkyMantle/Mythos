#!/usr/bin/env node
/**
 * Block repeated subagent launches and repeated failing tool/shell calls.
 *
 * State: $CURSOR_PROJECT_DIR/.cursor/hooks/state/<conversation_id>.json
 * Audit: $CURSOR_PROJECT_DIR/.cursor/hooks/state/events.jsonl
 */
"use strict";

const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");

const WATCHDOG_TYPE = "loop-watchdog";
const WATCHDOG_TYPES = new Set(["loop-watchdog", "delivery-watchdog"]);

const LIMITS = {
  sameTaskPerGeneration: 2,
  sameTaskPerConversation: 4,
  sameTypePerGeneration: 5,
  totalPerGeneration: 10,
  sameFailurePerGeneration: 2,
  sameFailurePerConversation: 3,
  watchdogPerGeneration: 1,
  watchdogPerConversation: 2,
};

function sha1(text) {
  return crypto.createHash("sha1").update(String(text)).digest("hex").slice(0, 16);
}

function normalize(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/\r/g, "")
    .replace(
      /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi,
      "<uuid>",
    )
    .replace(/\b[0-9a-f]{16,}\b/gi, "<hex>")
    .replace(/https?:\/\/\S+/gi, "<url>")
    .replace(/\s+/g, " ")
    .trim();
}

function fingerprint(parts) {
  return sha1(normalize(parts.filter(Boolean).join("|")).slice(0, 400));
}

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
  if (process.env.LOOP_GUARD_STATE_DIR) return process.env.LOOP_GUARD_STATE_DIR;
  return path.join(projectDir(payload), ".cursor", "hooks", "state");
}

function statePath(payload) {
  const safe = conversationId(payload).replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 80);
  return path.join(stateDir(payload), `${safe || "unknown"}.json`);
}

function eventsPath(payload) {
  return path.join(stateDir(payload), "events.jsonl");
}

function emptyState(payload) {
  return {
    conversation_id: conversationId(payload),
    updated_at: new Date().toISOString(),
    subagents: [],
    failures: [],
    denials: [],
    notices: [],
  };
}

function readJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return fallback;
  }
}

function withLock(dir, fn) {
  fs.mkdirSync(dir, { recursive: true });
  const lock = path.join(dir, ".lock");
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
          /* wait for the other hook process to drop the lock */
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
  const file = statePath(payload);
  const loaded = readJson(file, null);
  if (!loaded || typeof loaded !== "object") return emptyState(payload);
  loaded.subagents = Array.isArray(loaded.subagents) ? loaded.subagents : [];
  loaded.failures = Array.isArray(loaded.failures) ? loaded.failures : [];
  loaded.denials = Array.isArray(loaded.denials) ? loaded.denials : [];
  loaded.notices = Array.isArray(loaded.notices) ? loaded.notices : [];
  return loaded;
}

function trimState(state) {
  const cap = 200;
  for (const key of ["subagents", "failures", "denials", "notices"]) {
    if (state[key].length > cap) state[key] = state[key].slice(-cap);
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

function appendEvent(payload, record) {
  const line = JSON.stringify({
    ts: new Date().toISOString(),
    event: eventName(payload),
    conversation_id: conversationId(payload),
    generation_id: generationId(payload),
    ...record,
  });
  fs.mkdirSync(stateDir(payload), { recursive: true });
  fs.appendFileSync(eventsPath(payload), `${line}\n`, "utf8");
}

function mutate(payload, fn) {
  return withLock(stateDir(payload), () => {
    const state = loadState(payload);
    const result = fn(state);
    saveState(payload, state);
    return result;
  });
}

function isWatchdog(type) {
  return WATCHDOG_TYPES.has(normalize(type));
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

function callId(payload) {
  return String(payload.tool_use_id || payload.tool_call_id || payload.subagent_id || "");
}

function isTaskTool(name) {
  const n = name.toLowerCase();
  return n === "task" || n.endsWith(":task") || n.includes("subagent");
}

function taskFields(payload) {
  const input = toolInput(payload);
  const type = String(
    payload.subagent_type || input.subagent_type || input.subagentType || "",
  );
  const task = String(
    payload.task ||
      input.prompt ||
      input.description ||
      input.task ||
      "",
  );
  return {
    type,
    task,
    resume: input.resume,
    parallel: Boolean(payload.is_parallel_worker || input.run_in_background),
    fp: fingerprint([type, String(task).slice(0, 240)]),
  };
}

function commandFrom(payload) {
  const input = toolInput(payload);
  return String(payload.command || input.command || "");
}

function inputFingerprint(payload) {
  const name = toolName(payload) || (commandFrom(payload) ? "Shell" : "unknown");
  const input = toolInput(payload);
  const cmd = commandFrom(payload);
  if (cmd) return fingerprint([name, cmd]);
  if (isTaskTool(name)) {
    const t = taskFields(payload);
    return fingerprint([name, t.type, t.task.slice(0, 240)]);
  }
  const mcpName = String(payload.mcp_tool_name || input.tool_name || "");
  return fingerprint([name, mcpName, JSON.stringify(input).slice(0, 400)]);
}

function deny(userMessage, agentMessage, extra) {
  const out = {
    permission: "deny",
    user_message: userMessage,
  };
  if (agentMessage) out.agent_message = agentMessage;
  return Object.assign(out, extra || {});
}

function allow(extra) {
  return Object.assign({ permission: "allow" }, extra || {});
}

function countSubagents(state, pred) {
  return state.subagents.filter(pred).length;
}

function openDuplicate(state, fp, id) {
  return state.subagents.some(
    (row) =>
      row.fp === fp &&
      row.status === "started" &&
      row.id !== id &&
      !row.parallel,
  );
}

function recordDenial(state, payload, kind, detail) {
  state.denials.push({
    ts: nowMs(),
    generation_id: generationId(payload),
    kind,
    detail,
  });
}

function watchdogAllowed(state, payload, fields, id) {
  const gen = generationId(payload);
  const sameGen = countSubagents(
    state,
    (row) => isWatchdog(row.type) && row.generation_id === gen && row.id !== id,
  );
  const sameConv = countSubagents(
    state,
    (row) => isWatchdog(row.type) && row.id !== id,
  );
  if (sameGen >= LIMITS.watchdogPerGeneration) {
    return "loop-watchdog already ran this turn; do not spawn it again.";
  }
  if (sameConv >= LIMITS.watchdogPerConversation) {
    return "loop-watchdog already ran in this conversation; do not spawn it again.";
  }
  if (openDuplicate(state, fields.fp, id)) {
    return "loop-watchdog is already running; wait for it instead of launching another.";
  }
  return null;
}

function subagentDeniedReason(state, payload, fields, id) {
  if (fields.resume) return null;
  if (isWatchdog(fields.type)) return watchdogAllowed(state, payload, fields, id);

  const gen = generationId(payload);
  if (!fields.parallel && openDuplicate(state, fields.fp, id)) {
    return `subagent ${fields.type || "unknown"} with the same task is already running`;
  }

  const sameTaskGen = countSubagents(
    state,
    (row) => row.fp === fields.fp && row.generation_id === gen && row.id !== id,
  );
  if (sameTaskGen >= LIMITS.sameTaskPerGeneration) {
    return `repeated subagent ${fields.type || "unknown"} (same task, ${sameTaskGen} times this turn)`;
  }

  const sameTaskConv = countSubagents(
    state,
    (row) => row.fp === fields.fp && row.id !== id,
  );
  if (sameTaskConv >= LIMITS.sameTaskPerConversation) {
    return `repeated subagent ${fields.type || "unknown"} (same task, ${sameTaskConv} times this chat)`;
  }

  const failedSame = state.subagents.filter(
    (row) =>
      row.fp === fields.fp &&
      row.id !== id &&
      (row.status === "error" || row.status === "aborted"),
  );
  if (failedSame.length >= 1 && sameTaskConv >= 1) {
    return `subagent ${fields.type || "unknown"} already failed with the same task; do not relaunch it`;
  }

  const sameTypeGen = countSubagents(
    state,
    (row) =>
      row.type &&
      fields.type &&
      row.type === fields.type &&
      row.generation_id === gen &&
      row.id !== id,
  );
  if (fields.type && sameTypeGen >= LIMITS.sameTypePerGeneration) {
    return `too many ${fields.type} subagents this turn (${sameTypeGen})`;
  }

  const totalGen = countSubagents(
    state,
    (row) => row.generation_id === gen && row.id !== id,
  );
  if (totalGen >= LIMITS.totalPerGeneration) {
    return `too many subagents this turn (${totalGen})`;
  }
  return null;
}

function upsertSubagent(state, payload, fields, status) {
  const id = callId(payload) || `${fields.fp}:${nowMs()}`;
  let row = state.subagents.find((item) => item.id === id);
  if (!row) {
    row = {
      id,
      ts: nowMs(),
      generation_id: generationId(payload),
      type: fields.type,
      fp: fields.fp,
      task_preview: normalize(fields.task).slice(0, 160),
      parallel: fields.parallel,
      status,
    };
    state.subagents.push(row);
  } else {
    row.status = status;
    row.fp = fields.fp || row.fp;
    row.type = fields.type || row.type;
    if (fields.task) row.task_preview = normalize(fields.task).slice(0, 160);
  }
  return row;
}

function failureDeniedReason(state, payload, fp) {
  const gen = generationId(payload);
  const sameGen = state.failures.filter(
    (row) => row.fp === fp && row.generation_id === gen,
  ).length;
  const sameConv = state.failures.filter((row) => row.fp === fp).length;
  if (sameGen >= LIMITS.sameFailurePerGeneration) {
    return `same tool/command already failed ${sameGen} times this turn`;
  }
  if (sameConv >= LIMITS.sameFailurePerConversation) {
    return `same tool/command already failed ${sameConv} times this chat`;
  }
  return null;
}

function recordFailure(state, payload, fp, preview) {
  state.failures.push({
    ts: nowMs(),
    generation_id: generationId(payload),
    tool_name: toolName(payload) || (commandFrom(payload) ? "Shell" : ""),
    fp,
    preview: normalize(preview).slice(0, 240),
    failure_type: String(payload.failure_type || "error"),
  });
}

function looksFailedOutput(output) {
  const text = String(output || "");
  if (/\bexit_code["']?\s*[:=]\s*[1-9]\d*/i.test(text)) return true;
  if (/\bexit code[:\s]+[1-9]\d*/i.test(text)) return true;
  if (/traceback \(most recent call last\)/i.test(text)) return true;
  if (/\bcommand failed\b/i.test(text)) return true;
  return false;
}

function loopNotice(state) {
  const recentDenials = state.denials.filter((row) => nowMs() - row.ts < 15 * 60 * 1000);
  const recentFails = state.failures.filter((row) => nowMs() - row.ts < 15 * 60 * 1000);
  if (!recentDenials.length && recentFails.length < 2) return "";
  const lastNotice = state.notices[state.notices.length - 1];
  if (lastNotice && nowMs() - lastNotice.ts < 30 * 1000) return "";
  const detail =
    recentDenials[recentDenials.length - 1]?.detail ||
    recentFails[recentFails.length - 1]?.preview ||
    "repeated tool/subagent activity";
  state.notices.push({ ts: nowMs(), detail });
  return [
    "Loop guard: stop repeating the same subagent or the same failing command.",
    `Last signal: ${detail}.`,
    "Do not retry it. You may invoke the loop-watchdog subagent once this turn, then change approach.",
  ].join(" ");
}

function onSessionStart(payload) {
  mutate(payload, (state) => {
    state.conversation_id = conversationId(payload);
  });
  return {
    env: {
      LOOP_GUARD: "1",
    },
    additional_context: [
      "Loop guard is active in this repo.",
      "Do not spawn the same subagent twice, and do not retry a command/tool that already failed.",
      "If blocked, read .cursor/hooks/state/ and you may invoke loop-watchdog once this turn — never in a loop.",
    ].join(" "),
  };
}

function onSubagentStart(payload) {
  const fields = taskFields(payload);
  return mutate(payload, (state) => {
    const id = callId(payload);
    const reason = subagentDeniedReason(state, payload, fields, id);
    if (reason) {
      recordDenial(state, payload, "subagent", reason);
      appendEvent(payload, { kind: "deny_subagent", reason, type: fields.type, fp: fields.fp });
      return deny(
        `Хук зупинив повторний запуск агента (${fields.type || "subagent"}): ${reason}.`,
        [
          `Loop guard denied subagent '${fields.type || "unknown"}': ${reason}.`,
          "Do not relaunch that agent or rephrase the same task.",
          "Change approach, or invoke loop-watchdog once, then stop.",
        ].join(" "),
      );
    }
    upsertSubagent(state, payload, fields, "started");
    appendEvent(payload, { kind: "subagent_start", type: fields.type, fp: fields.fp });
    return allow();
  });
}

function onSubagentStop(payload) {
  const fields = taskFields(payload);
  const status = String(payload.status || "completed");
  mutate(payload, (state) => {
    upsertSubagent(state, payload, fields, status);
    appendEvent(payload, {
      kind: "subagent_stop",
      type: fields.type,
      fp: fields.fp,
      status,
    });
  });
  // Never continue a subagent loop from this hook.
  return {};
}

function onPreToolUse(payload) {
  const name = toolName(payload);
  const input = toolInput(payload);
  if (input.resume) return allow();

  if (isTaskTool(name)) {
    const fields = taskFields(payload);
    return mutate(payload, (state) => {
      const id = callId(payload);
      const reason = subagentDeniedReason(state, payload, fields, id);
      if (reason) {
        recordDenial(state, payload, "task", reason);
        appendEvent(payload, { kind: "deny_task", reason, type: fields.type, fp: fields.fp });
        return deny(
          `Хук зупинив повторний Task/агент (${fields.type || "Task"}): ${reason}.`,
          [
            `Loop guard denied Task '${fields.type || "unknown"}': ${reason}.`,
            "Do not spawn that subagent again.",
            "Invoke loop-watchdog at most once, then take a different action.",
          ].join(" "),
        );
      }
      upsertSubagent(state, payload, fields, "started");
      return allow();
    });
  }

  const fp = inputFingerprint(payload);
  return mutate(payload, (state) => {
    const reason = failureDeniedReason(state, payload, fp);
    if (reason) {
      recordDenial(state, payload, "tool", reason);
      appendEvent(payload, { kind: "deny_tool", reason, tool: name, fp });
      return deny(
        `Хук зупинив повторне виконання помилки (${name || "tool"}): ${reason}.`,
        [
          `Loop guard denied repeated failing tool '${name || "unknown"}': ${reason}.`,
          "Do not retry the same input. Change the command, inspect the last error, or invoke loop-watchdog once.",
        ].join(" "),
      );
    }
    return allow();
  });
}

function onBeforeShell(payload) {
  const fp = inputFingerprint({ ...payload, tool_name: "Shell" });
  return mutate(payload, (state) => {
    const reason = failureDeniedReason(state, payload, fp);
    if (reason) {
      recordDenial(state, payload, "shell", reason);
      appendEvent(payload, {
        kind: "deny_shell",
        reason,
        fp,
        command: normalize(commandFrom(payload)).slice(0, 160),
      });
      return deny(
        `Хук зупинив повторну shell-команду: ${reason}.`,
        [
          `Loop guard denied a repeated failing shell command: ${reason}.`,
          "Do not run that command again. Fix the cause or invoke loop-watchdog once.",
        ].join(" "),
      );
    }
    return allow();
  });
}

function onBeforeMcp(payload) {
  const fp = inputFingerprint(payload);
  return mutate(payload, (state) => {
    const reason = failureDeniedReason(state, payload, fp);
    if (reason) {
      recordDenial(state, payload, "mcp", reason);
      appendEvent(payload, {
        kind: "deny_mcp",
        reason,
        fp,
        tool: payload.tool_name,
      });
      return deny(
        `Хук зупинив повторний MCP-виклик: ${reason}.`,
        [
          `Loop guard denied a repeated failing MCP call: ${reason}.`,
          "Do not retry the same MCP input. Invoke loop-watchdog once if you need a diagnosis.",
        ].join(" "),
      );
    }
    return allow();
  });
}

function onFailure(payload) {
  const fp = inputFingerprint(payload);
  mutate(payload, (state) => {
    recordFailure(
      state,
      payload,
      fp,
      payload.error_message || commandFrom(payload) || toolName(payload),
    );
    appendEvent(payload, {
      kind: "failure",
      fp,
      tool: toolName(payload),
      error: String(payload.error_message || "").slice(0, 240),
    });
  });
  return {};
}

function onAfterShell(payload) {
  if (!looksFailedOutput(payload.output)) return {};
  const fp = inputFingerprint({ ...payload, tool_name: "Shell" });
  mutate(payload, (state) => {
    recordFailure(state, payload, fp, String(payload.output || "").slice(-240));
    appendEvent(payload, { kind: "shell_output_failure", fp });
  });
  return {};
}

function onPostToolUse(payload) {
  return mutate(payload, (state) => {
    const notice = loopNotice(state);
    if (!notice) return {};
    appendEvent(payload, { kind: "notice", detail: notice });
    return { additional_context: notice };
  });
}

function handle(payload) {
  const name = eventName(payload);
  switch (name) {
    case "sessionStart":
      return onSessionStart(payload);
    case "sessionEnd":
      return {};
    case "preToolUse":
      return onPreToolUse(payload);
    case "postToolUse":
      return onPostToolUse(payload);
    case "postToolUseFailure":
      return onFailure(payload);
    case "beforeShellExecution":
      return onBeforeShell(payload);
    case "afterShellExecution":
      return onAfterShell(payload);
    case "beforeMCPExecution":
      return onBeforeMcp(payload);
    case "subagentStart":
      return onSubagentStart(payload);
    case "subagentStop":
      return onSubagentStop(payload);
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
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "loop-guard-"));
  process.env.LOOP_GUARD_STATE_DIR = dir;
  const conv = { conversation_id: "c1", generation_id: "g1", workspace_roots: [dir] };

  const start1 = handle({
    ...conv,
    hook_event_name: "subagentStart",
    subagent_type: "fpvscan-web-debugger",
    task: "debug rotator slider",
    tool_call_id: "t1",
  });
  assert(start1.permission === "allow", "first subagent should allow");

  const start2 = handle({
    ...conv,
    hook_event_name: "subagentStart",
    subagent_type: "fpvscan-web-debugger",
    task: "debug rotator slider",
    tool_call_id: "t2",
  });
  assert(start2.permission === "deny", "duplicate running subagent should deny");

  handle({
    ...conv,
    hook_event_name: "subagentStop",
    subagent_type: "fpvscan-web-debugger",
    task: "debug rotator slider",
    tool_call_id: "t1",
    status: "error",
  });

  const afterFail = handle({
    ...conv,
    hook_event_name: "subagentStart",
    subagent_type: "fpvscan-web-debugger",
    task: "debug rotator slider",
    tool_call_id: "t3",
  });
  assert(afterFail.permission === "deny", "failed subagent should not relaunch");

  const watchdog1 = handle({
    ...conv,
    hook_event_name: "preToolUse",
    tool_name: "Task",
    tool_use_id: "w1",
    tool_input: {
      subagent_type: "loop-watchdog",
      prompt: "inspect loop state",
      description: "watchdog",
    },
  });
  assert(watchdog1.permission === "allow", "watchdog should allow once");

  const watchdog2 = handle({
    ...conv,
    hook_event_name: "preToolUse",
    tool_name: "Task",
    tool_use_id: "w2",
    tool_input: {
      subagent_type: "loop-watchdog",
      prompt: "inspect loop state again",
      description: "watchdog",
    },
  });
  assert(watchdog2.permission === "deny", "second watchdog this turn should deny");

  const failPayload = {
    ...conv,
    hook_event_name: "postToolUseFailure",
    tool_name: "Shell",
    tool_input: { command: "wget -O ~/x.py http://example.invalid/x.py" },
    error_message: "404",
  };
  handle(failPayload);
  handle(failPayload);

  const thirdShell = handle({
    ...conv,
    hook_event_name: "beforeShellExecution",
    command: "wget -O ~/x.py http://example.invalid/x.py",
  });
  assert(thirdShell.permission === "deny", "third identical failing shell should deny");

  const otherShell = handle({
    ...conv,
    hook_event_name: "beforeShellExecution",
    command: "ls",
  });
  assert(otherShell.permission === "allow", "different shell command should allow");

  handle({
    ...conv,
    hook_event_name: "sessionStart",
    session_id: "c1",
  });

  const notice = handle({
    ...conv,
    hook_event_name: "postToolUse",
    tool_name: "Read",
    tool_input: { path: "README.md" },
    tool_output: "{}",
  });
  assert(typeof notice.additional_context === "string", "should inject loop notice");

  fs.rmSync(dir, { recursive: true, force: true });
  process.stderr.write("stop_repeat self-test passed\n");
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
      process.stderr.write(`stop_repeat: invalid JSON on stdin: ${err.message}\n`);
      process.stdout.write("{}\n");
      process.exit(0);
      return;
    }
  }
  let out = {};
  try {
    out = handle(payload) || {};
  } catch (err) {
    process.stderr.write(`stop_repeat: ${err.stack || err.message}\n`);
    out = {};
  }
  process.stdout.write(`${JSON.stringify(out)}\n`);
}

if (require.main === module) {
  main();
}

module.exports = { handle, LIMITS, WATCHDOG_TYPE };
