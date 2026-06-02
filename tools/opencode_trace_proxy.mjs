#!/usr/bin/env bun
import http from "node:http";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

const DEFAULT_LISTEN = 9091;
const DEFAULT_UPSTREAM = "http://127.0.0.1:9090/v1";
const DEFAULT_TRACE_DIR = "/tmp/zinc-opencode-traces";

const numericArgKeys = new Set([
  "offset",
  "limit",
  "line",
  "lineNumber",
  "start",
  "end",
  "count",
  "depth",
  "timeout",
  "timeout_secs",
]);

export function parseArgs(argv = process.argv.slice(2)) {
  const opts = {
    command: "proxy",
    listen: DEFAULT_LISTEN,
    upstream: DEFAULT_UPSTREAM,
    traceDir: DEFAULT_TRACE_DIR,
    forceEnableThinking: null,
    forceToolChoice: null,
    maxTokensCap: null,
    model: null,
    temperature: null,
    topP: null,
    injectPathGuard: false,
    repairToolPaths: false,
    traceFile: null,
  };

  const args = [...argv];
  if (args[0] && !args[0].startsWith("--")) {
    opts.command = args.shift();
  }

  const readValue = (i) => {
    if (i + 1 >= args.length || args[i + 1].startsWith("--")) return [true, i];
    return [args[i + 1], i + 1];
  };

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === "--listen") {
      const [v, ni] = readValue(i);
      opts.listen = Number(v);
      i = ni;
    } else if (arg === "--upstream") {
      const [v, ni] = readValue(i);
      opts.upstream = String(v);
      i = ni;
    } else if (arg === "--trace-dir") {
      const [v, ni] = readValue(i);
      opts.traceDir = String(v);
      i = ni;
    } else if (arg === "--trace-file") {
      const [v, ni] = readValue(i);
      opts.traceFile = String(v);
      i = ni;
    } else if (arg === "--force-enable-thinking") {
      const [v, ni] = readValue(i);
      opts.forceEnableThinking = parseBool(v);
      i = ni;
    } else if (arg === "--force-tool-choice") {
      const [v, ni] = readValue(i);
      opts.forceToolChoice = String(v);
      i = ni;
    } else if (arg === "--max-tokens-cap") {
      const [v, ni] = readValue(i);
      opts.maxTokensCap = Number(v);
      i = ni;
    } else if (arg === "--model") {
      const [v, ni] = readValue(i);
      opts.model = String(v);
      i = ni;
    } else if (arg === "--temperature") {
      const [v, ni] = readValue(i);
      opts.temperature = Number(v);
      i = ni;
    } else if (arg === "--top-p") {
      const [v, ni] = readValue(i);
      opts.topP = Number(v);
      i = ni;
    } else if (arg === "--inject-path-guard") {
      opts.injectPathGuard = true;
    } else if (arg === "--repair-tool-paths") {
      opts.repairToolPaths = true;
    }
  }

  return opts;
}

function parseBool(v) {
  if (typeof v === "boolean") return v;
  const s = String(v).toLowerCase();
  return s === "1" || s === "true" || s === "yes" || s === "on";
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

export function stripToolBoundaryTail(value) {
  if (typeof value !== "string") return value;
  return value
    .replace(/\s*<\/\/parameter>[\s\S]*$/g, "")
    .replace(/\s*<\/parameter>[\s\S]*$/g, "")
    .replace(/\s*<parameter\b[\s\S]*$/g, "")
    .replace(/\s*<\|[^]*$/g, "")
    .trimEnd();
}

function cleanPathCandidate(raw) {
  let s = stripToolBoundaryTail(raw);
  s = s.replace(/<\/?(?:path|file|parameter|entry)>/g, "");
  s = s.replace(/[),;\]]+$/g, "");
  if ((s.endsWith(".") || s.endsWith(":")) && !/\.[A-Za-z0-9]+[.:]$/.test(s)) {
    s = s.slice(0, -1);
  }
  return s;
}

function collectStrings(value, out = []) {
  if (typeof value === "string") {
    out.push(value);
  } else if (Array.isArray(value)) {
    for (const v of value) collectStrings(v, out);
  } else if (value && typeof value === "object") {
    for (const v of Object.values(value)) collectStrings(v, out);
  }
  return out;
}

const absolutePathRe = /(?:\/private\/tmp|\/tmp|\/Users|\/root|\/Volumes|\/var)\/[^\s"'<>`{}]+/g;

export function extractKnownFilePaths(body) {
  const paths = new Set();
  const strings = collectStrings(body);
  for (const text of strings) {
    for (const match of text.matchAll(absolutePathRe)) {
      const candidate = cleanPathCandidate(match[0]);
      if (candidate && looksLikeFilePath(candidate)) paths.add(candidate);
    }
  }

  const combined = strings.join("\n");
  const dirRe = /<path>([^<]+)<\/path>\s*<type>directory<\/type>\s*<entries>([\s\S]*?)<\/entries>/g;
  for (const match of combined.matchAll(dirRe)) {
    const dir = cleanPathCandidate(match[1]);
    const entries = match[2]
      .split(/\r?\n|,/)
      .map((s) => s.trim())
      .filter(Boolean);
    for (const entry of entries) {
      if (looksLikeFilePath(entry)) paths.add(path.posix.join(dir, entry));
    }
  }

  return [...paths].sort();
}

function looksLikeFilePath(value) {
  return /\/[^/]+\.[A-Za-z0-9_+-]+$/.test(value) || /^[^/]+\.[A-Za-z0-9_+-]+$/.test(value);
}

function looksLikeDirectory(value) {
  return typeof value === "string" && value.length > 0 && !looksLikeFilePath(value);
}

export function extractWorkingDirectory(body) {
  const strings = collectStrings(body);
  for (const text of strings) {
    const m = text.match(/(?:Working directory|Workspace root folder):\s*([^\n\r]+)/i);
    if (m) return cleanPathCandidate(m[1].trim());
  }

  const known = extractKnownFilePaths(body);
  const src = known.find((p) => p.includes("/src/")) ?? known[0];
  if (src) {
    const marker = src.includes("/src/") ? "/src/" : src.includes("/test/") ? "/test/" : null;
    if (marker) return src.slice(0, src.indexOf(marker));
    return path.posix.dirname(src);
  }
  return "";
}

export function preferredEditablePath(body) {
  const known = extractKnownFilePaths(body);
  return (
    known.find((p) => p.includes("/src/") && !p.includes("/test/")) ??
    known.find((p) => !p.includes("/test/") && !p.endsWith("/package.json")) ??
    known[0] ??
    ""
  );
}

export function deriveSessionId(body) {
  const workingDir = extractWorkingDirectory(body);
  const firstUser = (body.messages ?? []).find((m) => m.role === "user")?.content ?? "";
  const basis = `${workingDir}\n${typeof firstUser === "string" ? firstUser : JSON.stringify(firstUser)}`;
  return `oc_${createHash("sha256").update(basis).digest("hex").slice(0, 24)}`;
}

export function injectSessionId(body) {
  const next = cloneJson(body);
  if (!next.session_id) next.session_id = deriveSessionId(next);
  return next;
}

export function buildPathGuardMessage(body) {
  const editable = preferredEditablePath(body);
  const known = extractKnownFilePaths(body);
  const lines = [
    "Tool path guard: Use exact absolute paths from recent tool results.",
  ];
  if (editable) {
    lines.push(`For edit/write, use only ${editable}; package and test files are read-only unless the user explicitly asks to change tests.`);
  }
  if (known.length > 0) {
    lines.push(`Known file paths: ${known.slice(0, 12).join(", ")}`);
  }
  return lines.join("\n");
}

export function buildCodingContinuationGuardMessage(body) {
  const editable = preferredEditablePath(body);
  const source = editable ? ` Edit/write ${editable} for source fixes.` : "";
  return [
    "OpenCode continuation guard: Continue the coding task with tool calls.",
    "Do not summarize instead of acting when tests are still failing.",
    "Never call edit or write on package.json or files under /test/ unless the user explicitly asks to change tests.",
    `${source} Run the project tests after source edits.`,
  ].join("\n");
}

function hasGuard(messages, prefix) {
  return messages.some((m) => m.role === "system" && typeof m.content === "string" && m.content.trimStart().startsWith(prefix));
}

export function applyRequestOverrides(body, opts = {}) {
  const next = injectSessionId(body);
  if (opts.model) next.model = opts.model;
  if (opts.forceEnableThinking !== null && opts.forceEnableThinking !== undefined) {
    next.enable_thinking = opts.forceEnableThinking;
  }
  if (opts.forceToolChoice) next.tool_choice = opts.forceToolChoice;
  if (Number.isFinite(opts.maxTokensCap)) {
    const current = Number.isFinite(next.max_tokens) ? Number(next.max_tokens) : opts.maxTokensCap;
    next.max_tokens = Math.min(current, opts.maxTokensCap);
  }
  if (Number.isFinite(opts.temperature)) next.temperature = opts.temperature;
  if (Number.isFinite(opts.topP)) next.top_p = opts.topP;

  if (opts.injectPathGuard) {
    next.messages = Array.isArray(next.messages) ? [...next.messages] : [];
    if (!hasGuard(next.messages, "Tool path guard:")) {
      next.messages.push({ role: "system", content: buildPathGuardMessage(next) });
    }
    if (!hasGuard(next.messages, "OpenCode continuation guard:")) {
      next.messages.push({ role: "system", content: buildCodingContinuationGuardMessage(next) });
    }
  }

  return next;
}

function recursivelyNormalizeStrings(value) {
  if (typeof value === "string") return stripToolBoundaryTail(value);
  if (Array.isArray(value)) return value.map(recursivelyNormalizeStrings);
  if (value && typeof value === "object") {
    for (const [k, v] of Object.entries(value)) value[k] = recursivelyNormalizeStrings(v);
  }
  return value;
}

function coerceNumericArgs(value) {
  if (!value || typeof value !== "object") return;
  for (const [k, v] of Object.entries(value)) {
    if (numericArgKeys.has(k) && typeof v === "string" && /^-?\d+$/.test(v.trim())) {
      value[k] = Number(v);
    } else if (v && typeof v === "object") {
      coerceNumericArgs(v);
    }
  }
}

function nearestKnownPath(raw, body, toolName) {
  const value = cleanPathCandidate(raw);
  const known = extractKnownFilePaths(body);
  const editable = preferredEditablePath(body);
  if (!value || value === "/" || value === "/private" || value === "/private/") {
    return toolName && /edit|write/i.test(toolName) && editable ? editable : known[0] ?? value;
  }

  if (known.includes(value)) return value;
  if (value.endsWith(".")) {
    const prefix = value.slice(0, -1);
    const byPrefix = known.find((p) => p.startsWith(prefix));
    if (byPrefix) return byPrefix;
  }

  const base = path.posix.basename(value).replace(/[.:]+$/g, "");
  if (base) {
    const byBase = known.find((p) => path.posix.basename(p) === base || path.posix.basename(p).startsWith(base));
    if (byBase) return byBase;
  }

  const didYouMean = collectStrings(body).join("\n").match(/Did you mean\s+([^?\n]+)\?/i);
  if (didYouMean) {
    const hinted = cleanPathCandidate(didYouMean[1].trim());
    const matched = known.find((p) => p.endsWith(hinted) || p.includes(hinted));
    if (matched) return matched;
  }

  if (/edit|write/i.test(toolName ?? "") && editable) return editable;
  return value;
}

export function repairArgumentsJson(argumentsText, requestBody, toolName = "") {
  let args;
  try {
    args = JSON.parse(argumentsText || "{}");
  } catch {
    const stripped = stripToolBoundaryTail(argumentsText || "{}");
    try {
      args = JSON.parse(stripped);
    } catch {
      return { text: stripped, changed: stripped !== argumentsText };
    }
  }

  const original = JSON.stringify(args);
  args = recursivelyNormalizeStrings(args);
  coerceNumericArgs(args);

  const workingDir = extractWorkingDirectory(requestBody);
  for (const [key, value] of Object.entries(args)) {
    if (typeof value !== "string") continue;
    const lower = key.toLowerCase();
    if (toolName.toLowerCase() === "glob" && lower === "path" && looksLikeFilePath(value)) {
      args[key] = workingDir || path.posix.dirname(value);
      continue;
    }
    if (lower === "filepath" || lower === "file_path" || lower === "path") {
      if (toolName.toLowerCase() === "glob" && looksLikeDirectory(value)) continue;
      args[key] = nearestKnownPath(value, requestBody, toolName);
    }
  }

  const text = JSON.stringify(args);
  return { text, changed: text !== original };
}

function repairToolCall(call, requestBody) {
  if (!call?.function || typeof call.function.arguments !== "string") return false;
  const name = call.function.name ?? call.name ?? "";
  const repaired = repairArgumentsJson(call.function.arguments, requestBody, name);
  call.function.arguments = repaired.text;
  return repaired.changed;
}

export function repairOpenAiToolCalls(payload, requestBody) {
  let changed = false;
  for (const choice of payload.choices ?? []) {
    for (const call of choice.delta?.tool_calls ?? []) {
      changed = repairToolCall(call, requestBody) || changed;
    }
    for (const call of choice.message?.tool_calls ?? []) {
      changed = repairToolCall(call, requestBody) || changed;
    }
  }
  return changed;
}

export function repairSseEvent(event, requestBody) {
  const lines = event.split(/\n/);
  let changed = false;
  const out = lines.map((line) => {
    if (!line.startsWith("data: ")) return line;
    const data = line.slice("data: ".length);
    if (!data.trim() || data.trim() === "[DONE]") return line;
    try {
      const payload = JSON.parse(data);
      if (repairOpenAiToolCalls(payload, requestBody)) changed = true;
      return `data: ${JSON.stringify(payload)}`;
    } catch {
      return line;
    }
  });
  return { event: out.join("\n"), changed };
}

function upstreamUrlFor(incomingUrl, upstream) {
  const base = new URL(upstream);
  const incoming = new URL(incomingUrl, "http://proxy.local");
  const suffix = incoming.pathname.startsWith("/v1/") ? incoming.pathname.slice(3) : incoming.pathname;
  const joinedBase = base.pathname.replace(/\/$/, "");
  base.pathname = `${joinedBase}${suffix}`;
  base.search = incoming.search;
  return base;
}

async function readRequestBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks).toString("utf8");
}

function responseHeaders(upstreamResponse, rewriteBody) {
  const headers = {};
  upstreamResponse.headers.forEach((value, key) => {
    if (key === "content-length" || key === "connection") return;
    headers[key] = value;
  });
  if (rewriteBody) delete headers["content-length"];
  return headers;
}

async function writeTrace(traceDir, trace) {
  await fs.mkdir(traceDir, { recursive: true });
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const name = `${ts}-${trace.request_summary?.model ?? "unknown"}-${trace.request_summary?.stream ? "stream" : "json"}.json`;
  const file = path.join(traceDir, name);
  await fs.writeFile(file, `${JSON.stringify(trace, null, 2)}\n`);
  return file;
}

async function handleProxyRequest(req, res, opts) {
  const rawBody = await readRequestBody(req);
  let requestBody = rawBody;
  let parsedBody = null;
  if (rawBody.trim().startsWith("{")) {
    parsedBody = applyRequestOverrides(JSON.parse(rawBody), opts);
    requestBody = JSON.stringify(parsedBody);
  }

  const upstreamUrl = upstreamUrlFor(req.url, opts.upstream);
  const trace = {
    created_at: new Date().toISOString(),
    method: req.method,
    path: new URL(req.url, "http://proxy.local").pathname,
    upstream: upstreamUrl.toString(),
    request_summary: {
      model: parsedBody?.model,
      stream: parsedBody?.stream,
      max_tokens: parsedBody?.max_tokens,
      enable_thinking: parsedBody?.enable_thinking,
      session_id: parsedBody?.session_id,
    },
    request_body: parsedBody,
    response_metrics: { bytes: 0, repaired_events: 0 },
    response_preview: "",
  };

  try {
    const upstreamResponse = await fetch(upstreamUrl, {
      method: req.method,
      headers: {
        "content-type": req.headers["content-type"] ?? "application/json",
      },
      body: req.method === "GET" || req.method === "HEAD" ? undefined : requestBody,
    });

    const contentType = upstreamResponse.headers.get("content-type") ?? "";
    res.writeHead(upstreamResponse.status, responseHeaders(upstreamResponse, true));

    if (contentType.includes("text/event-stream") && upstreamResponse.body) {
      const decoder = new TextDecoder();
      let pending = "";
      for await (const chunk of upstreamResponse.body) {
        pending += decoder.decode(chunk, { stream: true });
        let idx;
        while ((idx = pending.indexOf("\n\n")) >= 0) {
          const event = pending.slice(0, idx + 2);
          pending = pending.slice(idx + 2);
          const repaired = opts.repairToolPaths && parsedBody ? repairSseEvent(event, parsedBody) : { event, changed: false };
          if (repaired.changed) trace.response_metrics.repaired_events += 1;
          trace.response_metrics.bytes += Buffer.byteLength(repaired.event);
          if (trace.response_preview.length < 4096) trace.response_preview += repaired.event.slice(0, 4096 - trace.response_preview.length);
          res.write(repaired.event);
        }
      }
      if (pending) res.write(pending);
      res.end();
    } else {
      let text = await upstreamResponse.text();
      if (opts.repairToolPaths && parsedBody && text.trim().startsWith("{")) {
        const payload = JSON.parse(text);
        if (repairOpenAiToolCalls(payload, parsedBody)) {
          trace.response_metrics.repaired_events += 1;
          text = JSON.stringify(payload);
        }
      }
      trace.response_metrics.bytes = Buffer.byteLength(text);
      trace.response_preview = text.slice(0, 4096);
      res.end(text);
    }
  } catch (err) {
    trace.error = { message: err?.message ?? String(err), stack: err?.stack };
    res.writeHead(502, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: { message: trace.error.message, type: "proxy_error" } }));
  } finally {
    try {
      trace.trace_file = await writeTrace(opts.traceDir, trace);
    } catch (err) {
      console.error("trace write failed:", err);
    }
  }
}

export function startProxy(opts) {
  const server = http.createServer((req, res) => {
    handleProxyRequest(req, res, opts).catch((err) => {
      res.writeHead(500, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: err?.message ?? String(err), type: "proxy_error" } }));
    });
  });
  server.listen(opts.listen, "127.0.0.1", () => {
    console.log(`opencode trace proxy listening on http://127.0.0.1:${opts.listen}/v1 -> ${opts.upstream}`);
  });
  return server;
}

async function replay(opts) {
  if (!opts.traceFile) throw new Error("replay requires --trace-file");
  const trace = JSON.parse(await fs.readFile(opts.traceFile, "utf8"));
  const url = upstreamUrlFor(trace.path ?? "/v1/chat/completions", opts.upstream);
  const response = await fetch(url, {
    method: trace.method ?? "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(trace.request_body),
  });
  process.stdout.write(await response.text());
}

async function main() {
  const opts = parseArgs();
  if (opts.command === "proxy") {
    startProxy(opts);
  } else if (opts.command === "replay" || opts.command === "compare") {
    await replay(opts);
  } else {
    throw new Error(`unknown command: ${opts.command}`);
  }
}

if (import.meta.main) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
