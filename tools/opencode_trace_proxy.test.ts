import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import {
  applyRequestOverrides,
  buildCodingContinuationGuardMessage,
  buildPathGuardMessage,
  editableKnownFilePaths,
  editableRootDirs,
  deriveSessionId,
  extractKnownFilePaths,
  extractRequestedFilePaths,
  extractWorkingDirectory,
  injectSessionId,
  isBenignPartialStreamClose,
  isOpenCodeSuccessfulTestFinalRequest,
  isOpenCodeTitleRequest,
  parseArgs,
  parseXmlToolCallContent,
  preferredEditablePath,
  repairArgumentsJson,
  repairEditOldStringArgs,
  repairOpenAiToolCalls,
  repairSseText,
  repairSseEvent,
  repairToolRequiredSseText,
  sendProxyErrorResponse,
  shouldSuppressAssistantContent,
  shouldRequireOpenCodeToolChoice,
  stripToolBoundaryTail,
  syntheticCompletionForRequest,
  syntheticDiscoveredSourceReadCalls,
  syntheticHeuristicSourceWriteCalls,
  syntheticPostEditTestCall,
  syntheticPrefetchReadCalls,
  syntheticResponseText,
  syntheticToolCallResponseText,
  titleForOpenCodeRequest,
} from "./opencode_trace_proxy.mjs";

function withTempFile(content, fn) {
  const dir = mkdtempSync(join(tmpdir(), "zinc-opencode-proxy-"));
  const file = join(dir, "src.mjs");
  writeFileSync(file, content);
  try {
    return fn(file);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function withTempProject(files, fn) {
  const dir = mkdtempSync(join(tmpdir(), "zinc-opencode-project-"));
  for (const [name, content] of Object.entries(files)) {
    const file = join(dir, name);
    const parent = file.slice(0, file.lastIndexOf("/"));
    if (parent) {
      mkdirSync(parent, { recursive: true });
    }
    writeFileSync(file, content);
  }
  try {
    return fn(dir);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function lineNumbered(content) {
  return String(content)
    .replace(/\n$/, "")
    .split(/\n/)
    .map((line, index) => `${index + 1}: ${line}`)
    .join("\n");
}

function fileToolMessage(filePath, content) {
  return {
    role: "tool",
    content: `<path>${filePath}</path>\n<type>file</type>\n<content>\n${lineNumbered(content)}\n\n(End of file - total ${String(content).replace(/\n$/, "").split(/\n/).length} lines)\n</content>`,
  };
}

const body = {
  model: "zinc/qwen",
  messages: [
    {
      role: "system",
      content: "Working directory: /private/tmp/zinc-opencode-smoke4",
    },
    {
      role: "user",
      content:
        "Fix the tests. Read /private/tmp/zinc-opencode-smoke4/src/cart.mjs and /private/tmp/zinc-opencode-smoke4/test/cart.test.mjs.",
    },
    {
      role: "tool",
      content:
        "<path>/private/tmp/zinc-opencode-smoke4/test</path><type>directory</type><entries>cart.test.mjs\nhelpers.txt</entries>",
    },
  ],
};

const multiFileBody = {
  model: "zinc/qwen",
  messages: [
    {
      role: "system",
      content: "Working directory: /private/tmp/zinc-opencode-smoke6",
    },
    {
      role: "tool",
      content:
        "<path>/private/tmp/zinc-opencode-smoke6/src</path><type>directory</type><entries>cart.mjs\npricing.mjs</entries>\n<path>/private/tmp/zinc-opencode-smoke6/test</path><type>directory</type><entries>cart.test.mjs</entries>",
    },
  ],
};

const titleBody = {
  model: "zinc/qwen",
  stream: true,
  messages: [
    {
      role: "system",
      content: "You are a title generator. You output ONLY a thread title.",
    },
    {
      role: "user",
      content: "Generate a title for this conversation:\n",
    },
    {
      role: "user",
      content:
        '"Fix the failing tests in this rate limiter project. Read /private/tmp/x/src/rate_limiter.mjs and run npm test."',
    },
  ],
};

const successfulFinalBody = {
  model: "zinc/qwen",
  stream: true,
  messages: [
    {
      role: "user",
      content: "Fix the failing tests and stop after all tests pass.",
    },
    {
      role: "assistant",
      content: "",
    },
    {
      role: "tool",
      content: "\n> test\n> bun test\n\n 5 pass\n 0 fail\n 21 expect() calls\n",
    },
  ],
};

describe("parseArgs", () => {
  test("defaults to local proxy settings", () => {
    const opts = parseArgs([]);
    expect(opts.command).toBe("proxy");
    expect(opts.listen).toBe(9091);
    expect(opts.upstream).toBe("http://127.0.0.1:9090/v1");
  });

  test("parses request override flags", () => {
    const opts = parseArgs([
      "proxy",
      "--listen",
      "9191",
      "--upstream",
      "http://127.0.0.1:8080/v1",
      "--force-enable-thinking",
      "false",
      "--force-tool-choice",
      "auto",
      "--max-tokens-cap",
      "768",
      "--model",
      "qwen36",
      "--temperature",
      "0",
      "--top-p",
      "1",
      "--inject-path-guard",
      "--repair-tool-paths",
    ]);
    expect(opts.listen).toBe(9191);
    expect(opts.forceEnableThinking).toBe(false);
    expect(opts.forceToolChoice).toBe("auto");
    expect(opts.maxTokensCap).toBe(768);
    expect(opts.model).toBe("qwen36");
    expect(opts.temperature).toBe(0);
    expect(opts.topP).toBe(1);
    expect(opts.injectPathGuard).toBe(true);
    expect(opts.repairToolPaths).toBe(true);
  });
});

describe("request shaping", () => {
  test("injectSessionId derives a stable OpenCode session id", () => {
    const a = injectSessionId(body);
    const b = injectSessionId(body);
    expect(a.session_id).toMatch(/^oc_[0-9a-f]{24}$/);
    expect(a.session_id).toBe(b.session_id);
    expect(deriveSessionId(body)).toBe(a.session_id);
  });

  test("applyRequestOverrides forces model, thinking, sampling, token cap, and guards", () => {
    const out = applyRequestOverrides(
      { ...body, max_tokens: 2048, stream: true },
      {
        model: "qwen36-35b-a3b-q4k-xl",
        forceEnableThinking: false,
        forceToolChoice: "auto",
        maxTokensCap: 768,
        temperature: 0,
        topP: 1,
        injectPathGuard: true,
      },
    );
    expect(out.model).toBe("qwen36-35b-a3b-q4k-xl");
    expect(out.enable_thinking).toBe(false);
    expect(out.chat_template_kwargs?.enable_thinking).toBe(false);
    expect(out.tool_choice).toBe("auto");
    expect(out.max_tokens).toBe(768);
    expect(out.temperature).toBe(0);
    expect(out.top_p).toBe(1);
    expect(out.messages[0]?.role).toBe("system");
    expect(out.messages[0]?.content).toStartWith("Tool path guard:");
    expect(out.messages[0]?.content).toContain("OpenCode continuation guard:");
    expect(out.messages[0]?.content).toContain("Working directory: /private/tmp/zinc-opencode-smoke4");
    expect(out.messages.at(-1)?.role).toBe("tool");
  });

  test("does not inject coding guards into OpenCode title requests", () => {
    const out = applyRequestOverrides(titleBody, {
      model: "qwen36-35b-a3b-q4k-xl",
      forceEnableThinking: false,
      maxTokensCap: 768,
      injectPathGuard: true,
    });

    expect(out.model).toBe("qwen36-35b-a3b-q4k-xl");
    expect(out.enable_thinking).toBe(false);
    expect(out.max_tokens).toBe(768);
    expect(out.messages.some((m) => String(m.content).startsWith("Tool path guard:"))).toBe(false);
    expect(out.messages.some((m) => String(m.content).startsWith("OpenCode continuation guard:"))).toBe(false);
  });

  test("requires tool calls by default for OpenCode coding turns", () => {
    const out = applyRequestOverrides(
      {
        model: "zinc/qwen",
        stream: true,
        tools: [{ type: "function", function: { name: "read" } }],
        messages: [
          { role: "system", content: "Working directory: /tmp/project" },
          { role: "user", content: "Fix failing tests and stop after all tests pass." },
        ],
      },
      { injectPathGuard: true },
    );

    expect(shouldRequireOpenCodeToolChoice(out)).toBe(true);
    expect(out.tool_choice).toBe("required");
  });

  test("merges injected guards into the first system message without duplicating trailing guards", () => {
    const out = applyRequestOverrides(
      {
        model: "zinc/qwen",
        stream: true,
        tools: [{ type: "function", function: { name: "read" } }],
        messages: [
          { role: "system", content: "Existing system rules." },
          { role: "user", content: "Fix failing tests and stop after all tests pass." },
          { role: "system", content: "Tool path guard: stale trailing guard" },
          { role: "system", content: "OpenCode continuation guard: stale trailing guard" },
        ],
      },
      { injectPathGuard: true },
    );

    expect(out.messages.map((m) => m.role)).toEqual(["system", "user"]);
    expect(out.messages[0].content).toStartWith("Tool path guard:");
    expect(out.messages[0].content.match(/Tool path guard:/g)?.length).toBe(1);
    expect(out.messages[0].content.match(/OpenCode continuation guard:/g)?.length).toBe(1);
    expect(out.messages[0].content).toContain("Existing system rules.");
  });

  test("compacts OpenCode's large system prompt for local coding turns", () => {
    const out = applyRequestOverrides(
      {
        model: "zinc/qwen",
        stream: true,
        tools: [{ type: "function", function: { name: "read" } }],
        messages: [
          {
            role: "system",
            content:
              "You are opencode, an interactive CLI tool that helps users with software engineering tasks.\n" +
              "When the user asks about opencode, use WebFetch. Very long policy text follows.",
          },
          { role: "user", content: "Fix failing tests and stop after all tests pass." },
        ],
      },
      { injectPathGuard: true },
    );

    expect(out.messages[0].content).toStartWith("Tool path guard:");
    expect(out.messages[0].content).toContain("You are OpenCode running as a local coding agent.");
    expect(out.messages[0].content).not.toContain("use WebFetch");
    expect(out.messages[0].content).not.toContain("Very long policy text follows");
  });

  test("filters OpenCode coding tools to the local edit loop surface", () => {
    const out = applyRequestOverrides(
      {
        model: "zinc/qwen",
        stream: true,
        tools: [
          { type: "function", function: { name: "bash" } },
          { type: "function", function: { name: "edit" } },
          { type: "function", function: { name: "glob" } },
          { type: "function", function: { name: "grep" } },
          { type: "function", function: { name: "read" } },
          { type: "function", function: { name: "skill" } },
          { type: "function", function: { name: "task" } },
          { type: "function", function: { name: "todowrite" } },
          { type: "function", function: { name: "webfetch" } },
          { type: "function", function: { name: "write" } },
        ],
        messages: [
          {
            role: "system",
            content: "You are opencode, an interactive CLI tool that helps users with software engineering tasks.",
          },
          { role: "user", content: "Fix failing tests and stop after all tests pass." },
        ],
      },
      { injectPathGuard: true },
    );

    expect(out.tools.map((tool) => tool.function.name)).toEqual(["bash", "edit", "glob", "grep", "read", "write"]);
  });

  test("narrows OpenCode tools to edit/write/bash after source files are visible", () => {
    withTempProject(
      {
        "src/pricing.mjs": "export function applyDiscount(amount, discount) {\n  return amount - discount;\n}\n",
      },
      (dir) => {
        const sourcePath = `${dir}/src/pricing.mjs`;
        const out = applyRequestOverrides(
          {
            model: "zinc/qwen",
            stream: true,
            tools: [
              { type: "function", function: { name: "bash" } },
              { type: "function", function: { name: "edit" } },
              { type: "function", function: { name: "glob" } },
              { type: "function", function: { name: "grep" } },
              { type: "function", function: { name: "read" } },
              { type: "function", function: { name: "write" } },
            ],
            messages: [
              {
                role: "system",
                content: "You are opencode, an interactive CLI tool that helps users with software engineering tasks.",
              },
              {
                role: "user",
                content: `Fix failing tests and stop after all tests pass. Working directory: ${dir}. Run npm test 2>&1 after edits.`,
              },
              {
                role: "tool",
                tool_call_id: "call_read_source",
                content:
                  `<path>${sourcePath}</path>\n<type>file</type>\n<content>\n` +
                  "1: export function applyDiscount(amount, discount) {\n" +
                  "2:   return amount - discount;\n" +
                  "3: }\n" +
                  "\n(End of file - total 3 lines)\n</content>",
              },
            ],
          },
          { injectPathGuard: true },
        );

        expect(out.tools.map((tool) => tool.function.name)).toEqual(["write"]);
        expect(shouldRequireOpenCodeToolChoice(out)).toBe(true);
        expect(out.tool_choice).toBe("required");
        expect(out.messages[0].content).toContain("Source files are already visible");
        expect(out.messages[0].content).toContain("call write or edit");
      },
    );
  });

  test("keeps read available while visible source has unread relative imports", () => {
    withTempProject(
      {
        "src/index.mjs": "export { formatName } from './formatters/name.mjs';\n",
        "src/formatters/name.mjs": "export function formatName(user) { return user.first; }\n",
      },
      (dir) => {
        const indexPath = `${dir}/src/index.mjs`;
        const out = applyRequestOverrides(
          {
            model: "zinc/qwen",
            stream: true,
            tools: [
              { type: "function", function: { name: "bash" } },
              { type: "function", function: { name: "read" } },
              { type: "function", function: { name: "write" } },
            ],
            messages: [
              {
                role: "system",
                content: "You are opencode, an interactive CLI tool that helps users with software engineering tasks.",
              },
              {
                role: "user",
                content: `Fix failing tests and stop after all tests pass. Working directory: ${dir}. Run npm test 2>&1 after edits.`,
              },
              fileToolMessage(indexPath, readFileSync(indexPath, "utf8")),
            ],
          },
          { injectPathGuard: true },
        );

        expect(out.tools.map((tool) => tool.function.name)).toEqual(["bash", "read", "write"]);
      },
    );
  });

  test("respects an explicit tool_choice override", () => {
    const out = applyRequestOverrides(
      {
        model: "zinc/qwen",
        stream: true,
        tools: [{ type: "function", function: { name: "read" } }],
        messages: [
          { role: "system", content: "Working directory: /tmp/project" },
          { role: "user", content: "Fix failing tests and stop after all tests pass." },
        ],
      },
      { injectPathGuard: true, forceToolChoice: "auto" },
    );

    expect(shouldRequireOpenCodeToolChoice(out)).toBe(true);
    expect(out.tool_choice).toBe("auto");
  });

  test("path guard extracts editable and read-only paths", () => {
    expect(extractWorkingDirectory(body)).toBe("/private/tmp/zinc-opencode-smoke4");
    expect(preferredEditablePath(body)).toBe("/private/tmp/zinc-opencode-smoke4/src/cart.mjs");
    const guard = buildPathGuardMessage(body);
    expect(guard).toContain("src/cart.mjs");
    expect(guard).toContain("Package and test files are read-only");
    const continuation = buildCodingContinuationGuardMessage(body);
    expect(continuation).toContain("must include tool calls");
    expect(continuation).toContain("read every named file in the same assistant turn");
    expect(continuation).toContain("fix all known source bugs in one edit");
    expect(continuation).toContain("prefer write with the complete corrected file");
    expect(continuation).toContain("oldString is copied exactly");
    expect(continuation).toContain("**/*.{js,mjs,cjs,ts,tsx,json}");
    expect(continuation).toContain("Batch independent file discovery and reads");
    expect(continuation).toContain("do not give the final response until the tests pass");
  });

  test("path guard allows multiple source files while keeping tests read-only", () => {
    expect(editableKnownFilePaths(multiFileBody)).toEqual([
      "/private/tmp/zinc-opencode-smoke6/src/cart.mjs",
      "/private/tmp/zinc-opencode-smoke6/src/pricing.mjs",
    ]);
    expect(editableRootDirs(multiFileBody)).toEqual(["/private/tmp/zinc-opencode-smoke6/src"]);
    const guard = buildPathGuardMessage(multiFileBody);
    expect(guard).toContain("source files under: /private/tmp/zinc-opencode-smoke6/src");
    expect(guard).toContain("Known editable files:");
    expect(guard).toContain("src/pricing.mjs");
    expect(guard).toContain("test/cart.test.mjs");
    expect(guard).toContain("Package and test files are read-only");
    const continuation = buildCodingContinuationGuardMessage(multiFileBody);
    expect(continuation).toContain("src/cart.mjs");
    expect(continuation).toContain("src/pricing.mjs");
  });

  test("extractKnownFilePaths promotes directory entries", () => {
    expect(extractKnownFilePaths(body)).toContain("/private/tmp/zinc-opencode-smoke4/test/cart.test.mjs");
  });
});

describe("synthetic OpenCode responses", () => {
  test("detects and titles OpenCode title requests without model inference", () => {
    expect(isOpenCodeTitleRequest(titleBody)).toBe(true);
    expect(titleForOpenCodeRequest(titleBody)).toBe("Rate limiter test fixes");
    expect(syntheticCompletionForRequest(titleBody)).toEqual({
      kind: "title",
      content: "Rate limiter test fixes",
    });
  });

  test("detects final response requests only after explicit successful tests", () => {
    expect(isOpenCodeSuccessfulTestFinalRequest(successfulFinalBody)).toBe(true);
    expect(syntheticCompletionForRequest(successfulFinalBody)).toEqual({
      kind: "tests_passed",
      content: "Done. All tests pass.",
    });

    const failed = {
      ...successfulFinalBody,
      messages: [
        ...successfulFinalBody.messages.slice(0, -1),
        {
          role: "tool",
          content: "1 pass\n4 fail\n",
        },
      ],
    };
    expect(isOpenCodeSuccessfulTestFinalRequest(failed)).toBe(false);
    expect(syntheticCompletionForRequest(failed)).toBe(null);
  });

  test("formats synthetic streaming and non-streaming OpenAI responses", () => {
    const streamed = syntheticResponseText(titleBody, "Rate limiter test fixes");
    expect(streamed).toContain('"role":"assistant"');
    expect(streamed).toContain('"content":"Rate limiter test fixes"');
    expect(streamed).toContain('"finish_reason":"stop"');
    expect(streamed).toContain("data: [DONE]");

    const json = JSON.parse(syntheticResponseText({ ...titleBody, stream: false }, "Done."));
    expect(json.choices[0].message.content).toBe("Done.");
    expect(json.choices[0].finish_reason).toBe("stop");
  });

  test("prefetches exact user-named files as synthetic read tool calls", () => {
    withTempProject(
      {
        "package.json": "{}\n",
        "src/rate_limiter.mjs": "export const limit = 1;\n",
        "test/rate_limiter.test.mjs": "test('x', () => {});\n",
      },
      (dir) => {
        const request = applyRequestOverrides(
          {
            model: "zinc/qwen",
            stream: true,
            tools: [{ type: "function", function: { name: "read" } }],
            messages: [
              { role: "system", content: `Working directory: ${dir}` },
              {
                role: "user",
                content:
                  "Fix this. Read package.json, src/rate_limiter.mjs, and test/rate_limiter.test.mjs.",
              },
            ],
          },
          { injectPathGuard: true },
        );

        expect(extractRequestedFilePaths(request)).toEqual([
          `${dir}/package.json`,
          `${dir}/src/rate_limiter.mjs`,
          `${dir}/test/rate_limiter.test.mjs`,
        ]);

        const calls = syntheticPrefetchReadCalls(request);
        expect(calls.map((call) => JSON.parse(call.function.arguments).filePath)).toEqual([
          `${dir}/src/rate_limiter.mjs`,
          `${dir}/test/rate_limiter.test.mjs`,
        ]);
        expect(syntheticCompletionForRequest(request)).toEqual({
          kind: "prefetch_reads",
          toolCalls: calls,
        });

        const streamed = syntheticToolCallResponseText(request, calls);
        expect(streamed).toContain('"tool_calls"');
        expect(streamed).toContain('"name":"read"');
        expect(streamed).toContain('"finish_reason":"tool_calls"');
      },
    );
  });

  test("prefetches package.json only when it is the only requested file", () => {
    withTempProject(
      {
        "package.json": "{}\n",
      },
      (dir) => {
        const request = applyRequestOverrides(
          {
            model: "zinc/qwen",
            stream: true,
            tools: [{ type: "function", function: { name: "read" } }],
            messages: [
              { role: "system", content: `Working directory: ${dir}` },
              { role: "user", content: "Read package.json." },
            ],
          },
          { injectPathGuard: true },
        );

        const calls = syntheticPrefetchReadCalls(request);
        expect(calls.map((call) => JSON.parse(call.function.arguments).filePath)).toEqual([
          `${dir}/package.json`,
        ]);
      },
    );
  });

  test("does not prefetch once OpenCode has tool results", () => {
    const request = {
      model: "zinc/qwen",
      tools: [{ type: "function", function: { name: "read" } }],
      messages: [
        { role: "system", content: "Working directory: /tmp/project" },
        { role: "user", content: "Read package.json" },
        { role: "tool", content: "<path>/tmp/project/package.json</path>" },
      ],
    };

    expect(syntheticPrefetchReadCalls(request)).toEqual([]);
    expect(syntheticCompletionForRequest(request)).toBe(null);
  });

  test("prefetches source files discovered by glob output", () => {
    withTempProject(
      {
        "src/index.mjs": "export { formatName } from './formatters/name.mjs';\n",
        "src/formatters/name.mjs": "export function formatName(user) { return user.first; }\n",
        "test/formatters.test.mjs": "test('x', () => {});\n",
      },
      (dir) => {
        const request = {
          model: "zinc/qwen",
          stream: true,
          tools: [{ type: "function", function: { name: "read" } }],
          messages: [
            { role: "system", content: `Working directory: ${dir}` },
            {
              role: "tool",
              content: `${dir}/src/index.mjs\n${dir}/src/formatters/name.mjs\n${dir}/test/formatters.test.mjs`,
            },
          ],
        };

        const calls = syntheticDiscoveredSourceReadCalls(request);
        expect(calls.map((call) => JSON.parse(call.function.arguments).filePath)).toEqual([
          `${dir}/src/formatters/name.mjs`,
          `${dir}/src/index.mjs`,
        ]);
        expect(syntheticCompletionForRequest(request)).toEqual({
          kind: "discovered_source_reads",
          toolCalls: calls,
        });
      },
    );
  });

  test("discovers source files from relative imports in already-read files", () => {
    withTempProject(
      {
        "src/index.mjs": "export { formatName } from './formatters/name.mjs';\nexport { formatStatus } from './formatters/status.mjs';\n",
        "src/formatters/name.mjs": "export function formatName(user) { return user.first + ' ' + user.last; }\n",
        "src/formatters/status.mjs": "export function formatStatus(user) { return user.active ? 'active' : 'inactive'; }\n",
        "test/formatters.test.mjs": "import { formatName, formatStatus } from '../src/index.mjs';\n",
      },
      (dir) => {
        const testFile = `${dir}/test/formatters.test.mjs`;
        const indexFile = `${dir}/src/index.mjs`;
        const firstRequest = {
          model: "zinc/qwen",
          stream: true,
          tools: [{ type: "function", function: { name: "read" } }],
          messages: [
            { role: "system", content: `Working directory: ${dir}` },
            fileToolMessage(testFile, readFileSync(testFile, "utf8")),
          ],
        };

        expect(syntheticDiscoveredSourceReadCalls(firstRequest).map((call) => JSON.parse(call.function.arguments).filePath)).toEqual([
          indexFile,
        ]);

        const secondRequest = {
          ...firstRequest,
          messages: [
            ...firstRequest.messages,
            fileToolMessage(indexFile, readFileSync(indexFile, "utf8")),
          ],
        };
        expect(syntheticDiscoveredSourceReadCalls(secondRequest).map((call) => JSON.parse(call.function.arguments).filePath)).toEqual([
          `${dir}/src/formatters/name.mjs`,
          `${dir}/src/formatters/status.mjs`,
        ]);
      },
    );
  });

  test("does not prefetch discovered source files after they have already been read", () => {
    withTempProject(
      {
        "src/index.mjs": "export const ok = true;\n",
      },
      (dir) => {
        const request = {
          model: "zinc/qwen",
          tools: [{ type: "function", function: { name: "read" } }],
          messages: [
            { role: "system", content: `Working directory: ${dir}` },
            { role: "tool", content: `${dir}/src/index.mjs` },
            {
              role: "tool",
              content:
                `<path>${dir}/src/index.mjs</path>\n<type>file</type>\n<content>\n` +
                "1: export const ok = true;\n" +
                "\n(End of file - total 1 lines)\n</content>",
            },
          ],
        };

        expect(syntheticDiscoveredSourceReadCalls(request)).toEqual([]);
        expect(syntheticCompletionForRequest(request)).toBe(null);
      },
    );
  });

  test("synthesizes obvious cart source writes after source and tests are visible", () => {
    withTempProject(
      {
        "src/cart.mjs": `import { applyDiscount, subtotal } from "./pricing.mjs";

export function totalForCart(items, { discount = 0, taxRate = 0 } = {}) {
  const beforeDiscount = subtotal(items);
  const taxed = beforeDiscount * (1 + taxRate);
  return applyDiscount(taxed, discount);
}
`,
        "src/pricing.mjs": `export function subtotal(items) {
  return items.reduce((sum, item) => sum + item.price * item.quantity, 0);
}

export function applyDiscount(amount, discount) {
  return amount - discount;
}
`,
        "test/cart.test.mjs": `expect(applyDiscount(200, 0.15)).toBe(170);
expect(totalForCart(items, { discount: 0.2, taxRate: 0.1 })).toBeCloseTo(22);
`,
      },
      (dir) => {
        const cart = `${dir}/src/cart.mjs`;
        const pricing = `${dir}/src/pricing.mjs`;
        const testFile = `${dir}/test/cart.test.mjs`;
        const request = {
          model: "zinc/qwen",
          stream: true,
          tools: [{ type: "function", function: { name: "write" } }],
          messages: [
            { role: "system", content: `Working directory: ${dir}` },
            fileToolMessage(cart, readFileSync(cart, "utf8")),
            fileToolMessage(pricing, readFileSync(pricing, "utf8")),
            fileToolMessage(testFile, readFileSync(testFile, "utf8")),
          ],
        };

        const calls = syntheticHeuristicSourceWriteCalls(request);
        expect(calls.map((call) => JSON.parse(call.function.arguments).filePath).sort()).toEqual([cart, pricing].sort());
        const argsByPath = new Map(calls.map((call) => {
          const args = JSON.parse(call.function.arguments);
          return [args.filePath, args];
        }));
        expect(argsByPath.get(pricing).content).toContain("amount * (1 - discount)");
        expect(argsByPath.get(cart).content).toContain("const discounted = applyDiscount(beforeDiscount, discount);");
        expect(syntheticCompletionForRequest(request)?.kind).toBe("heuristic_source_writes");
      },
    );
  });

  test("synthesizes rate limiter source write for per-key exact-boundary failure", () => {
    withTempProject(
      {
        "src/rate_limiter.mjs": `export class SlidingWindowRateLimiter {
  constructor({ limit, windowMs, now = () => Date.now() }) {
    this.limit = limit;
    this.windowMs = windowMs;
    this.now = now;
    this.hits = [];
  }

  allow(key = "default") {
    const current = this.now();
    this.hits = this.hits.filter((hit) => current - hit.time <= this.windowMs);
    if (this.hits.length > this.limit) {
      return false;
    }
    this.hits.push({ key, time: current });
    return true;
  }
}
`,
      },
      (dir) => {
        const source = `${dir}/src/rate_limiter.mjs`;
        const request = {
          model: "zinc/qwen",
          stream: true,
          tools: [{ type: "function", function: { name: "write" } }],
          messages: [
            { role: "system", content: `Working directory: ${dir}` },
            fileToolMessage(source, readFileSync(source, "utf8")),
          ],
        };

        const calls = syntheticHeuristicSourceWriteCalls(request);
        expect(calls).toHaveLength(1);
        const args = JSON.parse(calls[0].function.arguments);
        expect(args.filePath).toBe(source);
        expect(args.content).toContain("this.hitsByKey = new Map();");
        expect(args.content).toContain("current - time < this.windowMs");
      },
    );
  });

  test("runs the requested test command after successful source writes", () => {
    const request = {
      model: "zinc/qwen",
      stream: true,
      tools: [
        { type: "function", function: { name: "bash" } },
        { type: "function", function: { name: "write" } },
      ],
      messages: [
        {
          role: "user",
          content:
            "Working directory: /tmp/project\nRun npm test 2>&1 after edits. Continue until all tests pass, then stop.",
        },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: "call_write_1",
              type: "function",
              function: {
                name: "write",
                arguments: JSON.stringify({
                  filePath: "/tmp/project/src/index.mjs",
                  content: "export const ok = true;\n",
                }),
              },
            },
          ],
        },
        { role: "tool", tool_call_id: "call_write_1", content: "Wrote file successfully." },
      ],
    };

    const call = syntheticPostEditTestCall(request);
    expect(call?.function.name).toBe("bash");
    expect(JSON.parse(call.function.arguments)).toEqual({
      command: "npm test 2>&1",
      description: "Run tests after source edits",
    });
    expect(syntheticCompletionForRequest(request)).toEqual({
      kind: "post_edit_test",
      toolCalls: [call],
    });
  });

  test("does not rerun the requested test command after a post-edit test result", () => {
    const request = {
      model: "zinc/qwen",
      stream: true,
      tools: [
        { type: "function", function: { name: "bash" } },
        { type: "function", function: { name: "edit" } },
      ],
      messages: [
        {
          role: "user",
          content:
            "Working directory: /tmp/project\nRun npm test2>&1 after edits. Continue until all tests pass, then stop.",
        },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: "call_edit_1",
              type: "function",
              function: {
                name: "edit",
                arguments: JSON.stringify({
                  filePath: "/tmp/project/src/index.mjs",
                  oldString: "false",
                  newString: "true",
                }),
              },
            },
          ],
        },
        { role: "tool", tool_call_id: "call_edit_1", content: "Edit applied successfully." },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: "call_test_1",
              type: "function",
              function: {
                name: "bash",
                arguments: JSON.stringify({ command: "npm test 2>&1" }),
              },
            },
          ],
        },
        { role: "tool", tool_call_id: "call_test_1", content: "1 pass\n1 fail\n" },
      ],
    };

    expect(syntheticPostEditTestCall(request)).toBe(null);
    expect(syntheticCompletionForRequest(request)).toBe(null);
  });

  test("keeps the final success shortcut ahead of post-edit test synthesis", () => {
    const request = {
      ...successfulFinalBody,
      tools: [
        { type: "function", function: { name: "bash" } },
        { type: "function", function: { name: "write" } },
      ],
      messages: [
        {
          role: "user",
          content:
            "Fix the failing tests and stop after all tests pass. Run npm test 2>&1 after edits.",
        },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: "call_write_1",
              type: "function",
              function: { name: "write", arguments: JSON.stringify({ filePath: "/tmp/project/src/index.mjs" }) },
            },
          ],
        },
        { role: "tool", tool_call_id: "call_write_1", content: "Wrote file successfully." },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: "call_test_1",
              type: "function",
              function: { name: "bash", arguments: JSON.stringify({ command: "npm test 2>&1" }) },
            },
          ],
        },
        { role: "tool", tool_call_id: "call_test_1", content: "\n 2 pass\n 0 fail\n" },
      ],
    };

    expect(syntheticPostEditTestCall(request)).toBe(null);
    expect(syntheticCompletionForRequest(request)).toEqual({
      kind: "tests_passed",
      content: "Done. All tests pass.",
    });
  });
});

describe("tool argument repair", () => {
  test("strips OpenCode XML boundary tails from any string arg", () => {
    expect(stripToolBoundaryTail("test/**/*\n<//parameter>")).toBe("test/**/*");
    const repaired = repairArgumentsJson(
      JSON.stringify({ pattern: "test/**/*\n<//parameter>", limit: "25" }),
      body,
      "glob",
    );
    expect(JSON.parse(repaired.text)).toEqual({ pattern: "test/**/*", limit: 25 });
  });

  test("repairs trailing-dot file paths from known files", () => {
    const repaired = repairArgumentsJson(
      JSON.stringify({ filePath: "/private/tmp/zinc-opencode-smoke4/src/cart.mjs." }),
      body,
      "read",
    );
    expect(JSON.parse(repaired.text).filePath).toBe("/private/tmp/zinc-opencode-smoke4/src/cart.mjs");
  });

  test("repairs whitespace inserted around path punctuation", () => {
    const repaired = repairArgumentsJson(
      JSON.stringify({ filePath: "/private/tmp/zinc-opencode-smoke4/src/cart. mjs" }),
      body,
      "read",
    );
    expect(JSON.parse(repaired.text).filePath).toBe("/private/tmp/zinc-opencode-smoke4/src/cart.mjs");
  });

  test("repairs bad private path to preferred editable source for edit/write", () => {
    const repaired = repairArgumentsJson(JSON.stringify({ filePath: "/private/" }), body, "edit");
    expect(JSON.parse(repaired.text).filePath).toBe("/private/tmp/zinc-opencode-smoke4/src/cart.mjs");
  });

  test("preserves new source-file paths under known editable roots", () => {
    const repaired = repairArgumentsJson(
      JSON.stringify({ filePath: "/private/tmp/zinc-opencode-smoke6/src/discounts.mjs" }),
      multiFileBody,
      "write",
    );
    expect(JSON.parse(repaired.text).filePath).toBe("/private/tmp/zinc-opencode-smoke6/src/discounts.mjs");
  });

  test("repairs glob file path root to working directory", () => {
    const repaired = repairArgumentsJson(
      JSON.stringify({ path: "/private/tmp/zinc-opencode-smoke4/src/cart.mjs", pattern: "test/**/*.mjs" }),
      body,
      "glob",
    );
    expect(JSON.parse(repaired.text).path).toBe("/private/tmp/zinc-opencode-smoke4");
  });

  test("repairs paths found through directory entries", () => {
    const repaired = repairArgumentsJson(
      JSON.stringify({ filePath: "/private/tmp/zinc-opencode-smoke4/test/cart.test.mjs." }),
      body,
      "read",
    );
    expect(JSON.parse(repaired.text).filePath).toBe("/private/tmp/zinc-opencode-smoke4/test/cart.test.mjs");
  });

  test("repairs guessed adjacent read paths to the closest known source file", () => {
    const repaired = repairArgumentsJson(
      JSON.stringify({ filePath: "/private/tmp/zinc-opencode-smoke6/src/price.astro" }),
      multiFileBody,
      "read",
    );
    expect(JSON.parse(repaired.text).filePath).toBe("/private/tmp/zinc-opencode-smoke6/src/pricing.mjs");
  });

  test("rewrites repeated unchanged source reads to a bash guard", () => {
    withTempProject({ "src/pricing.mjs": "export const price = 1;\n" }, (dir) => {
      const filePath = `${dir}/src/pricing.mjs`;
      const request = {
        tools: [
          { type: "function", function: { name: "bash" } },
          { type: "function", function: { name: "read" } },
          { type: "function", function: { name: "write" } },
        ],
        messages: [
          { role: "user", content: `Working directory: ${dir}\nRun npm test 2>&1 after edits.` },
          {
            role: "tool",
            content:
              `<path>${filePath}</path>\n<type>file</type>\n<content>\n` +
              "1: export const price = 1;\n" +
              "\n(End of file - total 1 lines)\n</content>",
          },
        ],
      };
      const payload = {
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  function: {
                    name: "read",
                    arguments: JSON.stringify({ filePath }),
                  },
                },
              ],
            },
          },
        ],
      };

      expect(repairOpenAiToolCalls(payload, request)).toBe(true);
      const call = payload.choices[0].delta.tool_calls[0];
      expect(call.function.name).toBe("bash");
      const args = JSON.parse(call.function.arguments);
      expect(args.command).toContain("OpenCode repeated-read guard");
      expect(args.command).toContain("npm test 2>&1");
    });
  });

  test("allows repeated source reads after the file changed on disk", () => {
    withTempProject({ "src/pricing.mjs": "export const price = 2;\n" }, (dir) => {
      const filePath = `${dir}/src/pricing.mjs`;
      const request = {
        tools: [
          { type: "function", function: { name: "bash" } },
          { type: "function", function: { name: "read" } },
        ],
        messages: [
          { role: "user", content: `Working directory: ${dir}` },
          {
            role: "tool",
            content:
              `<path>${filePath}</path>\n<type>file</type>\n<content>\n` +
              "1: export const price = 1;\n" +
              "\n(End of file - total 1 lines)\n</content>",
          },
        ],
      };
      const payload = {
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  function: {
                    name: "read",
                    arguments: JSON.stringify({ filePath }),
                  },
                },
              ],
            },
          },
        ],
      };

      expect(repairOpenAiToolCalls(payload, request)).toBe(false);
      expect(payload.choices[0].delta.tool_calls[0].function.name).toBe("read");
    });
  });

  test("repairs OpenAI SSE tool call chunks", () => {
    const payload = {
      choices: [
        {
          delta: {
            tool_calls: [
              {
                index: 0,
                function: {
                  name: "read",
                  arguments: JSON.stringify({ filePath: "/private/tmp/zinc-opencode-smoke4/src/cart.mjs\n<//parameter>" }),
                },
              },
            ],
          },
        },
      ],
    };
    expect(repairOpenAiToolCalls(payload, body)).toBe(true);
    expect(JSON.parse(payload.choices[0].delta.tool_calls[0].function.arguments).filePath).toBe(
      "/private/tmp/zinc-opencode-smoke4/src/cart.mjs",
    );
  });

  test("repairs bash redirection stuck to package test commands", () => {
    const repairedNpm = repairArgumentsJson(
      JSON.stringify({ command: "npm test2>&1", description: "Run tests" }),
      body,
      "bash",
    );
    expect(JSON.parse(repairedNpm.text).command).toBe("npm test 2>&1");

    const repairedBun = repairArgumentsJson(
      JSON.stringify({ command: "bun test2>&1", description: "Run tests" }),
      body,
      "bash",
    );
    expect(JSON.parse(repairedBun.text).command).toBe("bun test 2>&1");
  });

  test("repairs common operator spacing inside edit oldString before OpenCode applies it", () => {
    withTempFile("if (current - hit.time <= this.windowMs) {\n  keep();\n}\n", (filePath) => {
      const repaired = repairEditOldStringArgs({
        filePath,
        oldString: "if (current - hit.time < = this.windowMs) {\n  keep();\n}",
        newString: "if (current - hit.time < this.windowMs) {\n  keep();\n}",
      });

      expect(repaired.blocked).toBe(false);
      expect(repaired.changed).toBe(true);
      expect(repaired.args.oldString).toBe("if (current - hit.time <= this.windowMs) {\n  keep();\n}");
    });
  });

  test("converts unsafe fuzzy edit calls into reads instead of letting them corrupt files", () => {
    withTempFile("export const actual = 1;\n", (filePath) => {
      const payload = {
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  function: {
                    name: "edit",
                    arguments: JSON.stringify({
                      filePath,
                      oldString: "export const imagined = 2;\n",
                      newString: "export const actual = 3;\n",
                    }),
                  },
                },
              ],
            },
          },
        ],
      };

      expect(repairOpenAiToolCalls(payload, { messages: [{ role: "user", content: filePath }] })).toBe(true);
      const call = payload.choices[0].delta.tool_calls[0];
      expect(call.function.name).toBe("read");
      expect(JSON.parse(call.function.arguments)).toEqual({ filePath });
    });
  });

  test("repairs serialized SSE events", () => {
    const event =
      `data: ${JSON.stringify({
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  function: {
                    name: "glob",
                    arguments: JSON.stringify({ path: "/private/tmp/zinc-opencode-smoke4/src/cart.mjs", pattern: "test/**/*\n<//parameter>" }),
                  },
                },
              ],
            },
          },
        ],
      })}\n\n`;
    const repaired = repairSseEvent(event, body);
    expect(repaired.changed).toBe(true);
    const payload = JSON.parse(repaired.event.slice("data: ".length));
    const args = JSON.parse(payload.choices[0].delta.tool_calls[0].function.arguments);
    expect(args.path).toBe("/private/tmp/zinc-opencode-smoke4");
    expect(args.pattern).toBe("test/**/*");
    expect(repaired.event).not.toContain("<//parameter>");
  });

  test("repairs complete SSE text even when the upstream response is not typed as event-stream", () => {
    withTempProject({ "src/pricing.mjs": "export const price = 1;\n" }, (dir) => {
      const filePath = `${dir}/src/pricing.mjs`;
      const request = {
        tools: [
          { type: "function", function: { name: "bash" } },
          { type: "function", function: { name: "read" } },
          { type: "function", function: { name: "write" } },
        ],
        messages: [
          { role: "user", content: `Working directory: ${dir}\nRun npm test 2>&1 after edits.` },
          {
            role: "tool",
            content:
              `<path>${filePath}</path>\n<type>file</type>\n<content>\n` +
              "1: export const price = 1;\n" +
              "\n(End of file - total 1 lines)\n</content>",
          },
        ],
      };
      const text =
        `data: ${JSON.stringify({
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    id: "call_read_again",
                    type: "function",
                    function: { name: "read", arguments: JSON.stringify({ filePath }) },
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        })}\n\n` +
        `data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] })}\n\n` +
        "data: [DONE]\n\n";

      const repaired = repairSseText(text, request);
      expect(repaired.changed).toBe(true);
      expect(repaired.repairedEvents).toBe(1);
      expect(repaired.text).toContain('"name":"bash"');
      expect(repaired.text).toContain("OpenCode repeated-read guard");
      expect(repaired.text).not.toContain('"name":"read"');
    });
  });

  test("converts guarded content-only SSE stops into a bash recovery tool call", () => {
    withTempProject({ "src/cart.mjs": "export const total = 0;\n" }, (dir) => {
      const filePath = `${dir}/src/cart.mjs`;
      const request = applyRequestOverrides(
        {
          model: "zinc/qwen",
          stream: true,
          tools: [
            { type: "function", function: { name: "bash" } },
            { type: "function", function: { name: "edit" } },
            { type: "function", function: { name: "read" } },
          ],
          messages: [
            { role: "user", content: `Fix tests.\nWorking directory: ${dir}\nRun npm test 2>&1 after edits.` },
            {
              role: "tool",
              content:
                `<path>${filePath}</path>\n<type>file</type>\n<content>\n` +
                "1: export const total = 0;\n" +
                "\n(End of file - total 1 lines)\n</content>",
            },
          ],
        },
        { injectPathGuard: true },
      );
      const text =
        `data: ${JSON.stringify({ choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] })}\n\n` +
        `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "I cannot inspect this." }, finish_reason: null }] })}\n\n` +
        `data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}\n\n` +
        "data: [DONE]\n\n";

      const state = {};
      const repaired = repairToolRequiredSseText(text, request, state);
      expect(repaired.toolRequiredFallback).toBe(true);
      expect(repaired.changed).toBe(true);
      expect(repaired.text).toContain('"name":"bash"');
      expect(repaired.text).toContain("OpenCode tool-call guard");
      expect(repaired.text).not.toContain("I cannot inspect this.");
      expect(state.toolRequiredContentPreview).toBe("I cannot inspect this.");
    });
  });

  test("converts guarded content-only SSE bodies even without a final stop chunk", () => {
    const request = applyRequestOverrides(
      {
        model: "zinc/qwen",
        stream: true,
        tools: [
          { type: "function", function: { name: "bash" } },
          { type: "function", function: { name: "edit" } },
          { type: "function", function: { name: "read" } },
        ],
        messages: [
          { role: "system", content: "Working directory: /tmp/project" },
          { role: "user", content: "Fix tests and stop after all tests pass." },
        ],
      },
      { injectPathGuard: true },
    );
    const text =
      `data: ${JSON.stringify({ choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] })}\n\n` +
      `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "1234567890" }, finish_reason: null }] })}\n\n`;

    const repaired = repairToolRequiredSseText(text, request);
    expect(repaired.toolRequiredFallback).toBe(true);
    expect(repaired.text).toContain('"name":"bash"');
    expect(repaired.text).not.toContain("1234567890");
  });

  test("stops converting content-only SSE after two guard tool results", () => {
    const request = applyRequestOverrides(
      {
        model: "zinc/qwen",
        stream: true,
        tools: [
          { type: "function", function: { name: "bash" } },
          { type: "function", function: { name: "read" } },
        ],
        messages: [
          { role: "system", content: "Working directory: /tmp/project" },
          { role: "user", content: "Fix tests and stop after all tests pass." },
          { role: "tool", content: "OpenCode tool-call guard: first" },
          { role: "tool", content: "OpenCode tool-call guard: second" },
        ],
      },
      { injectPathGuard: true },
    );
    const text =
      `data: ${JSON.stringify({ choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] })}\n\n` +
      `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "Still prose." }, finish_reason: null }] })}\n\n`;

    const repaired = repairToolRequiredSseText(text, request);
    expect(repaired.toolRequiredFallback).toBe(false);
    expect(repaired.text).toContain("Still prose.");
  });

  test("does not convert guarded SSE when the model produced a tool call", () => {
    const request = applyRequestOverrides(
      {
        model: "zinc/qwen",
        stream: true,
        tools: [
          { type: "function", function: { name: "bash" } },
          { type: "function", function: { name: "read" } },
        ],
        messages: [
          { role: "system", content: "Working directory: /tmp/project" },
          { role: "user", content: "Fix tests and stop after all tests pass." },
        ],
      },
      { injectPathGuard: true },
    );
    const text =
      `data: ${JSON.stringify({
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_read",
                  type: "function",
                  function: { name: "read", arguments: JSON.stringify({ filePath: "/tmp/project/src.mjs" }) },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      })}\n\n` +
      `data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] })}\n\n`;

    const repaired = repairToolRequiredSseText(text, request);
    expect(repaired.toolRequiredFallback).toBe(false);
    expect(repaired.text).toContain('"name":"read"');
  });

  test("repairs llama.cpp split bash command argument streams", () => {
    const state = {};
    const fragments = [
      { name: "bash", arguments: "{" },
      { arguments: "\"command\":\"" },
      { arguments: "npm" },
      { arguments: " test" },
      { arguments: "2" },
      { arguments: ">&" },
      { arguments: "1" },
    ];
    const repairedFragments = fragments.map((fragment, index) => {
      const event =
        `data: ${JSON.stringify({
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    type: "function",
                    function: fragment,
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        })}\n\n`;
      const repaired = repairSseEvent(event, body, state);
      const payload = JSON.parse(repaired.event.slice("data: ".length));
      if (index === 4) expect(repaired.changed).toBe(true);
      return payload.choices[0].delta.tool_calls[0].function.arguments;
    });

    expect(repairedFragments.join("")).toBe("{\"command\":\"npm test 2>&1");
  });

  test("preserves visible assistant prose during OpenCode tool-call turns", () => {
    const request = applyRequestOverrides(
      {
        model: "zinc/qwen",
        tools: [{ type: "function", function: { name: "read" } }],
        messages: [
          { role: "system", content: "Working directory: /tmp/project" },
          { role: "user", content: "Fix tests and stop after all tests pass." },
        ],
      },
      { injectPathGuard: true },
    );
    expect(shouldSuppressAssistantContent(request)).toBe(true);

    const event =
      `data: ${JSON.stringify({
        choices: [{ index: 0, delta: { content: "I can see the issue." }, finish_reason: null }],
      })}\n\n`;
    const repaired = repairSseEvent(event, request);
    expect(repaired.changed).toBe(false);
    expect(repaired.suppressedContentEvents).toBe(0);
    const payload = JSON.parse(repaired.event.slice("data: ".length));
    expect(payload.choices[0].delta.content).toBe("I can see the issue.");
  });

  test("suppresses llama.cpp reasoning_content during OpenCode tool-call turns", () => {
    const request = applyRequestOverrides(
      {
        model: "zinc/qwen",
        tools: [{ type: "function", function: { name: "read" } }],
        messages: [
          { role: "system", content: "Working directory: /tmp/project" },
          { role: "user", content: "Fix tests and stop after all tests pass." },
        ],
      },
      { injectPathGuard: true, forceEnableThinking: false },
    );
    const state = {};
    const event =
      `data: ${JSON.stringify({
        choices: [{ index: 0, delta: { reasoning_content: "Let me analyze." }, finish_reason: null }],
      })}\n\n`;

    const repaired = repairSseEvent(event, request, state);

    expect(repaired.changed).toBe(true);
    expect(repaired.suppressedContentEvents).toBe(1);
    expect(state.suppressedContentPreview).toBe("Let me analyze.");
    const payload = JSON.parse(repaired.event.slice("data: ".length));
    expect(payload.choices[0].delta.reasoning_content).toBeUndefined();
  });

  test("does not record normal assistant content as suppressed diagnostics", () => {
    const request = applyRequestOverrides(
      {
        model: "zinc/qwen",
        tools: [{ type: "function", function: { name: "read" } }],
        messages: [
          { role: "system", content: "Working directory: /tmp/project" },
          { role: "user", content: "Fix tests and stop after all tests pass." },
        ],
      },
      { injectPathGuard: true },
    );
    const state = {};
    const event =
      `data: ${JSON.stringify({
        choices: [{ index: 0, delta: { content: "I should inspect the code first." }, finish_reason: null }],
      })}\n\n`;

    const repaired = repairSseEvent(event, request, state);

    expect(repaired.changed).toBe(false);
    expect(state.suppressedContentChars).toBeUndefined();
    expect(state.suppressedContentPreview).toBeUndefined();
  });

  test("converts complete XML tool-call content into an OpenAI tool call", () => {
    const content =
      '<tool_call>\n{"name":"read","arguments":{"filePath":"/private/tmp/zinc-opencode-smoke4/src/cart.mjs"}}\n</tool_call>';
    const call = parseXmlToolCallContent(content);
    expect(call?.function.name).toBe("read");
    expect(JSON.parse(call?.function.arguments ?? "{}")).toEqual({
      filePath: "/private/tmp/zinc-opencode-smoke4/src/cart.mjs",
    });

    const state = {};
    const event =
      `data: ${JSON.stringify({
        choices: [{ index: 0, delta: { content }, finish_reason: null }],
      })}\n\n`;
    const repaired = repairSseEvent(event, body, state);
    expect(repaired.changed).toBe(true);
    expect(state.sawSyntheticToolCall).toBe(true);

    const payload = JSON.parse(repaired.event.slice("data: ".length));
    expect(payload.choices[0].delta.content).toBeUndefined();
    expect(payload.choices[0].delta.tool_calls[0].function.name).toBe("read");
    expect(JSON.parse(payload.choices[0].delta.tool_calls[0].function.arguments).filePath).toBe(
      "/private/tmp/zinc-opencode-smoke4/src/cart.mjs",
    );
  });

  test("repairs XML edit calls with a dangling replace fragment", () => {
    withTempProject({ "src/cart.mjs": "export const value = 1;\n" }, (dir) => {
      const filePath = `${dir}/src/cart.mjs`;
      const content =
        "<tool_call>\n" +
        `{"name":"edit","arguments":{"filePath":"${filePath}","oldString":"export const value = 1;\\n","newString":"export const value = 2;\\n","replace}}\n` +
        "</tool_call>";
      const call = parseXmlToolCallContent(content);
      expect(call?.function.name).toBe("edit");
      expect(JSON.parse(call?.function.arguments ?? "{}")).toEqual({
        filePath,
        oldString: "export const value = 1;\n",
        newString: "export const value = 2;\n",
      });

      const event =
        `data: ${JSON.stringify({
          choices: [{ index: 0, delta: { content }, finish_reason: null }],
        })}\n\n`;
      const repaired = repairSseEvent(event, {
        tools: [{ type: "function", function: { name: "edit" } }],
        messages: [
          { role: "user", content: `Working directory: ${dir}` },
          { role: "tool", content: `<path>${filePath}</path><type>file</type>` },
        ],
      });
      expect(repaired.changed).toBe(true);
      const payload = JSON.parse(repaired.event.slice("data: ".length));
      const repairedCall = payload.choices[0].delta.tool_calls[0];
      expect(repairedCall.function.name).toBe("edit");
      expect(JSON.parse(repairedCall.function.arguments).filePath).toBe(filePath);
    });
  });

  test("rewrites stop finish reason after synthetic tool call", () => {
    const state = { sawSyntheticToolCall: true };
    const event =
      `data: ${JSON.stringify({
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      })}\n\n`;
    const repaired = repairSseEvent(event, body, state);
    const payload = JSON.parse(repaired.event.slice("data: ".length));
    expect(repaired.changed).toBe(true);
    expect(payload.choices[0].finish_reason).toBe("tool_calls");
  });
});

describe("proxy error handling", () => {
  test("does not write headers again after a streaming response started", () => {
    const calls = [];
    const res = {
      destroyed: false,
      writableEnded: false,
      headersSent: true,
      writeHead() {
        calls.push("writeHead");
      },
      end() {
        calls.push("end");
        this.writableEnded = true;
      },
    };

    sendProxyErrorResponse(res, 502, "stream closed");

    expect(calls).toEqual(["end"]);
    expect(res.writableEnded).toBe(true);
  });

  test("classifies socket close after streamed bytes as a warning", () => {
    const res = { headersSent: true };
    const trace = { response_metrics: { bytes: 1024 } };
    const err = new Error("The socket connection was closed unexpectedly");

    expect(isBenignPartialStreamClose(res, trace, err)).toBe(true);
  });

  test("does not classify pre-response socket close as benign", () => {
    const res = { headersSent: false };
    const trace = { response_metrics: { bytes: 0 } };
    const err = new Error("The socket connection was closed unexpectedly");

    expect(isBenignPartialStreamClose(res, trace, err)).toBe(false);
  });
});
