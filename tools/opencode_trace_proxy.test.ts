import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
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
  repairSseEvent,
  sendProxyErrorResponse,
  shouldSuppressAssistantContent,
  shouldRequireOpenCodeToolChoice,
  stripToolBoundaryTail,
  syntheticCompletionForRequest,
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
    expect(out.messages.at(-2)?.content).toStartWith("Tool path guard:");
    expect(out.messages.at(-1)?.content).toStartWith("OpenCode continuation guard:");
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
    expect(continuation).toContain("tool calls only");
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
          `${dir}/package.json`,
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

  test("suppresses visible assistant prose during OpenCode tool-call turns", () => {
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
    expect(repaired.changed).toBe(true);
    expect(repaired.suppressedContentEvents).toBe(1);
    const payload = JSON.parse(repaired.event.slice("data: ".length));
    expect(payload.choices[0].delta.content).toBeUndefined();
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

  test("records suppressed assistant prose for trace diagnostics", () => {
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

    expect(repaired.changed).toBe(true);
    expect(state.suppressedContentChars).toBe("I should inspect the code first.".length);
    expect(state.suppressedContentPreview).toBe("I should inspect the code first.");
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
