import { describe, expect, test } from "bun:test";
import {
  applyRequestOverrides,
  buildCodingContinuationGuardMessage,
  buildPathGuardMessage,
  editableKnownFilePaths,
  editableRootDirs,
  deriveSessionId,
  extractKnownFilePaths,
  extractWorkingDirectory,
  injectSessionId,
  isBenignPartialStreamClose,
  parseArgs,
  preferredEditablePath,
  repairArgumentsJson,
  repairOpenAiToolCalls,
  repairSseEvent,
  sendProxyErrorResponse,
  stripToolBoundaryTail,
} from "./opencode_trace_proxy.mjs";

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
    expect(out.tool_choice).toBe("auto");
    expect(out.max_tokens).toBe(768);
    expect(out.temperature).toBe(0);
    expect(out.top_p).toBe(1);
    expect(out.messages.at(-2)?.content).toStartWith("Tool path guard:");
    expect(out.messages.at(-1)?.content).toStartWith("OpenCode continuation guard:");
  });

  test("path guard extracts editable and read-only paths", () => {
    expect(extractWorkingDirectory(body)).toBe("/private/tmp/zinc-opencode-smoke4");
    expect(preferredEditablePath(body)).toBe("/private/tmp/zinc-opencode-smoke4/src/cart.mjs");
    const guard = buildPathGuardMessage(body);
    expect(guard).toContain("src/cart.mjs");
    expect(guard).toContain("Package and test files are read-only");
    const continuation = buildCodingContinuationGuardMessage(body);
    expect(continuation).toContain("fix all known source bugs in one edit");
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
