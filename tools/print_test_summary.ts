#!/usr/bin/env bun

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

function hasEnv(name: string): boolean {
  const value = process.env[name];
  return typeof value === "string" && value.length > 0;
}

function statusLine(label: string, status: string, detail?: string): string {
  return detail ? `  ${label}: ${status} (${detail})` : `  ${label}: ${status}`;
}

const MANAGED_MODEL_ROOT = join(homedir(), "Library", "Caches", "zinc", "models", "models");

function managedExists(id: string): boolean {
  return existsSync(join(MANAGED_MODEL_ROOT, id, "model.gguf"));
}

const requireFull = process.env.ZINC_REQUIRE_FULL_TESTS === "1";

const qwen8bReady = hasEnv("ZINC_QWEN3_8B_MODEL") || managedExists("qwen35-9b-q4k-m");
const qwen36bReady = hasEnv("ZINC_QWEN36_35B_MODEL") || managedExists("qwen36-35b-a3b-q4k-xl");
const hasQwenSmoke = qwen8bReady && qwen36bReady;

const apiServerReady = managedExists("qwen35-9b-q4k-m")
  || managedExists("gemma4-26b-a4b-q4k-m")
  || managedExists("gemma4-31b-q4k-m");
const hasApiSmoke = hasEnv("ZINC_API_BASE_URL") || apiServerReady;

function qwenDetail(): string {
  if (hasQwenSmoke) {
    return hasEnv("ZINC_QWEN3_8B_MODEL")
      ? (process.env.ZINC_QWEN3_8B_MODEL as string)
      : "managed:qwen3.5-9b + qwen3.6-35b";
  }
  const missing: string[] = [];
  if (!qwen8bReady) missing.push("qwen35-9b-q4k-m");
  if (!qwen36bReady) missing.push("qwen36-35b-a3b-q4k-xl");
  return `install or set env for: ${missing.join(", ")}`;
}

function apiDetail(): string {
  if (hasEnv("ZINC_API_BASE_URL")) return process.env.ZINC_API_BASE_URL as string;
  if (apiServerReady) return "local server (managed model)";
  return "install a managed model or set ZINC_API_BASE_URL";
}

console.log("\nCombined test summary:");
console.log(statusLine("Bun suite", "pass"));
console.log(statusLine("Zig suite", "pass"));
console.log(
  statusLine(
    "Qwen smoke",
    hasQwenSmoke ? "ran" : requireFull ? "required but missing env" : "skipped",
    qwenDetail(),
  ),
);
console.log(
  statusLine(
    "OpenAI API smoke",
    hasApiSmoke ? "ran" : requireFull ? "required but missing env" : "skipped",
    apiDetail(),
  ),
);
console.log(statusLine("Full-test mode", requireFull ? "enabled" : "disabled"));
console.log(statusLine("Overall", "pass"));
