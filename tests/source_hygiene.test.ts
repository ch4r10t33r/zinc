import { expect, test } from "bun:test";
import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";

const repoRoot = resolve(import.meta.dir, "..");
const sourceExts = [".zig", ".cu", ".cuh", ".comp", ".glsl", ".h", ".c", ".metal"];

function trackedSourceFiles(): string[] {
  try {
    const out = execFileSync("git", ["ls-files", "build.zig", "benchmarks", "src"], {
      cwd: repoRoot,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    });
    return out
      .split("\n")
      .filter(Boolean)
      .filter((file) => sourceExts.some((ext) => file.endsWith(ext)));
  } catch {
    return walkSourceFiles(["build.zig", "benchmarks", "src"]);
  }
}

function walkSourceFiles(roots: string[]): string[] {
  const files: string[] = [];
  for (const root of roots) {
    const full = resolve(repoRoot, root);
    const stat = statSync(full);
    if (stat.isFile()) {
      if (sourceExts.some((ext) => root.endsWith(ext))) files.push(root);
      continue;
    }
    for (const entry of readdirSync(full)) {
      files.push(...walkSourceFiles([`${root}/${entry}`]));
    }
  }
  return files;
}

test("Zig, CUDA, and shader source avoid naming the external comparison runtime", () => {
  const banned = new RegExp(`${"llama"}[._ -]?${"cpp"}|${"llama"}${"cpp"}`, "i");
  const offenders = trackedSourceFiles().filter((file) => {
    const text = readFileSync(resolve(repoRoot, file), "utf8");
    return banned.test(text);
  });

  expect(offenders).toEqual([]);
});
