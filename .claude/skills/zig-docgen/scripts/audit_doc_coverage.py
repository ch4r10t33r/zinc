#!/usr/bin/env python3
"""Audit Zig source files against the docgen contract documented in
.claude/skills/zig-docgen/SKILL.md.

Reports four classes of issue:
  1. file_no_module_doc   — file missing the leading //! summary
  2. file_no_section      — file has //! header but no //! @section line
  3. symbol_no_doc        — top-level pub fn/const/var with no /// doc above
  4. symbol_no_summary    — has /// but the first line is empty/missing
  5. method_no_doc        — pub fn inside an exported container with no /// above
  6. param_undocumented   — function with params but no @param tags (info only)

Heuristic Zig parser — purposely conservative. Excludes vk.zig and test files.
"""
from __future__ import annotations
import os, re, json, sys
from pathlib import Path

ROOT = Path("src")
# Mirror EXCLUDED_MODULES in site/src/lib/zig-api-loader.ts.
EXCLUDE_FILES = {"src/vulkan/vk.zig", "src/regression_tests.zig"}
# Skip generated artifacts and tools that aren't part of the public Zig API.
EXCLUDE_DIR_PARTS = {".zig-api-cache", "shaders", "metal"}


def is_excluded(path: Path) -> bool:
    rel = str(path).replace(os.sep, "/")
    if rel in EXCLUDE_FILES:
        return True
    parts = path.parts
    return any(p in EXCLUDE_DIR_PARTS for p in parts)

PUB_TOP = re.compile(r"^(pub\s+(?:fn|const|var)\s+)([A-Za-z_]\w*)")
# Methods inside an exported container — indented pub fn (4-space convention).
PUB_METHOD = re.compile(r"^(\s+)(pub\s+fn\s+)([A-Za-z_]\w*)")
# Anything that opens a documentable container (struct/enum/union assigned to a pub const).
EXPORTED_CONTAINER = re.compile(r"^pub\s+const\s+([A-Za-z_]\w*)\s*=\s*(?:extern\s+|packed\s+)?(?:struct|enum|union)\s*\(?")

def issues_for_file(path: Path) -> list[dict]:
    if is_excluded(path):
        return []
    rel = str(path).replace(os.sep, "/")
    text = path.read_text()
    lines = text.splitlines()
    out = []

    # ── 1. Module doc comment ────────────────────────────────────────
    # Find first non-empty line.
    first_idx = next((i for i, ln in enumerate(lines) if ln.strip()), -1)
    if first_idx < 0:
        return []
    if not lines[first_idx].lstrip().startswith("//!"):
        out.append({"kind": "file_no_module_doc", "path": rel, "line": 1})
    else:
        # Walk contiguous //! block; check for @section.
        block = []
        i = first_idx
        while i < len(lines) and lines[i].lstrip().startswith("//!"):
            block.append(lines[i])
            i += 1
        if not any("@section" in ln for ln in block):
            out.append({"kind": "file_no_section", "path": rel, "line": first_idx + 1})

    # ── 2. Walk tokens for pub declarations ──────────────────────────
    in_test = False
    container_depth = 0  # tracking exported container nesting (for pub method check)
    container_stack: list[bool] = []  # per-brace level: is_in_exported_container
    brace_depth = 0
    i = 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.lstrip()

        # Track brace depth on this line (very approximate, but good enough for
        # well-formatted Zig where braces are balanced per-line groups).
        # We don't try to be perfect — just enough to know whether we're at
        # top level vs inside a container.
        opens = stripped.count("{") - stripped.count("}")
        # Treat any pub container declaration on this line as opening container.
        if EXPORTED_CONTAINER.match(stripped):
            # Push new container frame to be popped when its braces close.
            # This is approximate.
            container_stack.append(True)
        elif opens > 0 and container_stack:
            # Nested struct/enum inside container — push False (not directly exported).
            for _ in range(opens):
                container_stack.append(False)
        # Pop frames for closing braces.
        for _ in range(max(0, -opens)):
            if container_stack:
                container_stack.pop()

        # Test functions are ok to skip.
        if re.match(r"^test\s+\"", stripped):
            in_test = True

        # Top-level pub declarations (column 0).
        m_top = PUB_TOP.match(ln)
        if m_top:
            name = m_top.group(2)
            doc_lines = collect_doc_above(lines, i)
            if not doc_lines:
                out.append({"kind": "symbol_no_doc", "path": rel, "line": i + 1, "name": name})
            else:
                if not any(stripline.strip().startswith("///") and stripline.strip().lstrip("/").strip() for stripline in doc_lines):
                    out.append({"kind": "symbol_no_summary", "path": rel, "line": i + 1, "name": name})

        # Indented pub fn methods inside exported containers.
        m_method = PUB_METHOD.match(ln)
        if m_method and any(container_stack):
            name = m_method.group(3)
            doc_lines = collect_doc_above(lines, i)
            if not doc_lines:
                out.append({"kind": "method_no_doc", "path": rel, "line": i + 1, "name": name})

        i += 1

    return out


def collect_doc_above(lines: list[str], idx: int) -> list[str]:
    """Walk upward collecting consecutive /// lines (skipping blank lines is NOT allowed)."""
    j = idx - 1
    block: list[str] = []
    while j >= 0:
        s = lines[j].strip()
        if s.startswith("///"):
            block.insert(0, lines[j])
            j -= 1
        else:
            break
    return block


def main() -> int:
    issues: list[dict] = []
    for path in sorted(ROOT.rglob("*.zig")):
        issues.extend(issues_for_file(path))

    by_kind: dict[str, int] = {}
    for it in issues:
        by_kind[it["kind"]] = by_kind.get(it["kind"], 0) + 1

    print(f"Total issues: {len(issues)}")
    print("By kind:")
    for k in sorted(by_kind):
        print(f"  {k}: {by_kind[k]}")
    print()

    print("Files missing //! header:")
    for it in issues:
        if it["kind"] == "file_no_module_doc":
            print(f"  {it['path']}")
    print()

    print("Files missing @section:")
    for it in issues:
        if it["kind"] == "file_no_section":
            print(f"  {it['path']}")
    print()

    print("Top 30 undocumented top-level public symbols:")
    syms = [it for it in issues if it["kind"] == "symbol_no_doc"]
    for it in syms[:30]:
        print(f"  {it['path']}:{it['line']}  {it['name']}")
    if len(syms) > 30:
        print(f"  ... {len(syms) - 30} more")
    print()

    print("Top 20 undocumented public methods:")
    methods = [it for it in issues if it["kind"] == "method_no_doc"]
    for it in methods[:20]:
        print(f"  {it['path']}:{it['line']}  {it['name']}")
    if len(methods) > 20:
        print(f"  ... {len(methods) - 20} more")

    json_out = "/tmp/zig_doc_audit.json"
    with open(json_out, "w") as f:
        json.dump(issues, f, indent=2)
    print(f"\nFull report: {json_out} ({len(issues)} entries)")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
