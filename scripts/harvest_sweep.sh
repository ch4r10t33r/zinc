#!/usr/bin/env bash
# harvest_sweep.sh — classify perf/* branches for harvesting to main by CONTENT,
# not just SHA ancestry. A win that was cherry-picked onto main has a different SHA
# (so `git merge-base --is-ancestor` calls it "unmerged") but an EMPTY code diff vs
# main — this tool reports that correctly so already-landed wins aren't re-harvested.
#
#   scripts/harvest_sweep.sh [prefix]      # default prefix: perf/e27-
#   scripts/harvest_sweep.sh perf/e26-
#
# Per branch it prints exactly one status (CODE = src/ *.zig/*.cu):
#   MERGED    branch HEAD is an ancestor of origin/main (fully merged)
#   ON-MAIN   code diff vs origin/main is EMPTY (already landed, e.g. cherry-picked,
#             or a tried-then-reverted negative whose tree returned to main) — SKIP
#   DOC-ONLY  diff touches no .zig/.cu (a logged negative / doc-only branch) — SKIP
#   HARVEST   real un-landed code change vs origin/main — a candidate win to merge
set -u
PREFIX="${1:-perf/e27-}"
git fetch origin -q --prune 2>/dev/null || true
printf "  %-9s %-36s %s\n" STATUS BRANCH SUBJECT
git branch -r | sed 's/^[* ]*//' | grep -E "^origin/${PREFIX}" | while read -r ref; do
  b="${ref#origin/}"
  subj=$(git log -1 --format='%s' "$ref" 2>/dev/null | cut -c1-50)
  if git merge-base --is-ancestor "$ref" origin/main 2>/dev/null; then
    st="MERGED"
  elif [ "$(git diff origin/main "$ref" -- 'src/' 2>/dev/null | wc -l | tr -d ' ')" = "0" ]; then
    st="ON-MAIN"
  elif git diff --name-only origin/main "$ref" 2>/dev/null | grep -qE '\.(zig|cu)$'; then
    st="HARVEST"
  else
    st="DOC-ONLY"
  fi
  printf "  %-9s %-36s %s\n" "$st" "$b" "$subj"
done
echo "  (HARVEST = merge to main; ON-MAIN/DOC-ONLY/MERGED = skip. CODE scope = src/ *.zig *.cu)"
