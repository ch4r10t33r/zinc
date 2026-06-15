#!/usr/bin/env bash
# harvest_sweep.sh — classify perf/* branches for harvesting to main by
# PATCH-EQUIVALENCE (git cherry), so a win cherry-picked onto main (different SHA,
# same patch) is correctly seen as already-landed, not a false "unmerged". Using a
# plain `git diff origin/main <branch>` is WRONG: it's non-empty whenever main is
# merely AHEAD of the branch (always true here), not only when the branch carries
# un-landed code.
#
#   scripts/harvest_sweep.sh [prefix]      # default prefix: perf/e27-
#   scripts/harvest_sweep.sh perf/e26-
#
# Per branch, exactly one status (git cherry marks each branch commit '-' = patch
# already in main, '+' = not in main; CODE = a '+' commit touching *.zig/*.cu):
#   MERGED    branch HEAD is an ancestor of origin/main
#   ON-MAIN   every branch commit is patch-equivalent to one already on main — SKIP
#   DOC-ONLY  only un-landed commits are docs/logs (a logged negative) — SKIP
#   HARVEST   an un-landed commit changes code — a candidate win to merge
set -u
PREFIX="${1:-perf/e27-}"
git fetch origin -q --prune 2>/dev/null || true
printf "  %-9s %-36s %s\n" STATUS BRANCH SUBJECT
git branch -r | sed 's/^[* ]*//' | grep -E "^origin/${PREFIX}" | while read -r ref; do
  b="${ref#origin/}"
  subj=$(git log -1 --format='%s' "$ref" 2>/dev/null | cut -c1-50)
  if git merge-base --is-ancestor "$ref" origin/main 2>/dev/null; then
    st="MERGED"
  else
    unlanded=$(git cherry origin/main "$ref" 2>/dev/null | sed -n 's/^+ //p')
    if [ -z "$unlanded" ]; then
      st="ON-MAIN"
    else
      st="DOC-ONLY"
      for c in $unlanded; do
        if git show --name-only --format= "$c" 2>/dev/null | grep -qE '\.(zig|cu)$'; then st="HARVEST"; break; fi
      done
    fi
  fi
  printf "  %-9s %-36s %s\n" "$st" "$b" "$subj"
done
echo "  (HARVEST = un-landed code win → merge; ON-MAIN/DOC-ONLY/MERGED = skip)"
