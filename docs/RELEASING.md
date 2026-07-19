# Releasing ZINC

ZINC uses semantic versioning with tag-driven releases. The release workflow
(`.github/workflows/release.yml`) already builds, packages, checksums, and
drafts a GitHub Release when a `v*` tag is pushed. This document is the
manual process that feeds it.

## Version policy (semver)

Versions are `MAJOR.MINOR.PATCH`, tags are `vMAJOR.MINOR.PATCH`.

While ZINC is pre-1.0:

- **MINOR** — new features, new supported models or backends, performance
  work, and any breaking change (CLI flags, API shapes, cache layout,
  managed-catalog format). Breaking changes must be called out in the notes.
- **PATCH** — bug fixes and documentation only. No behavior changes a client
  or script could depend on.

From 1.0 on, breaking changes require a MAJOR bump.

## Cutting a release, step by step

1. **Open a release PR** from a branch named `release/vX.Y.Z` containing:
   - `docs/releases/X.Y.Z-notes.md` — the release notes. This file is
     mandatory: the workflow's draft-release job fails without it. Follow the
     structure of `docs/releases/0.1.0-notes.md` (summary, downloads,
     install, checksums, changes, known issues).
   - Any user-facing doc updates that should ship with the release
     (README, getting-started).
   - Nothing else. A release PR must not contain code changes; land those
     first through normal PRs so the release diff is reviewable at a glance.
2. **Merge the release PR** once CI is green.
3. **Tag the merge commit** on `main` and push the tag:

   ```bash
   git checkout main && git pull
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

4. **The Release workflow runs automatically**: it validates (tests + release
   build), builds `linux-x86_64` (Vulkan) and `macos-aarch64` (Metal),
   packages both via `scripts/package_release.sh`, generates
   `SHA256SUMS.txt`, and creates a **draft** GitHub Release with the notes
   file as the body. It refuses to touch an already-published release.
5. **Verify the draft**: download both tarballs, check
   `sha256sum -c SHA256SUMS.txt`, run `bin/zinc --version` (must print
   `X.Y.Z`) and `bin/zinc --check` on at least one machine per platform.
6. **Publish the release** in the GitHub UI.
7. **Verify the installer** end to end on a clean machine:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/zolotukhin/zinc/main/scripts/install.sh | bash
   zinc --version
   ```

## Asset naming contract

`scripts/install.sh` and `scripts/package_release.sh` share this contract;
change them together or not at all:

- Tarball: `zinc-vX.Y.Z-<target>.tar.gz`, containing a single top-level
  directory `zinc-vX.Y.Z-<target>/` with `bin/zinc`,
  `share/zinc/shaders[...]`, `README.md`, `LICENSE`, `VERSION.json`.
- Checksums: `SHA256SUMS.txt` with `sha256sum` format lines, one per tarball.
- Targets: `linux-x86_64`, `macos-aarch64`.

The binary locates shaders relative to its own resolved path
(`../share/zinc/shaders`), so the tree must stay intact after install; the
installer symlinks the binary rather than copying it out of the tree.

## Re-running a failed release build

The workflow supports `workflow_dispatch` with a tag input, so a transient
build failure can be retried from the Actions tab without re-tagging. If the
tag itself pointed at the wrong commit, delete the tag and the draft release,
then re-tag; never re-point a tag that has a published release.
