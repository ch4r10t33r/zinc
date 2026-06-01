#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/package_release.sh --version <version> [options]

Options:
  --version <version>      Release version without leading v, for example 0.1.0
  --target <name>          Artifact target name, for example linux-x86_64
  --backend <name>         Backend name: vulkan or metal
  --prefix <path>          Zig install prefix (default: zig-out)
  --dist <path>            Output directory (default: dist)
  --commit <sha>           Commit hash for VERSION.json (default: git rev-parse HEAD)
  -h, --help               Show this help

The script packages an already-built install tree. It does not build ZINC.
Run zig build first with the matching target/backend/options.
USAGE
}

version=""
target_name=""
backend=""
prefix="zig-out"
dist_dir="dist"
commit=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      version="${2:-}"
      shift 2
      ;;
    --target)
      target_name="${2:-}"
      shift 2
      ;;
    --backend)
      backend="${2:-}"
      shift 2
      ;;
    --prefix)
      prefix="${2:-}"
      shift 2
      ;;
    --dist)
      dist_dir="${2:-}"
      shift 2
      ;;
    --commit)
      commit="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$version" ]]; then
  echo "--version is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "$target_name" ]]; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) target_name="linux-x86_64" ;;
    Darwin-arm64) target_name="macos-aarch64" ;;
    *)
      echo "--target is required on this platform" >&2
      exit 2
      ;;
  esac
fi

if [[ -z "$backend" ]]; then
  case "$target_name" in
    linux-*) backend="vulkan" ;;
    macos-*) backend="metal" ;;
    *)
      echo "--backend is required for target $target_name" >&2
      exit 2
      ;;
  esac
fi

if [[ -z "$commit" ]]; then
  commit="$(git rev-parse --short=12 HEAD)"
fi

case "$backend" in
  vulkan|metal) ;;
  *)
    echo "Unsupported backend: $backend" >&2
    exit 2
    ;;
esac

binary="${prefix}/bin/zinc"
if [[ ! -x "$binary" ]]; then
  echo "Missing executable: $binary" >&2
  echo "Run zig build before packaging." >&2
  exit 1
fi

artifact="zinc-v${version}-${target_name}"
package_dir="${dist_dir}/${artifact}"
archive="${dist_dir}/${artifact}.tar.gz"

rm -rf "$package_dir" "$archive"
mkdir -p "$package_dir/bin" "$package_dir/share/zinc"

cp "$binary" "$package_dir/bin/zinc"
cp README.md LICENSE "$package_dir/"

case "$backend" in
  vulkan)
    shader_src="${prefix}/share/zinc/shaders"
    if [[ ! -d "$shader_src" ]]; then
      echo "Missing compiled shader directory: $shader_src" >&2
      exit 1
    fi
    if ! find "$shader_src" -maxdepth 1 -name '*.spv' -print -quit | grep -q .; then
      echo "No .spv files found in $shader_src" >&2
      exit 1
    fi
    mkdir -p "$package_dir/share/zinc/shaders"
    cp "$shader_src"/*.spv "$package_dir/share/zinc/shaders/"
    ;;
  metal)
    shader_src="${prefix}/share/zinc/shaders/metal"
    if [[ ! -d "$shader_src" ]]; then
      shader_src="src/shaders/metal"
    fi
    if [[ ! -d "$shader_src" ]]; then
      echo "Missing Metal shader directory" >&2
      exit 1
    fi
    if ! find "$shader_src" -maxdepth 1 -name '*.metal' -print -quit | grep -q .; then
      echo "No .metal files found in $shader_src" >&2
      exit 1
    fi
    mkdir -p "$package_dir/share/zinc/shaders/metal"
    cp "$shader_src"/*.metal "$package_dir/share/zinc/shaders/metal/"
    ;;
esac

built_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
cat > "$package_dir/VERSION.json" <<JSON
{
  "name": "zinc",
  "version": "${version}",
  "tag": "v${version}",
  "commit": "${commit}",
  "target": "${target_name}",
  "optimize": "ReleaseFast",
  "backend": "${backend}",
  "built_at": "${built_at}"
}
JSON

tar -C "$dist_dir" -czf "$archive" "$artifact"

echo "$archive"
