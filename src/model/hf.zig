//! Resolve and download Hugging Face GGUF models for the `-hf` CLI flag.
//! @section Managed Models
//! Turns an `owner/repo[:quant]` spec into an installed GGUF in the managed
//! model cache, resolving the concrete file name via the Hugging Face
//! `/v2/<repo>/manifests/<tag>` endpoint and reusing the managed download
//! pipeline for transfer, staging, and manifest bookkeeping.
const std = @import("std");
const catalog = @import("catalog.zig");
const managed = @import("managed.zig");

// The ollama-style manifest is a small bounded document (config + layers +
// ggufFile), typically ~1 KiB; the cap only guards against a misbehaving
// server.
const manifest_body_limit = 4 * 1024 * 1024;
const redirect_buffer_len = 4096;

/// GGUF file metadata resolved from the Hugging Face manifest endpoint.
const ResolvedFile = struct {
    /// Repo-relative GGUF file name (`ggufFile.rfilename`).
    file_name: []u8,
    /// Lowercase sha256 hex of the file (`ggufFile.lfs.sha256`), or null
    /// when the manifest does not carry a usable digest.
    sha256: ?[]u8,
    /// File size in bytes, or 0 when unknown.
    size_bytes: u64,

    fn deinit(self: *ResolvedFile, allocator: std.mem.Allocator) void {
        allocator.free(self.file_name);
        if (self.sha256) |s| allocator.free(s);
        self.* = undefined;
    }
};

/// Parsed `-hf` argument: a Hugging Face repo plus a quantization tag.
pub const Spec = struct {
    /// Repo in `owner/name` form, e.g. `unsloth/Qwen3.5-9B-GGUF`.
    repo: []const u8,
    /// Quantization tag (e.g. `Q4_K_M`), or `latest` to let Hugging Face
    /// pick the repo's default quantization.
    tag: []const u8,
};

/// Parses an `-hf` argument of the form `owner/repo[:quant]`.
///
/// The quantization suffix is optional; when absent the tag defaults to
/// `latest`, which the Hugging Face manifest endpoint resolves to the repo's
/// recommended quantization.
/// @param text Raw CLI argument value.
/// @returns A `Spec` whose slices point into `text`.
pub fn parseSpec(text: []const u8) !Spec {
    var repo = text;
    var tag: []const u8 = "latest";
    if (std.mem.lastIndexOfScalar(u8, text, ':')) |pos| {
        repo = text[0..pos];
        tag = text[pos + 1 ..];
        if (tag.len == 0) return error.InvalidHfSpec;
    }
    const slash = std.mem.indexOfScalar(u8, repo, '/') orelse return error.InvalidHfSpec;
    if (slash == 0 or slash == repo.len - 1) return error.InvalidHfSpec;
    if (std.mem.indexOfScalarPos(u8, repo, slash + 1, '/') != null) return error.InvalidHfSpec;
    return .{ .repo = repo, .tag = tag };
}

/// Derives the managed-cache model id for a Hugging Face spec.
///
/// The id is a single filesystem-safe path component so the download can
/// live in the same `<cache_root>/models/<id>/model.gguf` layout as catalog
/// models. Distinct specs map to distinct ids; `:latest` is cached
/// separately from an explicit quantization even if both resolve to the
/// same upstream file.
/// @param allocator Owns the returned id slice.
/// @param spec Parsed Hugging Face spec.
/// @returns Heap-allocated id like `hf--unsloth--qwen3.5-9b-gguf--q4_k_m`.
pub fn cacheId(allocator: std.mem.Allocator, spec: Spec) ![]u8 {
    const raw = try std.fmt.allocPrint(allocator, "hf--{s}--{s}", .{ spec.repo, spec.tag });
    for (raw) |*c| {
        c.* = std.ascii.toLower(c.*);
        switch (c.*) {
            'a'...'z', '0'...'9', '.', '_', '-' => {},
            else => c.* = '-',
        }
    }
    return raw;
}

/// Ensures the model described by an `-hf` spec is installed and returns its path.
///
/// If the spec is already cached the installed path is returned without any
/// network access. Otherwise the GGUF file name and sha256 digest are
/// resolved via the Hugging Face manifest endpoint and the file is
/// downloaded through the managed pull pipeline (staged `.partial` file,
/// progress bar, manifest write), which verifies the download against the
/// pinned digest when the manifest provides one.
/// @param spec_text Raw `-hf` argument (`owner/repo[:quant]`).
/// @param allocator Used for HTTP, path construction, and the returned path.
/// @param writer Receives human-readable status lines and download progress.
/// @returns Heap-allocated absolute path to the installed GGUF; caller owns it.
/// @note Gated or private repos are not supported: they fail with `error.HfAuthRequired`.
pub fn ensureModel(spec_text: []const u8, allocator: std.mem.Allocator, writer: anytype) ![]u8 {
    const spec = try parseSpec(spec_text);
    const id = try cacheId(allocator, spec);
    defer allocator.free(id);

    if (managed.isInstalled(id, allocator)) {
        const path = try managed.resolveInstalledModelPath(id, allocator);
        try writer.print("Using cached Hugging Face model: {s}\n", .{path});
        try writer.flush();
        return path;
    }

    var resolved = try resolveGgufFile(allocator, spec, writer);
    defer resolved.deinit(allocator);

    const download_url = try std.fmt.allocPrint(
        allocator,
        "https://huggingface.co/{s}/resolve/main/{s}?download=true",
        .{ spec.repo, resolved.file_name },
    );
    defer allocator.free(download_url);
    const homepage_url = try std.fmt.allocPrint(allocator, "https://huggingface.co/{s}", .{spec.repo});
    defer allocator.free(homepage_url);

    const entry = catalog.CatalogEntry{
        .id = id,
        .display_name = spec_text,
        .release_date = "",
        .family = "huggingface",
        .format = "gguf",
        .quantization = spec.tag,
        .file_name = resolved.file_name,
        .homepage_url = homepage_url,
        .download_url = download_url,
        // Pinning the manifest digest makes the pull pipeline verify the
        // pre-download x-linked-etag and the post-download file hash.
        .sha256 = resolved.sha256 orelse "",
        .size_bytes = resolved.size_bytes,
        .required_vram_bytes = 0,
        .default_context_length = 4096,
        .recommended_for_chat = false,
        .thinking_stable = false,
        .status = .experimental,
        .tested_profiles = &.{},
    };

    try managed.pullModelWithObserver(entry, allocator, writer, null);
    return managed.resolveInstalledModelPath(id, allocator);
}

fn resolveGgufFile(allocator: std.mem.Allocator, spec: Spec, writer: anytype) !ResolvedFile {
    const manifest_url = try std.fmt.allocPrint(
        allocator,
        "https://huggingface.co/v2/{s}/manifests/{s}",
        .{ spec.repo, spec.tag },
    );
    defer allocator.free(manifest_url);

    try writer.print("Resolving Hugging Face model: {s} (tag: {s})\n", .{ spec.repo, spec.tag });
    try writer.flush();

    var client: std.http.Client = .{ .allocator = allocator };
    defer client.deinit();

    const uri = try std.Uri.parse(manifest_url);
    // Hugging Face only includes the resolved `ggufFile` object in the
    // manifest response when the User-Agent contains "llama-cpp". The typed
    // header slot must be overridden; an extra_headers entry would be sent
    // in addition to the client's default `zig/x.y.z (std.http)` agent, and
    // Hugging Face matches on the first user-agent header. accept-encoding
    // is forced to identity because `response.reader` does not decompress.
    var req = try client.request(.GET, uri, .{
        .headers = .{
            .user_agent = .{ .override = "zinc (llama-cpp compatible)" },
            .accept_encoding = .{ .override = "identity" },
        },
        .extra_headers = &.{
            .{ .name = "accept", .value = "application/json" },
        },
    });
    defer req.deinit();
    try req.sendBodiless();

    var redirect_buffer: [redirect_buffer_len]u8 = undefined;
    var response = try req.receiveHead(&redirect_buffer);
    switch (response.head.status.class()) {
        .success => {},
        else => switch (response.head.status) {
            .unauthorized, .forbidden => return error.HfAuthRequired,
            .not_found => return error.HfModelNotFound,
            else => return error.HfManifestFailed,
        },
    }

    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    const content_length = response.head.content_length;
    var transfer_buffer: [4096]u8 = undefined;
    var reader = response.reader(&transfer_buffer);
    var chunk: [4096]u8 = undefined;
    while (true) {
        // Guard: stop once all expected bytes are received.
        // Zig 0.15 std.http panics if we read past the content-length boundary.
        if (content_length) |total| {
            if (body.items.len >= total) break;
        }
        if (body.items.len >= manifest_body_limit) return error.HfManifestFailed;
        const n = reader.readSliceShort(&chunk) catch |err| switch (err) {
            error.ReadFailed => return response.bodyErr().?,
        };
        if (n == 0) break;
        try body.appendSlice(allocator, chunk[0..n]);
    }

    return parseManifestBody(allocator, body.items);
}

fn parseManifestBody(allocator: std.mem.Allocator, body: []const u8) !ResolvedFile {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch
        return error.HfManifestFailed;
    defer parsed.deinit();

    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.HfManifestFailed,
    };
    const gguf = switch (root.get("ggufFile") orelse return error.HfManifestMissingGguf) {
        .object => |o| o,
        else => return error.HfManifestMissingGguf,
    };
    const file_name = switch (gguf.get("rfilename") orelse return error.HfManifestMissingGguf) {
        .string => |s| s,
        else => return error.HfManifestMissingGguf,
    };
    if (file_name.len == 0) return error.HfManifestMissingGguf;

    var size_bytes: u64 = 0;
    if (gguf.get("size")) |v| switch (v) {
        .integer => |n| {
            if (n > 0) size_bytes = @intCast(n);
        },
        else => {},
    };

    var sha256: ?[]u8 = null;
    errdefer if (sha256) |s| allocator.free(s);
    if (gguf.get("lfs")) |lfs_value| switch (lfs_value) {
        .object => |lfs| {
            if (lfs.get("sha256")) |v| switch (v) {
                .string => |s| {
                    if (isSha256Hex(s)) {
                        const owned = try allocator.dupe(u8, s);
                        for (owned) |*c| c.* = std.ascii.toLower(c.*);
                        sha256 = owned;
                    }
                },
                else => {},
            };
        },
        else => {},
    };

    return .{
        .file_name = try allocator.dupe(u8, file_name),
        .sha256 = sha256,
        .size_bytes = size_bytes,
    };
}

fn isSha256Hex(s: []const u8) bool {
    if (s.len != 64) return false;
    for (s) |c| {
        if (!std.ascii.isHex(c)) return false;
    }
    return true;
}

test "parseSpec accepts repo without tag" {
    const spec = try parseSpec("unsloth/Qwen3.5-9B-GGUF");
    try std.testing.expectEqualStrings("unsloth/Qwen3.5-9B-GGUF", spec.repo);
    try std.testing.expectEqualStrings("latest", spec.tag);
}

test "parseSpec splits quantization tag" {
    const spec = try parseSpec("unsloth/Qwen3.5-9B-GGUF:Q4_K_M");
    try std.testing.expectEqualStrings("unsloth/Qwen3.5-9B-GGUF", spec.repo);
    try std.testing.expectEqualStrings("Q4_K_M", spec.tag);
}

test "parseSpec rejects malformed specs" {
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("no-slash"));
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("/leading"));
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("trailing/"));
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("a/b/c"));
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("a/b:"));
}

test "cacheId is a lowercase single path component" {
    const spec = try parseSpec("unsloth/Qwen3.5-9B-GGUF:Q4_K_M");
    const id = try cacheId(std.testing.allocator, spec);
    defer std.testing.allocator.free(id);
    try std.testing.expectEqualStrings("hf--unsloth-qwen3.5-9b-gguf--q4_k_m", id);
    try std.testing.expect(std.mem.indexOfScalar(u8, id, '/') == null);
}

test "parseManifestBody reads file name, digest, and size from ggufFile" {
    const manifest_body =
        \\{"siblings":[{"rfilename":"README.md"}],"ggufFile":{"rfilename":"Qwen3.5-9B-Q4_K_M.gguf","size":5650000000,"lfs":{"sha256":"9465E63A22ADD5354D9BB4B99E90117043C7124007664907259BD16D043BB031","size":5650000000}},"other":{"rfilename":"mmproj.gguf"}}
    ;
    var resolved = try parseManifestBody(std.testing.allocator, manifest_body);
    defer resolved.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("Qwen3.5-9B-Q4_K_M.gguf", resolved.file_name);
    try std.testing.expectEqualStrings(
        "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
        resolved.sha256.?,
    );
    try std.testing.expectEqual(@as(u64, 5_650_000_000), resolved.size_bytes);
}

test "parseManifestBody tolerates a missing or malformed digest" {
    const manifest_body =
        \\{"ggufFile":{"rfilename":"model.gguf","size":-5,"lfs":{"sha256":"not-a-digest"}}}
    ;
    var resolved = try parseManifestBody(std.testing.allocator, manifest_body);
    defer resolved.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("model.gguf", resolved.file_name);
    try std.testing.expectEqual(@as(?[]u8, null), resolved.sha256);
    try std.testing.expectEqual(@as(u64, 0), resolved.size_bytes);
}

test "parseManifestBody rejects manifests without a ggufFile" {
    try std.testing.expectError(
        error.HfManifestMissingGguf,
        parseManifestBody(std.testing.allocator, "{\"siblings\":[{\"rfilename\":\"README.md\"}]}"),
    );
}

test "parseManifestBody rejects non-JSON bodies" {
    try std.testing.expectError(
        error.HfManifestFailed,
        parseManifestBody(std.testing.allocator, "<html>rate limited</html>"),
    );
}
