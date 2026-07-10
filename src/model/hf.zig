//! Resolve and download Hugging Face GGUF models for the `-hf` CLI flag.
//! @section Managed Models
//! Turns an `owner/repo[:quant]` spec into an installed GGUF in the managed
//! model cache, resolving the concrete file name via the Hugging Face
//! `/v2/<repo>/manifests/<tag>` endpoint and reusing the managed download
//! pipeline for transfer, staging, and manifest bookkeeping.
const std = @import("std");
const catalog = @import("catalog.zig");
const managed = @import("managed.zig");

const manifest_body_limit = 1024 * 1024;
const redirect_buffer_len = 4096;

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
/// network access. Otherwise the GGUF file name is resolved via the Hugging
/// Face manifest endpoint and the file is downloaded through the managed
/// pull pipeline (staged `.partial` file, progress bar, manifest write).
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

    const file_name = try resolveGgufFileName(allocator, spec, writer);
    defer allocator.free(file_name);

    const download_url = try std.fmt.allocPrint(
        allocator,
        "https://huggingface.co/{s}/resolve/main/{s}?download=true",
        .{ spec.repo, file_name },
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
        .file_name = file_name,
        .homepage_url = homepage_url,
        .download_url = download_url,
        .sha256 = "",
        .size_bytes = 0,
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

fn resolveGgufFileName(allocator: std.mem.Allocator, spec: Spec, writer: anytype) ![]u8 {
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

    const file_name = extractGgufFileName(body.items) orelse return error.HfManifestMissingGguf;
    return allocator.dupe(u8, file_name);
}

fn extractGgufFileName(body: []const u8) ?[]const u8 {
    const gguf_pos = std.mem.indexOf(u8, body, "\"ggufFile\"") orelse return null;
    const tail = body[gguf_pos..];
    const key = "\"rfilename\"";
    const key_pos = std.mem.indexOf(u8, tail, key) orelse return null;
    var rest = tail[key_pos + key.len ..];
    var i: usize = 0;
    while (i < rest.len and (rest[i] == ':' or rest[i] == ' ' or rest[i] == '\t')) i += 1;
    if (i >= rest.len or rest[i] != '"') return null;
    rest = rest[i + 1 ..];
    const end = std.mem.indexOfScalar(u8, rest, '"') orelse return null;
    if (end == 0) return null;
    return rest[0..end];
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

test "extractGgufFileName reads rfilename from the ggufFile object" {
    const manifest_body =
        \\{"siblings":[{"rfilename":"README.md"}],"ggufFile":{"rfilename":"Qwen3.5-9B-Q4_K_M.gguf","size":5650000000},"other":{"rfilename":"mmproj.gguf"}}
    ;
    try std.testing.expectEqualStrings("Qwen3.5-9B-Q4_K_M.gguf", extractGgufFileName(manifest_body).?);
}

test "extractGgufFileName returns null when ggufFile is absent" {
    try std.testing.expect(extractGgufFileName("{\"siblings\":[{\"rfilename\":\"README.md\"}]}") == null);
}
