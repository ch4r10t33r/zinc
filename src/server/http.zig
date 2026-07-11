//! Minimal HTTP/1.1 server for the OpenAI-compatible inference API.
//! @section API Server
//! Provides connection-level request parsing, JSON and SSE response helpers,
//! and a TCP listener that hands off accepted connections to the route dispatcher.
const std = @import("std");

const log = std.log.scoped(.http);

/// Hard ceiling on a request body once it no longer fits the inline header
/// buffer. Generous for large multi-turn chat histories and long prompts on
/// the biggest supported context lengths, while still bounding worst-case
/// per-connection memory against a hostile Content-Length.
const max_overflow_body_bytes = 16 * 1024 * 1024;

/// Active client connection with request/response capabilities.
/// Wraps a TCP stream and provides helpers for reading HTTP requests
/// and writing JSON, error, and SSE responses.
pub const Connection = struct {
    /// Underlying TCP stream to the client.
    stream: std.net.Stream,
    /// Allocator used for per-connection allocations.
    allocator: std.mem.Allocator,
    /// Internal read buffer for request parsing (64 KiB). Holds headers plus
    /// the body for the common case; larger bodies spill into `overflow_body`.
    read_buf: [65536]u8 = undefined,
    /// Number of valid bytes currently in read_buf.
    read_len: usize = 0,
    /// Heap-allocated body buffer used only when Content-Length exceeds what
    /// remains of `read_buf`. Freed by `close`.
    overflow_body: ?[]u8 = null,

    /// Read and parse an HTTP/1.1 request from the connection.
    /// Reads headers until `\r\n\r\n`, extracts method/path/Content-Length,
    /// then reads the remaining body bytes.
    /// @param self Active connection to read from.
    /// @returns Parsed request with method, path, and body.
    pub fn readRequest(self: *Connection) !Request {
        // Read data until we find \r\n\r\n (end of headers)
        var total: usize = 0;
        var header_end: ?usize = null;
        while (total < self.read_buf.len) {
            const n = self.stream.read(self.read_buf[total..]) catch |err| {
                if (total > 0) break;
                return err;
            };
            if (n == 0) break; // connection closed
            total += n;
            // Search for header terminator
            if (total >= 4) {
                const search_start = if (total > n + 3) total - n - 3 else 0;
                for (search_start..total - 3) |i| {
                    if (std.mem.eql(u8, self.read_buf[i .. i + 4], "\r\n\r\n")) {
                        header_end = i + 4;
                        break;
                    }
                }
                if (header_end != null) break;
            }
        }
        self.read_len = total;
        const hdr_end = header_end orelse return error.MalformedRequest;
        const header_str = self.read_buf[0..hdr_end];

        // Parse request line: METHOD PATH HTTP/1.1\r\n
        const first_line_end = std.mem.indexOf(u8, header_str, "\r\n") orelse return error.MalformedRequest;
        const request_line = header_str[0..first_line_end];

        // Split by spaces
        var method: Method = .UNKNOWN;
        var path: []const u8 = "/";
        var part: usize = 0;
        var start: usize = 0;
        for (request_line, 0..) |c, i| {
            if (c == ' ' or i == request_line.len - 1) {
                const end = if (c == ' ') i else i + 1;
                const token = request_line[start..end];
                if (part == 0) {
                    if (std.mem.eql(u8, token, "GET")) method = .GET else if (std.mem.eql(u8, token, "POST")) method = .POST else if (std.mem.eql(u8, token, "OPTIONS")) method = .OPTIONS;
                } else if (part == 1) {
                    path = token;
                }
                part += 1;
                start = i + 1;
            }
        }

        // Extract Content-Length
        var content_length: usize = 0;
        var line_start: usize = first_line_end + 2;
        while (line_start < hdr_end) {
            const line_end = std.mem.indexOf(u8, header_str[line_start..], "\r\n") orelse break;
            const line = header_str[line_start .. line_start + line_end];
            if (line.len > 16 and (line[0] == 'C' or line[0] == 'c')) {
                // Case-insensitive Content-Length check
                const lower = "content-length: ";
                if (line.len > lower.len) {
                    var matches = true;
                    for (lower, 0..) |lc, ci| {
                        const hc = if (line[ci] >= 'A' and line[ci] <= 'Z') line[ci] + 32 else line[ci];
                        if (hc != lc) {
                            matches = false;
                            break;
                        }
                    }
                    if (matches) {
                        content_length = std.fmt.parseInt(usize, std.mem.trim(u8, line[lower.len..], " \t"), 10) catch 0;
                    }
                }
            }
            line_start += line_end + 2;
        }

        // Read remaining body if needed
        const body_already = total - hdr_end;
        if (body_already < content_length) {
            const remaining = content_length - body_already;
            const buf_remaining = self.read_buf.len - total;
            if (remaining > buf_remaining) {
                // Body doesn't fit what's left of the inline buffer: spill
                // the whole body into a heap allocation instead of failing
                // outright, bounded by max_overflow_body_bytes.
                if (content_length > max_overflow_body_bytes) return error.RequestTooLarge;
                const owned = try self.allocator.alloc(u8, content_length);
                errdefer self.allocator.free(owned);
                @memcpy(owned[0..body_already], self.read_buf[hdr_end..total]);
                var read_so_far: usize = body_already;
                while (read_so_far < content_length) {
                    const n = try self.stream.read(owned[read_so_far..]);
                    if (n == 0) break;
                    read_so_far += n;
                }
                if (read_so_far < content_length) return error.MalformedRequest;
                self.overflow_body = owned;
                return Request{ .method = method, .path = path, .body = owned };
            }
            var read_so_far: usize = 0;
            while (read_so_far < remaining) {
                const n = try self.stream.read(self.read_buf[total + read_so_far .. total + read_so_far + remaining - read_so_far]);
                if (n == 0) break;
                read_so_far += n;
            }
            self.read_len = total + read_so_far;
            // A closed connection before all declared body bytes arrived
            // would otherwise return a body slice padded with uninitialized
            // read_buf memory ([65536]u8 = undefined, never zeroed).
            if (read_so_far < remaining) return error.MalformedRequest;
        }

        const body = if (content_length > 0) self.read_buf[hdr_end .. hdr_end + content_length] else "";
        return Request{ .method = method, .path = path, .body = body };
    }

    /// Send a JSON response with the given HTTP status code.
    /// @param self Active connection to write to.
    /// @param status HTTP status code (200, 400, 404, etc.).
    /// @param body JSON-encoded response body.
    pub fn sendJson(self: *Connection, status: u16, body: []const u8) !void {
        var buf: [512]u8 = undefined;
        const status_text = switch (status) {
            200 => "OK",
            202 => "Accepted",
            400 => "Bad Request",
            404 => "Not Found",
            409 => "Conflict",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            else => "OK",
        };
        const header = std.fmt.bufPrint(&buf, "HTTP/1.1 {d} {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", .{ status, status_text, body.len }) catch return error.HeaderTooLarge;
        try self.stream.writeAll(header);
        try self.stream.writeAll(body);
    }

    /// Send an OpenAI-format JSON error response.
    /// @param self Active connection to write to.
    /// @param status HTTP status code for the error.
    /// @param err_type OpenAI error type string (e.g. "invalid_request_error").
    /// @param message Human-readable error message.
    pub fn sendError(self: *Connection, status: u16, err_type: []const u8, message: []const u8) !void {
        var buf: [2048]u8 = undefined;
        const body = std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"{s}\",\"code\":{d}}}}}", .{ message, err_type, status }) catch return error.HeaderTooLarge;
        try self.sendJson(status, body);
    }

    /// Send SSE streaming response headers with chunked transfer encoding.
    /// After this call, use writeSseEvent to send individual events.
    /// @param self Active connection to write to.
    pub fn sendSseStart(self: *Connection) !void {
        const header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: keep-alive\r\nTransfer-Encoding: chunked\r\n\r\n";
        try self.stream.writeAll(header);
    }

    /// Write a single SSE event as a chunked transfer-encoding frame.
    /// @param self Active connection to write to.
    /// @param data Event payload (typically JSON), sent as `data: {payload}\n\n`.
    pub fn writeSseEvent(self: *Connection, data: []const u8) !void {
        // Chunked format: {size_hex}\r\n{data}\r\n
        var size_buf: [16]u8 = undefined;
        // data: {json}\n\n = data.len + "data: ".len + "\n\n".len
        const event_prefix = "data: ";
        const event_suffix = "\n\n";
        const chunk_len = event_prefix.len + data.len + event_suffix.len;
        const size_str = std.fmt.bufPrint(&size_buf, "{x}\r\n", .{chunk_len}) catch unreachable;
        try self.stream.writeAll(size_str);
        try self.stream.writeAll(event_prefix);
        try self.stream.writeAll(data);
        try self.stream.writeAll(event_suffix);
        try self.stream.writeAll("\r\n");
    }

    /// Write a chunked SSE comment line (`: {text}\n\n`) to keep the connection alive.
    /// Useful when a client or intermediate proxy would time out on a quiet
    /// stream before the next model token is ready.
    /// @param self Active connection to write to.
    /// @param text Comment payload sent verbatim after the `: ` prefix.
    pub fn writeSseComment(self: *Connection, text: []const u8) !void {
        var size_buf: [16]u8 = undefined;
        const chunk_len = ": ".len + text.len + "\n\n".len;
        const size_str = std.fmt.bufPrint(&size_buf, "{x}\r\n", .{chunk_len}) catch unreachable;
        try self.stream.writeAll(size_str);
        try self.stream.writeAll(": ");
        try self.stream.writeAll(text);
        try self.stream.writeAll("\n\n\r\n");
    }

    /// Write the final SSE `[DONE]` event and send the chunked transfer terminator.
    /// @param self Active connection to write to.
    pub fn writeSseDone(self: *Connection) !void {
        try self.writeSseEvent("[DONE]");
        // Chunked terminator: 0\r\n\r\n
        try self.stream.writeAll("0\r\n\r\n");
    }

    /// Check whether the remote connection is definitively dead, to let a
    /// streaming decode loop bail out before the next token instead of
    /// discovering the same thing on the next write.
    ///
    /// This only reports `true` on an unambiguous signal: a TCP reset,
    /// refusal, or an already-disconnected socket. A zero-byte peek (the
    /// peer's write side reached EOF) is deliberately treated as "not
    /// closed": HTTP/1.1 clients commonly half-close the upload side right
    /// after sending the request body while still reading the response, so
    /// that signal cannot distinguish "client hung up" from "compliant
    /// client, still receiving the SSE stream". Bailing out on it would
    /// truncate valid streams for such clients — worse than the wasted
    /// decode time this function is meant to save. Only a write failure
    /// (`catch return` at each write-path call site) proves that case.
    /// @param self Active connection to inspect.
    /// @returns `true` only for a hard reset/refused/disconnected socket.
    pub fn isPeerClosed(self: *Connection) bool {
        var probe: [1]u8 = undefined;
        _ = std.posix.recv(self.stream.handle, &probe, std.posix.MSG.PEEK | std.posix.MSG.DONTWAIT) catch |err| {
            return switch (err) {
                error.ConnectionResetByPeer, error.ConnectionRefused, error.SocketNotConnected => true,
                // No data pending (the common case on a healthy connection)
                // or any other unexpected error: default to "not closed",
                // matching this function's conservative contract.
                error.WouldBlock => false,
                else => false,
            };
        };
        return false;
    }

    /// Close the underlying TCP stream and free any heap-allocated overflow
    /// body from `readRequest`.
    /// @param self Connection to close.
    pub fn close(self: *Connection) void {
        if (self.overflow_body) |body| self.allocator.free(body);
        self.stream.close();
    }
};

/// HTTP request method parsed from the request line.
/// `UNKNOWN` is the fallback for any method not explicitly handled by the server.
pub const Method = enum { GET, POST, OPTIONS, UNKNOWN };

/// Parsed HTTP request produced by `Connection.readRequest`.
/// @note `body` and `path` are slices into the owning `Connection`'s internal
/// read buffer, or (for a body too large for that buffer) a heap allocation
/// owned by the `Connection` and freed by its `close`. Either way the slice
/// is only valid until the next call on that connection.
pub const Request = struct {
    method: Method,
    path: []const u8,
    body: []const u8,
};

/// HTTP server that binds and listens on a TCP port.
/// Accepts connections and wraps them in Connection structs for request handling.
pub const Server = struct {
    /// Underlying TCP listener.
    listener: std.net.Server,
    /// Allocator passed to accepted connections.
    allocator: std.mem.Allocator,

    /// Bind to all interfaces on the given port and start listening.
    /// @param allocator Allocator for connection resources.
    /// @param port TCP port to listen on.
    /// @returns A Server ready to accept connections.
    pub fn init(allocator: std.mem.Allocator, port: u16) !Server {
        const address = std.net.Address.initIp4(.{ 0, 0, 0, 0 }, port);
        const listener = try address.listen(.{ .reuse_address = true });
        return Server{ .listener = listener, .allocator = allocator };
    }

    /// Block until a client connects, then return a Connection for that client.
    /// @param self Active server to accept on.
    /// @returns A new Connection wrapping the accepted TCP stream.
    pub fn accept(self: *Server) !Connection {
        const conn = try self.listener.accept();
        return Connection{ .stream = conn.stream, .allocator = self.allocator };
    }

    /// Stop listening and release the server socket.
    /// @param self Server to tear down.
    pub fn deinit(self: *Server) void {
        self.listener.deinit();
    }
};

/// POSIX `struct linger` layout (matches Linux and Darwin's BSD-socket ABI).
const Linger = extern struct {
    l_onoff: i32,
    l_linger: i32,
};

fn loopbackPair() !struct { server: std.net.Stream, client: std.net.Stream } {
    var listener = try std.net.Address.listen(
        std.net.Address.initIp4(.{ 127, 0, 0, 1 }, 0),
        .{ .reuse_address = true },
    );
    defer listener.deinit();

    const client = try std.net.tcpConnectToAddress(listener.listen_address);
    const accepted = try listener.accept();
    return .{ .server = accepted.stream, .client = client };
}

test "isPeerClosed returns false on a healthy idle connection" {
    const pair = try loopbackPair();
    defer pair.server.close();
    defer pair.client.close();

    var conn = Connection{ .stream = pair.server, .allocator = std.testing.allocator };
    try std.testing.expect(!conn.isPeerClosed());
}

test "isPeerClosed returns true after the peer sends a hard reset" {
    const pair = try loopbackPair();
    defer pair.server.close();

    // SO_LINGER{on, 0} makes close() send RST instead of the usual
    // graceful FIN, giving an unambiguous "connection reset" signal.
    const linger = Linger{ .l_onoff = 1, .l_linger = 0 };
    try std.posix.setsockopt(pair.client.handle, std.posix.SOL.SOCKET, std.posix.SO.LINGER, std.mem.asBytes(&linger));
    pair.client.close();

    var conn = Connection{ .stream = pair.server, .allocator = std.testing.allocator };
    // The RST can take a moment to arrive on the loopback interface.
    var closed = false;
    for (0..200) |_| {
        if (conn.isPeerClosed()) {
            closed = true;
            break;
        }
        std.Thread.sleep(1 * std.time.ns_per_ms);
    }
    try std.testing.expect(closed);
}

test "isPeerClosed returns false on a graceful half-close (ambiguous case)" {
    // A client that finishes sending its request and calls shutdown(SHUT_WR)
    // — or just closes gracefully — produces a read-side EOF that must NOT
    // be treated as "peer gone": it is indistinguishable from a compliant
    // client that is still reading the response. See the isPeerClosed doc
    // comment.
    //
    // Not timing-dependent: a graceful FIN makes the MSG_PEEK recv succeed
    // with 0 bytes rather than error, and isPeerClosed() returns false on
    // every success path regardless of byte count. So this holds whether
    // the FIN has arrived yet (WouldBlock, caught below and also false) or
    // not — both branches agree. The sleep only makes the "FIN definitely
    // arrived" case explicit; it isn't required for the assertion to hold.
    const pair = try loopbackPair();
    defer pair.server.close();
    pair.client.close();

    var conn = Connection{ .stream = pair.server, .allocator = std.testing.allocator };
    try std.testing.expect(!conn.isPeerClosed()); // before the FIN necessarily arrives
    std.Thread.sleep(20 * std.time.ns_per_ms);
    try std.testing.expect(!conn.isPeerClosed()); // after the FIN has definitely arrived
}

test "Server struct size" {
    try std.testing.expect(@sizeOf(Server) > 0);
}

test "Method enum has expected values" {
    try std.testing.expect(@intFromEnum(Method.GET) != @intFromEnum(Method.POST));
    try std.testing.expect(@intFromEnum(Method.OPTIONS) != @intFromEnum(Method.UNKNOWN));
}

test "Connection struct has expected fields" {
    try std.testing.expect(@sizeOf(Connection) > 0);
    // read_buf is 64KB
    try std.testing.expect(@sizeOf(Connection) >= 65536);
}

test "Request struct stores method and path" {
    const req = Request{ .method = .POST, .path = "/v1/chat/completions", .body = "{}" };
    try std.testing.expectEqual(Method.POST, req.method);
    try std.testing.expectEqualStrings("/v1/chat/completions", req.path);
    try std.testing.expectEqualStrings("{}", req.body);
}

fn sendRawRequest(client: std.net.Stream, method: []const u8, path: []const u8, body: []const u8) !void {
    var header_buf: [256]u8 = undefined;
    const header = try std.fmt.bufPrint(
        &header_buf,
        "{s} {s} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {d}\r\n\r\n",
        .{ method, path, body.len },
    );
    try client.writeAll(header);
    try client.writeAll(body);
}

test "readRequest parses method, path, and a small body" {
    const pair = try loopbackPair();
    defer pair.server.close();
    defer pair.client.close();

    try sendRawRequest(pair.client, "POST", "/v1/chat/completions", "{\"model\":\"x\"}");

    var conn = Connection{ .stream = pair.server, .allocator = std.testing.allocator };
    const req = try conn.readRequest();
    try std.testing.expectEqual(Method.POST, req.method);
    try std.testing.expectEqualStrings("/v1/chat/completions", req.path);
    try std.testing.expectEqualStrings("{\"model\":\"x\"}", req.body);
    try std.testing.expect(conn.overflow_body == null);
}

test "readRequest rejects a request with no header terminator" {
    const pair = try loopbackPair();
    defer pair.server.close();
    try pair.client.writeAll("GET /health HTTP/1.1\r\nHost: localhost\r\n");
    pair.client.close();

    var conn = Connection{ .stream = pair.server, .allocator = std.testing.allocator };
    try std.testing.expectError(error.MalformedRequest, conn.readRequest());
}

test "readRequest rejects a body that is truncated by an early close" {
    const pair = try loopbackPair();
    defer pair.server.close();

    var header_buf: [128]u8 = undefined;
    const header = try std.fmt.bufPrint(&header_buf, "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 100\r\n\r\n", .{});
    try pair.client.writeAll(header);
    try pair.client.writeAll("short body, not 100 bytes");
    pair.client.close();

    var conn = Connection{ .stream = pair.server, .allocator = std.testing.allocator };
    try std.testing.expectError(error.MalformedRequest, conn.readRequest());
}

test "readRequest spills a body larger than the inline buffer onto the heap" {
    const pair = try loopbackPair();
    defer pair.client.close();

    // Larger than the 64 KiB inline read_buf so it must take the overflow path.
    const body_len = 65536 + 4096;
    const body = try std.testing.allocator.alloc(u8, body_len);
    defer std.testing.allocator.free(body);
    @memset(body, 'a');

    const writer_thread = try std.Thread.spawn(.{}, struct {
        fn run(client: std.net.Stream, b: []const u8) void {
            sendRawRequest(client, "POST", "/v1/chat/completions", b) catch {};
        }
    }.run, .{ pair.client, body });
    defer writer_thread.join();

    var conn = Connection{ .stream = pair.server, .allocator = std.testing.allocator };
    const req = try conn.readRequest();
    try std.testing.expectEqual(@as(usize, body_len), req.body.len);
    try std.testing.expect(std.mem.allEqual(u8, req.body, 'a'));
    try std.testing.expect(conn.overflow_body != null);
    // Closes pair.server's fd and frees overflow_body; std.testing.allocator
    // catches a leak if the free doesn't happen.
    conn.close();
}

test "readRequest rejects a Content-Length beyond the overflow cap" {
    const pair = try loopbackPair();
    defer pair.server.close();
    defer pair.client.close();

    var header_buf: [128]u8 = undefined;
    const header = try std.fmt.bufPrint(
        &header_buf,
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {d}\r\n\r\n",
        .{max_overflow_body_bytes + 1},
    );
    try pair.client.writeAll(header);

    var conn = Connection{ .stream = pair.server, .allocator = std.testing.allocator };
    try std.testing.expectError(error.RequestTooLarge, conn.readRequest());
}
