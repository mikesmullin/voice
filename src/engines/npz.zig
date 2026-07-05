//! Minimal ZIP (STORED entries only) + NumPy .npy reader, just enough to
//! read Kokoro's voice style-vector pack (kokoro-voices-*.bin, an .npz -
//! i.e. a plain ZIP - of uncompressed float32 .npy arrays, one per voice
//! name, each shaped (510, 1, 256)).
//!
//! Does NOT implement DEFLATE - the reference kokoro-voices-v1.0.bin file
//! uses STORED (no compression) for every entry (verified with Python's
//! zipfile module), which keeps this simple. If a differently-packed
//! voices file ever shows up with compressed entries, this will need a
//! DEFLATE decoder added.

const std = @import("std");

pub const NpyArray = struct {
    /// Raw float32 data (owned by caller's allocator).
    data: []f32,
    shape: []usize,
};

const EOCD_SIG: u32 = 0x06054b50;
const CENTRAL_SIG: u32 = 0x02014b50;
const LOCAL_SIG: u32 = 0x04034b50;

fn readU32(bytes: []const u8, off: usize) u32 {
    return std.mem.readInt(u32, bytes[off..][0..4], .little);
}
fn readU16(bytes: []const u8, off: usize) u16 {
    return std.mem.readInt(u16, bytes[off..][0..2], .little);
}

/// Finds `entry_name` (e.g. "af_bella.npy") inside the zip and returns its
/// raw (decompressed, since STORED) bytes as a slice into `zip_bytes`.
fn findEntry(zip_bytes: []const u8, entry_name: []const u8) ![]const u8 {
    // Scan backwards for the EOCD signature (comment field makes a forward
    // scan unsafe in general, but backwards-from-end is standard practice).
    var eocd_off: ?usize = null;
    var i: usize = zip_bytes.len;
    while (i >= 4) {
        i -= 1;
        if (i + 4 <= zip_bytes.len and readU32(zip_bytes, i) == EOCD_SIG) {
            eocd_off = i;
            break;
        }
        if (zip_bytes.len - i > 66000) break; // EOCD comment max 65535 bytes
    }
    const eocd = eocd_off orelse return error.NotAZip;

    const entry_count = readU16(zip_bytes, eocd + 10);
    const central_dir_offset = readU32(zip_bytes, eocd + 16);

    var pos: usize = central_dir_offset;
    var n: usize = 0;
    while (n < entry_count) : (n += 1) {
        if (readU32(zip_bytes, pos) != CENTRAL_SIG) return error.BadCentralDirectory;
        const compression = readU16(zip_bytes, pos + 10);
        const uncompressed_size = readU32(zip_bytes, pos + 24);
        const name_len = readU16(zip_bytes, pos + 28);
        const extra_len = readU16(zip_bytes, pos + 30);
        const comment_len = readU16(zip_bytes, pos + 32);
        const local_header_offset = readU32(zip_bytes, pos + 42);
        const name = zip_bytes[pos + 46 ..][0..name_len];

        if (std.mem.eql(u8, name, entry_name)) {
            if (compression != 0) return error.UnsupportedCompression;
            const lh = local_header_offset;
            if (readU32(zip_bytes, lh) != LOCAL_SIG) return error.BadLocalHeader;
            const lh_name_len = readU16(zip_bytes, lh + 26);
            const lh_extra_len = readU16(zip_bytes, lh + 28);
            const data_start = lh + 30 + lh_name_len + lh_extra_len;
            return zip_bytes[data_start..][0..uncompressed_size];
        }

        pos += 46 + name_len + extra_len + comment_len;
    }
    return error.EntryNotFound;
}

/// Parses a .npy buffer (float32, C-order) into a shape + data view. `bytes`
/// must outlive the returned NpyArray (data is a reinterpreted slice, no
/// copy - caller should copy out if the source buffer is transient).
fn parseNpy(bytes: []const u8, alloc: std.mem.Allocator) !NpyArray {
    if (!std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.NotAnNpy;
    const major = bytes[6];
    var header_len: usize = undefined;
    var header_start: usize = undefined;
    if (major == 1) {
        header_len = readU16(bytes, 8);
        header_start = 10;
    } else {
        header_len = readU32(bytes, 8);
        header_start = 12;
    }
    const header = bytes[header_start..][0..header_len];

    // Extract shape tuple text, e.g. "(510, 1, 256)", from the header dict.
    const shape_key = "'shape':";
    const shape_idx = std.mem.indexOf(u8, header, shape_key) orelse return error.NoShapeInHeader;
    const paren_open = std.mem.indexOfScalarPos(u8, header, shape_idx, '(') orelse return error.NoShapeInHeader;
    const paren_close = std.mem.indexOfScalarPos(u8, header, paren_open, ')') orelse return error.NoShapeInHeader;
    const shape_str = header[paren_open + 1 .. paren_close];

    var shape_list: std.ArrayList(usize) = .empty;
    var parts = std.mem.splitScalar(u8, shape_str, ',');
    while (parts.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \t");
        if (trimmed.len == 0) continue;
        try shape_list.append(alloc, try std.fmt.parseInt(usize, trimmed, 10));
    }

    const data_start = header_start + header_len;
    const raw = bytes[data_start..];
    const float_count = raw.len / 4;
    const data = try alloc.alloc(f32, float_count);
    const src: [*]align(1) const f32 = @ptrCast(raw.ptr);
    @memcpy(data, src[0..float_count]);

    return .{ .data = data, .shape = try shape_list.toOwnedSlice(alloc) };
}

/// Loads one named voice's float32 array (shape (510, 1, 256) for Kokoro)
/// out of an .npz-style zip file already read fully into memory.
pub fn loadVoice(alloc: std.mem.Allocator, npz_bytes: []const u8, voice_name: []const u8) !NpyArray {
    const entry_name = try std.mem.concat(alloc, u8, &.{ voice_name, ".npy" });
    const npy_bytes = try findEntry(npz_bytes, entry_name);
    return parseNpy(npy_bytes, alloc);
}
