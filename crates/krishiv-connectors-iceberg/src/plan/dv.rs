// SPDX-License-Identifier: Apache-2.0
//! The `deletion-vector-v1` Puffin blob (table spec v3; design note §3.3 with
//! the fact check's S2), Krishiv's own codec (ADR 0014: Krishiv owns DVs):
//!
//! ```text
//! [length: u32 BE][magic D1 D3 39 64][portable roaring64 bitmap][CRC-32: u32 BE]
//! ```
//!
//! `length` counts the magic and bitmap; the CRC covers the same bytes. The
//! bitmap is the portable 64-bit layout: a little-endian `u64` count of 32-bit
//! bitmaps, then each as a little-endian `u32` key (the high 32 bits, keys
//! ascending) and a portable 32-bit roaring bitmap.
//!
//! Decoding checks what Java's `BitmapPositionDeleteIndex.deserialize` checks,
//! in its order and with its texts: the length against the entry's
//! `content_size_in_bytes`, the magic, the keys, the cardinality against the
//! entry's `record_count`, then the CRC.

use std::io::Cursor;

use krishiv_common::error::KrishivError;
use roaring::{RoaringBitmap, RoaringTreemap};

use super::deletes::invalid;

/// The magic bytes, Java's little-endian int 1681511377.
const MAGIC: [u8; 4] = [0xD1, 0xD3, 0x39, 0x64];
/// Length prefix, magic and CRC sizes.
const LENGTH_BYTES: usize = 4;
const MAGIC_BYTES: usize = 4;
const CRC_BYTES: usize = 4;

/// CRC-32 (IEEE, as `java.util.zip.CRC32`).
fn crc32(bytes: &[u8]) -> u32 {
    const fn table() -> [u32; 256] {
        let mut t = [0u32; 256];
        let mut n = 0;
        while n < 256 {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "n < 256 in a const table builder"
            )]
            let mut c = n as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 == 1 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
                k += 1;
            }
            t[n] = c;
            n += 1;
        }
        t
    }
    static TABLE: [u32; 256] = table();
    !bytes.iter().fold(!0u32, |c, &b| {
        TABLE[usize::from((c.to_le_bytes()[0]) ^ b)] ^ (c >> 8)
    })
}

fn u32_be(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

/// The deleted positions of the DV blob `blob` (`content_size_in_bytes` bytes
/// of `location` from `content_offset`), whose manifest entry says it holds
/// `record_count` positions. The length prefix is checked against the
/// entry's `content_size_in_bytes`, as Java does.
///
/// # Errors
/// `KRISHIV_ICEBERG_INVALID_DELETES` with Java's text for a length, magic,
/// key, cardinality or CRC that does not check.
pub fn decode_dv(
    blob: &[u8],
    content_size_in_bytes: i64,
    record_count: u64,
    location: &str,
) -> Result<RoaringTreemap, KrishivError> {
    let too_short = || {
        invalid(format!(
            "Invalid deletion vector {location}: {} bytes is too short",
            blob.len()
        ))
    };
    if blob.len() < LENGTH_BYTES + CRC_BYTES {
        return Err(too_short());
    }
    let declared = u32_be(blob, 0).ok_or_else(too_short)?;
    let expected = content_size_in_bytes.saturating_sub(8);
    // Java reads the length as a signed int.
    #[expect(clippy::cast_possible_wrap, reason = "Java's signed int")]
    let length = i64::from(declared as i32);
    if length != expected {
        return Err(invalid(format!(
            "Invalid bitmap data length: {length}, expected {expected}"
        )));
    }
    if i64::try_from(blob.len()).ok() != Some(content_size_in_bytes) {
        return Err(too_short());
    }
    let body = &blob[LENGTH_BYTES..blob.len() - CRC_BYTES];
    let (magic, mut bitmap) = body.split_at(MAGIC_BYTES.min(body.len()));
    if magic != MAGIC {
        let got = magic
            .try_into()
            .map(i32::from_le_bytes)
            .map_or_else(|_| "?".to_owned(), |m: i32| m.to_string());
        return Err(invalid(format!(
            "Invalid magic number: {got}, expected {}",
            i32::from_le_bytes(MAGIC)
        )));
    }
    let positions = read_bitmap(&mut bitmap, location)?;
    let cardinality = positions.len();
    if cardinality != record_count {
        return Err(invalid(format!(
            "Invalid cardinality: {cardinality}, expected {record_count}"
        )));
    }
    let crc = crc32(body);
    let stored = u32_be(blob, blob.len() - CRC_BYTES).unwrap_or_default();
    if crc != stored {
        return Err(invalid(format!(
            "Invalid CRC for deletion vector {location}: 0x{crc:x}, expected 0x{stored:x}"
        )));
    }
    Ok(positions)
}

/// The portable roaring64 bitmap at the start of `b`, as Java's
/// `RoaringPositionBitmap.deserialize` reads it.
fn read_bitmap(b: &mut &[u8], location: &str) -> Result<RoaringTreemap, KrishivError> {
    let short = || {
        invalid(format!(
            "Invalid deletion vector {location}: truncated bitmap"
        ))
    };
    let count = i64::from_le_bytes(
        b.get(..8)
            .ok_or_else(short)?
            .try_into()
            .map_err(|_| short())?,
    );
    *b = &b[8..];
    if !(0..=i64::from(i32::MAX)).contains(&count) {
        return Err(invalid(format!("Invalid bitmap count: {count}")));
    }
    let mut out = RoaringTreemap::new();
    let mut last: i64 = -1;
    for _ in 0..count {
        let key = i32::from_le_bytes(
            b.get(..4)
                .ok_or_else(short)?
                .try_into()
                .map_err(|_| short())?,
        );
        *b = &b[4..];
        if key < 0 {
            return Err(invalid(format!("Invalid unsigned key: {key}")));
        }
        if key > i32::MAX - 1 {
            return Err(invalid(format!("Key is too large: {key}")));
        }
        if i64::from(key) <= last {
            return Err(invalid("Keys must be sorted in ascending order".to_owned()));
        }
        last = i64::from(key);
        let mut cursor = Cursor::new(*b);
        let bitmap = RoaringBitmap::deserialize_from(&mut cursor)
            .map_err(|e| invalid(format!("Invalid deletion vector {location}: {e}")))?;
        let used = usize::try_from(cursor.position()).map_err(|_| short())?;
        *b = &b[used..];
        let high = u64::from(key.unsigned_abs()) << 32;
        out.extend(bitmap.iter().map(|low| high | u64::from(low)));
    }
    Ok(out)
}

/// The blob of `positions` (the writer side; T027's DV writer and the tests).
///
/// # Errors
/// `KRISHIV_ICEBERG_INVALID_DELETES` for a position Java cannot hold (a key
/// of `i32::MAX` or more) or a blob over 2 GB ("Can't serialize index > 2GB",
/// Java's text).
pub fn encode_dv(positions: &RoaringTreemap) -> Result<Vec<u8>, KrishivError> {
    if positions
        .max()
        .is_some_and(|m| m >> 32 >= u64::from(i32::MAX.unsigned_abs()))
    {
        return Err(invalid(format!(
            "Invalid position: {}",
            positions.max().unwrap_or_default()
        )));
    }
    let mut body = MAGIC.to_vec();
    // The treemap's own serialisation is the portable roaring64 layout.
    positions
        .serialize_into(&mut body)
        .map_err(|e| invalid(format!("Cannot serialize deletion vector: {e}")))?;
    let length = u32::try_from(body.len())
        .ok()
        .filter(|l| *l <= i32::MAX.unsigned_abs() - 8)
        .ok_or_else(|| invalid("Can't serialize index > 2GB".to_owned()))?;
    let mut out = Vec::with_capacity(body.len() + 8);
    out.extend_from_slice(&length.to_be_bytes());
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc32(&body).to_be_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn len(b: &[u8]) -> i64 {
        i64::try_from(b.len()).unwrap()
    }

    #[test]
    fn crc32_matches_the_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn round_trip_across_keys() {
        let positions: RoaringTreemap = [0u64, 5, 70_000, (3u64 << 32) | 7].into_iter().collect();
        let blob = encode_dv(&positions).unwrap();
        let back = decode_dv(&blob, len(&blob), 4, "dv.puffin").unwrap();
        assert_eq!(back, positions);
    }

    #[test]
    fn corrupt_blobs_fail_with_javas_texts() {
        let positions: RoaringTreemap = [1u64, 2, 3].into_iter().collect();
        let blob = encode_dv(&positions).unwrap();
        let msg = |b: &[u8], n: u64| {
            decode_dv(b, len(b), n, "x.puffin")
                .unwrap_err()
                .message()
                .to_owned()
        };
        // Cardinality against the manifest's record count (FC S2).
        assert_eq!(msg(&blob, 4), "Invalid cardinality: 3, expected 4");
        // A flipped bitmap byte past the header: the CRC.
        let mut bad = blob.clone();
        let last = bad.len() - 5;
        bad[last] ^= 0x01;
        let m = msg(&bad, 3);
        assert!(
            m.starts_with("Invalid CRC for deletion vector x.puffin: 0x")
                || m.contains("cardinality"),
            "{m}"
        );
        // Magic.
        let mut bad = blob.clone();
        bad[4] = 0;
        assert!(
            msg(&bad, 3).starts_with("Invalid magic number: "),
            "{}",
            msg(&bad, 3)
        );
        // Length: a blob cut short.
        let m = msg(&blob[..blob.len() - 1], 3);
        assert!(m.starts_with("Invalid bitmap data length: "), "{m}");
        let e = decode_dv(&blob, len(&blob), 4, "x").unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::KrishivIcebergInvalidDeletes
        );
    }
}
