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
//! ascending) and a portable 32-bit roaring bitmap. [`encode_dv`] writes
//! Java's layout (PR-36c): every key from 0 to the highest, empty bitmaps
//! for the gaps, run containers where smaller; the decoder also takes sparse
//! keys, as Java's does. Byte equality with Java is checked for array
//! containers only (the fixtures hold no runs). At the run/array size
//! boundary the two may choose differently: `roaring`'s `optimize` keeps a
//! run container while `4r + 2 <= 2c` (`r` runs, `c` values), Java's
//! `RunContainer.toEfficientContainer` compares against an array of
//! `2c + 2` bytes (recalled, not cited; review 36c1 m7), so at `c = 2r` the
//! container kinds can differ. Both are valid portable roaring, and every
//! reader (Java's, Krishiv's) takes either: rows never differ.
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

/// Java's `RoaringPositionBitmap.serialize` (apache-iceberg 1.12.0,
/// `RoaringPositionBitmap.java:265-289`) after `runLengthEncode`
/// (`BitmapPositionDeleteIndex.java:130`): the count of 32-bit bitmaps as a
/// little-endian `u64`, then **every** key from 0 to the highest (Java's
/// array is dense, so a gap is an empty bitmap), each a little-endian `u32`
/// and a standard 32-bit roaring bitmap with run containers where they are
/// smaller (`runOptimize`, the same rule as `roaring`'s `optimize`).
fn write_bitmap(positions: &RoaringTreemap, out: &mut Vec<u8>) -> std::io::Result<()> {
    let bitmaps: Vec<(u32, &RoaringBitmap)> = positions.bitmaps().collect();
    let count = bitmaps.last().map_or(0, |(k, _)| u64::from(*k) + 1);
    out.extend_from_slice(&count.to_le_bytes());
    let empty = RoaringBitmap::new();
    let mut next = bitmaps.into_iter().peekable();
    for key in 0..count {
        let Ok(key) = u32::try_from(key) else { break };
        out.extend_from_slice(&key.to_le_bytes());
        match next.next_if(|(k, _)| *k == key) {
            Some((_, b)) => {
                let mut b = b.clone();
                b.optimize();
                b.serialize_into(&mut *out)?;
            }
            None => empty.serialize_into(&mut *out)?,
        }
    }
    Ok(())
}

/// The blob of `positions` (the writer side; T027's DV writer and the
/// tests), as Java's `BitmapPositionDeleteIndex.serialize` (byte for byte
/// but for the run/array boundary in the module docs).
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
    // The dense key array is at least 12 bytes a key (the key and an empty bitmap's 8): refuse a
    // sparse position at a high key before expanding its gaps, not after (a 2 GB body is the limit).
    let keys = positions
        .bitmaps()
        .next_back()
        .map_or(0, |(k, _)| u64::from(k) + 1);
    if keys.saturating_mul(12) > u64::from(i32::MAX.unsigned_abs()) - 8 - 12 {
        return Err(invalid("Can't serialize index > 2GB".to_owned()));
    }
    let mut body = MAGIC.to_vec();
    write_bitmap(positions, &mut body)
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
    fn a_sparse_high_key_is_refused_before_its_gaps_are_expanded() {
        let high =
            RoaringTreemap::from_iter([((u64::from(i32::MAX.unsigned_abs()) - 1) << 32) + 1]);
        let err = encode_dv(&high).unwrap_err();
        assert!(err.to_string().contains("2GB"), "{err}");
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

    /// A blob around `bitmap` (the bytes after the magic) with a correct
    /// length prefix and CRC.
    fn frame(bitmap: &[u8]) -> Vec<u8> {
        let mut body = MAGIC.to_vec();
        body.extend_from_slice(bitmap);
        let mut out = u32::try_from(body.len()).unwrap().to_be_bytes().to_vec();
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc32(&body).to_be_bytes());
        out
    }

    /// The message of decoding `frame(bitmap)`, whose class is
    /// `KRISHIV_ICEBERG_INVALID_DELETES`.
    fn frame_err(bitmap: &[u8]) -> String {
        let b = frame(bitmap);
        let e = decode_dv(&b, len(&b), 0, "k.puffin").unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::KrishivIcebergInvalidDeletes,
            "{e}"
        );
        e.message().to_owned()
    }

    #[test]
    fn short_and_mismatched_blobs() {
        let m = decode_dv(&[1, 2, 3], 3, 0, "s.puffin").unwrap_err();
        assert_eq!(
            m.message(),
            "Invalid deletion vector s.puffin: 3 bytes is too short"
        );
        // Length prefix agrees with the entry but the blob is longer than the entry says.
        let blob = encode_dv(&[1u64].into_iter().collect()).unwrap();
        let m = decode_dv(&blob, len(&blob), 1, "x").unwrap();
        assert_eq!(m.len(), 1);
        let mut longer = blob.clone();
        longer.push(0);
        let e = decode_dv(&longer, len(&blob), 1, "l.puffin").unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::KrishivIcebergInvalidDeletes
        );
        assert!(
            e.message().ends_with("bytes is too short"),
            "{}",
            e.message()
        );
        // A body shorter than the magic reports "?".
        let mut tiny = vec![0, 0, 0, 2, 9, 9];
        tiny.extend_from_slice(&[0; 4]);
        let e = decode_dv(&tiny, 10, 0, "t").unwrap_err();
        assert_eq!(e.message(), "Invalid magic number: ?, expected 1681511377");
    }

    #[test]
    fn bitmap_header_errors_use_javas_texts() {
        assert_eq!(
            frame_err(&[1, 2, 3]),
            "Invalid deletion vector k.puffin: truncated bitmap"
        );
        assert_eq!(
            frame_err(&(-1i64).to_le_bytes()),
            "Invalid bitmap count: -1"
        );
        let with_key = |count: i64, keys: &[i32]| {
            let mut b = count.to_le_bytes().to_vec();
            for k in keys {
                b.extend_from_slice(&k.to_le_bytes());
            }
            b
        };
        // Count says one bitmap but the key is cut off.
        assert_eq!(
            frame_err(
                &with_key(1, &[])
                    .iter()
                    .chain(&[0u8, 0])
                    .copied()
                    .collect::<Vec<_>>()
            ),
            "Invalid deletion vector k.puffin: truncated bitmap"
        );
        assert_eq!(frame_err(&with_key(1, &[-5])), "Invalid unsigned key: -5");
        assert_eq!(
            frame_err(&with_key(1, &[i32::MAX])),
            format!("Key is too large: {}", i32::MAX)
        );
        // A valid empty 32-bit bitmap then a repeated key.
        let mut empty = Vec::new();
        RoaringBitmap::new().serialize_into(&mut empty).unwrap();
        let mut b = with_key(2, &[3]);
        b.extend_from_slice(&empty);
        b.extend_from_slice(&3i32.to_le_bytes());
        b.extend_from_slice(&empty);
        assert_eq!(frame_err(&b), "Keys must be sorted in ascending order");
        // Key present, roaring payload garbage.
        let mut b = with_key(1, &[0]);
        b.extend_from_slice(&[0xFF; 3]);
        assert!(
            frame_err(&b).starts_with("Invalid deletion vector k.puffin: "),
            "{}",
            frame_err(&b)
        );
    }

    /// The blobs of a Puffin file Iceberg 1.12.0 wrote (Spark 4.1.0,
    /// `tests/fixtures/iceberg-1.12.0/wh/db/dvs`): `(offset, length)` from its
    /// footer and the positions they hold.
    fn java_blobs() -> Vec<(Vec<u8>, Vec<u64>)> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/iceberg-1.12.0/wh/db/dvs/data");
        let a = std::fs::read(
            dir.join("00000-15-77972ffe-990c-40dc-bec0-dcf8b7ba1726-00001-deletes.puffin"),
        )
        .unwrap();
        let b = std::fs::read(
            dir.join("00000-17-5ab633b0-5fb4-4385-91a2-058f88cc71f6-00001-deletes.puffin"),
        )
        .unwrap();
        let blob = |f: &[u8], at: usize, n: usize| f[at..at + n].to_vec();
        // (bytes, the footer's `cardinality`).
        [
            (blob(&a, 4, 48), 4),
            (blob(&a, 52, 46), 3),
            (blob(&b, 4, 50), 5),
        ]
        .into_iter()
        .map(|(bytes, n)| {
            let positions = decode_dv(&bytes, len(&bytes), n, "x").unwrap();
            (bytes, positions.iter().collect())
        })
        .collect()
    }

    /// Java's bytes for the same positions (`BitmapPositionDeleteIndex
    /// .serialize`): the blobs of a Java-written Puffin file re-encoded
    /// exactly.
    #[test]
    fn encode_writes_javas_bytes() {
        let blobs = java_blobs();
        let wanted: Vec<Vec<u64>> = blobs.iter().map(|(_, p)| p.clone()).collect();
        assert_eq!(
            wanted,
            [vec![0, 4, 8, 12], vec![3, 7, 11], vec![0, 1, 3, 7, 11]]
        );
        for (bytes, positions) in blobs {
            let bitmap: RoaringTreemap = positions.iter().copied().collect();
            assert_eq!(encode_dv(&bitmap).unwrap(), bytes, "{positions:?}");
        }
    }

    /// Java's 64-bit layout writes every key from 0 to the highest, empty
    /// bitmaps for the gaps (`RoaringPositionBitmap.serialize`), and run
    /// containers after `runLengthEncode`; both decode back.
    #[test]
    fn encode_writes_dense_keys_and_run_containers() {
        let positions: RoaringTreemap = [5_u64, (2_u64 << 32) | 9].into_iter().collect();
        let blob = encode_dv(&positions).unwrap();
        // count 3: keys 0, 1 (empty) and 2.
        assert_eq!(&blob[8..16], &3_i64.to_le_bytes());
        let empty = {
            let mut e = Vec::new();
            RoaringBitmap::new().serialize_into(&mut e).unwrap();
            e
        };
        // After the length, magic, count, key 0 and its bitmap of one value
        // (cookie, size, descriptor, offset, value: 18 bytes).
        let at = 20 + 18;
        assert_eq!(&blob[at..at + 4], &1_i32.to_le_bytes());
        assert_eq!(&blob[at + 4..at + 4 + empty.len()], empty.as_slice());
        assert_eq!(decode_dv(&blob, len(&blob), 2, "d").unwrap(), positions);
        // A long run is a run container (cookie 12347).
        let run: RoaringTreemap = (0_u64..10_000).collect();
        let blob = encode_dv(&run).unwrap();
        assert_eq!(&blob[20..22], &12_347_u16.to_le_bytes());
        assert_eq!(decode_dv(&blob, len(&blob), 10_000, "r").unwrap(), run);
        assert!(blob.len() < 64, "{}", blob.len());
    }

    /// The CRC is Java's `java.util.zip.CRC32` over the magic and bitmap,
    /// stored big-endian after them.
    #[test]
    fn the_crc_covers_magic_and_bitmap() {
        let blob = encode_dv(&[1_u64, 9].into_iter().collect()).unwrap();
        let n = blob.len();
        assert_eq!(
            u32::from_be_bytes(blob[n - 4..].try_into().unwrap()),
            crc32(&blob[4..n - 4])
        );
        assert_eq!(
            u32::from_be_bytes(blob[..4].try_into().unwrap()) as usize,
            n - 8
        );
    }

    #[test]
    fn encode_refuses_a_position_with_too_large_a_key() {
        let positions: RoaringTreemap = [u64::from(i32::MAX.unsigned_abs()) << 32]
            .into_iter()
            .collect();
        let e = encode_dv(&positions).unwrap_err();
        assert_eq!(
            e.message(),
            format!(
                "Invalid position: {}",
                u64::from(i32::MAX.unsigned_abs()) << 32
            )
        );
    }
}
