// SPDX-License-Identifier: Apache-2.0
//! Tests for [`LineReader`] over a compressed stream cut at every kind of
//! point: it returns exactly the complete lines of the bytes decoded before
//! the break (a lone `\r` or part of a custom separator at the break does not
//! end a line), then the error, or the end when lenient (RS-26 follow-up;
//! Hadoop's `LineReader` returns the lines read before its codec throws).

use bytes::Bytes;

use super::codec::{COMPRESSED_CHUNK, Codec, Compressor, Decompressor};
use super::reader::{LineReader, ReadChunk};
use crate::error::FormatError;
use crate::io::ByteRange;

use super::LineSep;

fn pack(codec: Codec, plain: &[u8]) -> Vec<u8> {
    let mut c = Compressor::new(codec);
    c.write(plain).expect("write");
    let mut out = c.take();
    out.extend(c.finish().expect("finish"));
    out
}

/// The plain bytes decoded from `bytes` pushed in the reader's chunks, and
/// whether the stream broke.
fn decoded(codec: Codec, bytes: &[u8]) -> (Vec<u8>, bool) {
    let mut d = Decompressor::new(codec);
    let mut out = Vec::new();
    let step = usize::try_from(COMPRESSED_CHUNK).expect("chunk");
    for c in bytes.chunks(step) {
        if d.push(c, &mut out).is_err() {
            return (out, true);
        }
    }
    let broken = d.finish(&mut out).is_err();
    (out, broken)
}

/// The lines Hadoop's reader returns from `p`: after a break, the partial
/// last line is dropped, and so is a line whose terminator might go on.
fn complete_lines(p: &[u8], sep: &LineSep, broken: bool) -> Vec<Vec<u8>> {
    let mut lines = Vec::new();
    let mut i = 0;
    loop {
        let rest = &p[i..];
        let found = match sep {
            LineSep::Default => {
                memchr::memchr2(b'\n', b'\r', rest).and_then(|j| match (rest[j], rest.get(j + 1)) {
                    (b'\r', Some(b'\n')) => Some((j, 2)),
                    (b'\r', None) if broken => None,
                    _ => Some((j, 1)),
                })
            }
            LineSep::Custom(s) => memchr::memmem::find(rest, s).map(|j| (j, s.len())),
        };
        match found {
            Some((j, t)) => {
                lines.push(rest[..j].to_vec());
                i += j + t;
            }
            None => {
                if !broken && !rest.is_empty() {
                    lines.push(rest.to_vec());
                }
                return lines;
            }
        }
    }
}

/// Every line the reader returns, and how it ended.
fn read(
    codec: Codec,
    bytes: &[u8],
    sep: &LineSep,
    lenient: bool,
) -> (Vec<Vec<u8>>, Result<(), FormatError>) {
    let size = bytes.len() as u64;
    let mut r = LineReader::new(size, ByteRange::new(0, size), false, sep.clone())
        .compressed(Some(codec))
        .lenient(lenient);
    let mut lines = Vec::new();
    loop {
        match r.next_line() {
            Ok(ReadChunk::End) => return (lines, Ok(())),
            Ok(ReadChunk::Line(l, _)) => lines.push(l),
            Ok(ReadChunk::NeedData(range)) => {
                let a = usize::try_from(range.start).expect("range");
                let b = usize::try_from(range.end).expect("range");
                r.push(&range, &Bytes::copy_from_slice(&bytes[a..b]))
                    .expect("push");
            }
            Err(e) => return (lines, Err(e)),
        }
    }
}

/// Lines of random hex (so the compressed file spans several chunks) with
/// terminators cycling through `\n`, `\r\n` and `\r`, empty lines among them;
/// or ended by `sep` when custom.
fn plain(seed: u64, bytes: usize, sep: &LineSep) -> Vec<u8> {
    let mut x = seed;
    let mut next = |n: u64| {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (x >> 33) % n
    };
    let mut out = Vec::with_capacity(bytes + 64);
    let mut k = 0usize;
    while out.len() < bytes {
        for _ in 0..next(40) {
            out.push(b"0123456789abcdef"[usize::try_from(next(16)).expect("hex")]);
        }
        match sep {
            LineSep::Default => out.extend_from_slice([&b"\n"[..], b"\r\n", b"\r"][k % 3]),
            LineSep::Custom(s) => out.extend_from_slice(s),
        }
        k += 1;
    }
    out
}

/// Cut points: a spread over the file, both sides of every chunk boundary,
/// the last bytes, and `extra`.
fn cuts(len: usize, spread: usize, extra: &[usize]) -> Vec<usize> {
    let chunk = usize::try_from(COMPRESSED_CHUNK).expect("chunk");
    let mut c: Vec<usize> = (1..spread).map(|i| len * i / spread).collect();
    let mut b = chunk;
    while b < len {
        c.extend([b - 1, b, b + 1]);
        b += chunk;
    }
    c.extend((1..12).map(|d| len.saturating_sub(d)));
    c.extend(extra);
    c.retain(|&x| x > 0 && x <= len);
    c.sort_unstable();
    c.dedup();
    c
}

fn check(codec: Codec, compressed: &[u8], sep: &LineSep, at: &[usize]) {
    for &cut in at {
        let bytes = &compressed[..cut];
        let (prefix, broken) = decoded(codec, bytes);
        let want = complete_lines(&prefix, sep, broken);
        let (got, end) = read(codec, bytes, sep, false);
        assert_eq!(got.len(), want.len(), "{codec:?} {sep:?} cut {cut}");
        assert!(got == want, "{codec:?} {sep:?} cut {cut}: lines differ");
        match end {
            Err(FormatError::Corrupt(_)) => assert!(broken, "{codec:?} cut {cut}"),
            Err(e) => panic!("{codec:?} cut {cut}: not corrupt: {e}"),
            Ok(()) => assert!(!broken, "{codec:?} cut {cut}: break not raised"),
        }
        let (lenient, end) = read(codec, bytes, sep, true);
        assert!(
            lenient == want,
            "{codec:?} {sep:?} cut {cut}: lenient lines differ"
        );
        assert!(end.is_ok(), "{codec:?} cut {cut}: lenient raised");
    }
}

#[test]
fn a_cut_stream_returns_the_complete_lines_decoded_before_the_break() {
    let sep = LineSep::Default;
    let text = plain(7, 300_000, &sep);
    for (codec, spread) in [(Codec::Gzip, 60), (Codec::Deflate, 60), (Codec::Bzip2, 6)] {
        let packed = pack(codec, &text);
        check(codec, &packed, &sep, &cuts(packed.len(), spread, &[]));
        // Whole, the file reads as the plain file.
        let (got, end) = read(codec, &packed, &sep, false);
        assert!(end.is_ok());
        assert_eq!(got, complete_lines(&text, &sep, false));
    }
}

/// gzip and deflate hand over the bytes they inflated before the break, so
/// a cut late in the file keeps most of its lines (flate2's write decoder
/// held them until `finish` before).
#[test]
fn a_late_cut_keeps_most_of_the_file() {
    let sep = LineSep::Default;
    let text = plain(11, 300_000, &sep);
    for codec in [Codec::Gzip, Codec::Deflate] {
        let packed = pack(codec, &text);
        let (prefix, broken) = decoded(codec, &packed[..packed.len() - 100]);
        assert!(broken, "{codec:?}");
        assert!(
            prefix.len() > text.len() * 9 / 10,
            "{codec:?}: {}",
            prefix.len()
        );
    }
}

/// A lone `\r` as the last byte decoded before a break might be the first
/// half of `\r\n`: its line is not returned.
#[test]
fn a_lone_carriage_return_at_the_break_does_not_end_its_line() {
    let text = b"a\nbb\r";
    for codec in [Codec::Gzip, Codec::Deflate, Codec::Bzip2] {
        // A whole stream of `text` followed by the start of a second one
        // breaks right after the `\r`.
        let mut packed = pack(codec, text);
        let second = pack(codec, b"\ncc\n");
        let whole_len = packed.len();
        packed.extend_from_slice(&second[..second.len() / 2]);
        let (prefix, broken) = decoded(codec, &packed);
        assert!(broken, "{codec:?}");
        assert!(prefix.starts_with(text), "{codec:?}");
        let (got, end) = read(codec, &packed, &LineSep::Default, false);
        assert_eq!(
            got,
            complete_lines(&prefix, &LineSep::Default, true),
            "{codec:?}"
        );
        if prefix == text {
            assert_eq!(got, vec![b"a".to_vec()], "{codec:?}");
        }
        assert!(matches!(end, Err(FormatError::Corrupt(_))), "{codec:?}");
        // Ending cleanly there, the `\r` ends `bb`.
        let (got, end) = read(codec, &packed[..whole_len], &LineSep::Default, false);
        assert_eq!(got, vec![b"a".to_vec(), b"bb".to_vec()], "{codec:?}");
        assert!(end.is_ok(), "{codec:?}");
    }
}

#[test]
fn a_custom_separator_cut_at_the_break_does_not_end_its_line() {
    let sep = LineSep::Custom(b"<|>".to_vec());
    let text = plain(13, 200_000, &sep);
    for codec in [Codec::Gzip, Codec::Deflate] {
        let packed = pack(codec, &text);
        check(codec, &packed, &sep, &cuts(packed.len(), 40, &[]));
    }
    // Part of the separator, then a break.
    for codec in [Codec::Gzip, Codec::Bzip2] {
        let mut packed = pack(codec, b"a<|>b<|");
        packed.extend_from_slice(&pack(codec, b">c<|>")[..3]);
        let (got, end) = read(codec, &packed, &sep, false);
        assert_eq!(got, vec![b"a".to_vec()], "{codec:?}");
        assert!(matches!(end, Err(FormatError::Corrupt(_))), "{codec:?}");
    }
}

/// Concatenated streams (gzip members): a cut at a stream boundary is a
/// clean end, a cut inside the second stream keeps every line of the first.
#[test]
fn a_cut_in_a_later_stream_keeps_the_earlier_streams() {
    let sep = LineSep::Default;
    let first = plain(17, 120_000, &sep);
    let second = plain(19, 120_000, &sep);
    for (codec, spread) in [(Codec::Gzip, 40), (Codec::Deflate, 40), (Codec::Bzip2, 4)] {
        let mut packed = pack(codec, &first);
        let boundary = packed.len();
        packed.extend(pack(codec, &second));
        let at = cuts(
            packed.len(),
            spread,
            &[boundary - 1, boundary, boundary + 1, boundary + 9],
        );
        check(codec, &packed, &sep, &at);
        let (got, end) = read(codec, &packed[..boundary + 9], &sep, false);
        assert!(end.is_err(), "{codec:?}");
        assert!(
            got.len() >= complete_lines(&first, &sep, false).len(),
            "{codec:?}"
        );
        let (got, end) = read(codec, &packed[..boundary], &sep, false);
        assert!(end.is_ok(), "{codec:?}");
        assert_eq!(got, complete_lines(&first, &sep, false), "{codec:?}");
    }
}
