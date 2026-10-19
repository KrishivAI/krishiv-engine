// SPDX-License-Identifier: Apache-2.0
//! The text formats' shared readers and charsets.

use bytes::Bytes;

use super::reader::{LineReader, ReadChunk};
use super::{Charset, LineSep};
use crate::io::ByteRange;

/// The bytes of `range` in `bytes`.
fn slice<'a>(bytes: &'a [u8], range: &std::ops::Range<u64>) -> &'a [u8] {
    let (a, b) = (usize::try_from(range.start), usize::try_from(range.end));
    &bytes[a.expect("range")..b.expect("range")]
}

fn lines_of(bytes: &[u8], split: ByteRange, sep: LineSep) -> Vec<(String, u64)> {
    let mut r = LineReader::new(bytes.len() as u64, split, true, sep);
    let mut out = Vec::new();
    loop {
        match r.next_line().expect("read") {
            ReadChunk::End => return out,
            ReadChunk::Line(l, at) => out.push((String::from_utf8(l).expect("utf8"), at)),
            ReadChunk::NeedData(range) => {
                let b = Bytes::copy_from_slice(slice(bytes, &range));
                r.push(&range, &b).expect("push");
            }
        }
    }
}

/// A tiny deterministic generator.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % n
    }
}

#[test]
fn any_split_boundaries_read_each_line_once() {
    // More than one read chunk, every kind of terminator, empty lines and a
    // `\r` / `\n` pair that straddles a chunk boundary.
    let mut text = Vec::new();
    let mut r = Lcg(7);
    let mut i = 0;
    while text.len() < 2_600_000 {
        text.extend_from_slice(
            format!(
                "line {i} {}",
                "x".repeat(usize::try_from(r.next(40)).expect("small"))
            )
            .as_bytes(),
        );
        text.extend_from_slice(match r.next(5) {
            0 => b"\r\n".as_slice(),
            1 => b"\r",
            2 => b"\n\n",
            _ => b"\n",
        });
        i += 1;
    }
    let whole = lines_of(&text, ByteRange::whole(text.len() as u64), LineSep::Default);
    assert!(whole.len() > 50_000);
    let size = text.len() as u64;
    for _ in 0..6 {
        let mut cuts: Vec<u64> = (0..r.next(9)).map(|_| r.next(size)).collect();
        cuts.extend([0, size, 1_048_576, 1_048_577, 2_097_152]);
        cuts.sort_unstable();
        cuts.dedup();
        let mut got = Vec::new();
        for w in cuts.windows(2) {
            got.extend(lines_of(
                &text,
                ByteRange::new(w[0], w[1]),
                LineSep::Default,
            ));
        }
        got.sort_by_key(|l| l.1);
        assert_eq!(got, whole, "cuts {cuts:?}");
    }
}

#[test]
fn custom_line_separators_and_a_last_line_without_one() {
    let got = lines_of(
        b"a;;b;c",
        ByteRange::new(0, 6),
        LineSep::Custom(b";".to_vec()),
    );
    let lines: Vec<&str> = got.iter().map(|l| l.0.as_str()).collect();
    assert_eq!(lines, ["a", "", "b", "c"]);
    assert!(lines_of(b"", ByteRange::new(0, 0), LineSep::Default).is_empty());
    assert_eq!(
        lines_of(b"\n", ByteRange::new(0, 1), LineSep::Default).len(),
        1
    );
    assert!(lines_of(b"a\nb", ByteRange::new(9, 12), LineSep::Default).is_empty());
}

#[test]
fn charsets_decode_and_encode_like_java() {
    assert_eq!(Charset::parse(None).expect("utf8"), Charset::Utf8);
    assert_eq!(Charset::parse(Some("latin1")).expect("l1"), Charset::Latin1);
    assert_eq!(
        Charset::parse(Some("us_ascii")).expect("ascii"),
        Charset::Ascii
    );
    assert!(Charset::parse(Some("UTF-16")).is_err());
    assert_eq!(Charset::Latin1.decode(b"a\xe9"), "a\u{e9}");
    assert_eq!(Charset::Ascii.decode(b"a\xe9"), "a\u{fffd}");
    assert_eq!(Charset::Utf8.decode(b"a\xff"), "a\u{fffd}");
    let mut out = Vec::new();
    Charset::Latin1.encode("a\u{e9}\u{20ac}", &mut out);
    assert_eq!(out, b"a\xe9?");
    out.clear();
    Charset::Ascii.encode("\u{e9}z", &mut out);
    assert_eq!(out, b"?z");
}

#[test]
fn readers_ask_for_chunks_and_refuse_misplaced_bytes() {
    use super::reader::WholeReader;
    let text = vec![b'x'; 2_500_000];
    let size = text.len() as u64;
    let mut whole = WholeReader::new(size, ByteRange::whole(size));
    let mut asked = 0;
    while let Some(range) = whole.need() {
        assert_eq!(
            whole.upcoming(1000).first(),
            Some(&(range.start..range.start + 1000))
        );
        whole
            .push(&range, &Bytes::copy_from_slice(slice(&text, &range)))
            .expect("push");
        asked += 1;
    }
    assert_eq!(asked, 3);
    assert!(whole.buffered_bytes() >= text.len());
    assert_eq!(whole.take().map(|b| b.len()), Some(text.len()));
    assert!(whole.take().is_none() && whole.need().is_none() && whole.upcoming(9).is_empty());
    // Bytes that do not continue the file are an error, not a panic.
    let mut r = LineReader::new(10, ByteRange::new(0, 10), true, LineSep::Default);
    assert_eq!(r.upcoming(4).first(), Some(&(0..4)));
    assert!(r.push(&(3..5), &Bytes::from_static(b"ab")).is_err());
    assert!(r.push(&(0..2), &Bytes::from_static(b"a")).is_err());
    assert!(r.buffered_bytes() == 0);
    // A split that starts after the end of the file reads nothing.
    let late = LineReader::new(10, ByteRange::new(20, 30), true, LineSep::Default);
    assert!(late.upcoming(4).is_empty());
}

#[test]
fn compressed_files_and_codecs_are_refused() {
    use krishiv_common::error::ErrorClass::KrishivUnsupportedFeature;
    for path in ["a.gz", "a.deflate", "a.bz2", "a.lz4", "a.snappy", "a.zst"] {
        let e = super::check_plain(path).expect_err(path);
        assert_eq!(e.error_class(), KrishivUnsupportedFeature, "{path}");
    }
    assert!(super::check_plain("a.csv").is_ok());
}
