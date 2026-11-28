// SPDX-License-Identifier: Apache-2.0
//! Tests for the compression codecs of text files: round trips, concatenated members, size limits and corrupt input.
use bytes::Bytes;

use super::LineSep;
use super::codec::{
    COMPRESSED_CHUNK, Codec, CompressionSuffixes, Compressor, Decompressor, MAX_PUSH_OUTPUT,
};
use super::reader::{LineReader, ReadChunk, WholeReader};
use crate::error::FormatError;
use crate::io::ByteRange;

const CODECS: [Codec; 5] = [
    Codec::Gzip,
    Codec::Deflate,
    Codec::Bzip2,
    Codec::Lz4,
    Codec::Snappy,
];

fn fixture(name: &str) -> Vec<u8> {
    let p = format!(
        "{}/../../tests/fixtures/spark-4.2.0/compressed-text/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn unpack(codec: Codec, bytes: &[u8], step: usize) -> Result<Vec<u8>, crate::error::FormatError> {
    let mut d = Decompressor::new(codec);
    let mut out = Vec::new();
    for c in bytes.chunks(step) {
        d.push(c, &mut out)?;
    }
    d.finish(&mut out)?;
    Ok(out)
}

fn pack(codec: Codec, plain: &[u8], step: usize) -> Vec<u8> {
    let mut c = Compressor::new(codec);
    let mut out = Vec::new();
    for p in plain.chunks(step) {
        c.write(p).expect("write");
        out.extend(c.take());
    }
    out.extend(c.finish().expect("finish"));
    out
}

/// Files Spark 4.2 wrote (`tests/scripts/gen_compressed_text.py`) decompress
/// to the file Spark wrote uncompressed, however the bytes arrive.
#[test]
fn spark_written_files_decompress_to_the_plain_file() {
    for (base, ext) in [("csv", "csv"), ("json", "json"), ("text", "txt")] {
        let plain = fixture(&format!("{base}.{ext}"));
        for codec in CODECS {
            let name = format!("{base}.{ext}{}", codec.suffix());
            let packed = fixture(&name);
            assert_eq!(
                Codec::of_path(&name, CompressionSuffixes::Exact),
                Some(codec)
            );
            for step in [1, 3, packed.len()] {
                let got = unpack(codec, &packed, step).unwrap_or_else(|e| panic!("{name}: {e:?}"));
                assert_eq!(got, plain, "{name} in steps of {step}");
            }
        }
    }
}

#[test]
fn round_trips_across_blocks_and_chunks() {
    let plain: Vec<u8> = (0..60_000u32)
        .flat_map(|i| format!("row {i},{}\n", i % 17).into_bytes())
        .collect();
    for codec in CODECS {
        for step in [7_000, 300_000, plain.len()] {
            let packed = pack(codec, &plain, step);
            assert!(packed.len() < plain.len(), "{codec:?}");
            assert_eq!(
                unpack(codec, &packed, 70_000).expect("unpack"),
                plain,
                "{codec:?}"
            );
        }
        assert!(
            unpack(codec, &pack(codec, b"", 1), 1)
                .expect("empty")
                .is_empty()
        );
    }
}

#[test]
fn gzip_members_may_be_concatenated() {
    let mut two = pack(Codec::Gzip, b"a\n", 8);
    two.extend(pack(Codec::Gzip, b"b\n", 8));
    assert_eq!(
        unpack(Codec::Gzip, &two, 5).expect("two members"),
        b"a\nb\n"
    );
}

#[test]
fn truncated_and_bad_input_is_corrupt() {
    for codec in CODECS {
        let packed = pack(codec, &vec![b'x'; 5_000], 5_000);
        let cut = &packed[..packed.len() - 3];
        assert!(unpack(codec, cut, 100).is_err(), "{codec:?} truncated");
    }
    assert!(unpack(Codec::Gzip, b"not gzip at all", 4).is_err());
    // A block longer than the limit.
    assert!(unpack(Codec::Lz4, &[0x7f, 0, 0, 0, 0, 0, 0, 1], 8).is_err());
}

#[test]
fn write_codec_names() {
    assert_eq!(Codec::parse_write("GZIP"), Some(Some(Codec::Gzip)));
    assert_eq!(Codec::parse_write("none"), Some(None));
    assert_eq!(Codec::parse_write("Uncompressed"), Some(None));
    assert_eq!(Codec::parse_write("zstd"), None);
    assert_eq!(
        Codec::parse_write("snappy").flatten().map(Codec::suffix),
        Some(".snappy")
    );
}

/// Hadoop reads every stream of a `.deflate` or `.bz2` file (Spark 4.2
/// returns both lines of two joined streams).
#[test]
fn joined_streams_are_all_read() {
    for codec in [Codec::Deflate, Codec::Bzip2, Codec::Gzip] {
        let mut two = pack(codec, b"a\n", 8);
        two.extend(pack(codec, b"b\n", 8));
        for step in [1, 7, two.len()] {
            assert_eq!(
                unpack(codec, &two, step).expect("two streams"),
                b"a\nb\n",
                "{codec:?} {step}"
            );
        }
        // A second stream cut short is a truncated file.
        let mut cut = pack(codec, b"a\n", 8);
        cut.extend(&pack(codec, b"b\n", 8)[..6]);
        assert!(unpack(codec, &cut, 4).is_err(), "{codec:?}");
    }
}

#[test]
fn a_chunk_that_inflates_past_the_limit_is_an_error() {
    let zeros = vec![0u8; 150 << 20];
    for codec in [Codec::Gzip, Codec::Deflate] {
        let packed = pack(codec, &zeros, zeros.len());
        assert!(packed.len() < 1 << 20);
        assert!(unpack(codec, &packed, packed.len()).is_err(), "{codec:?}");
    }
}

#[test]
fn a_snappy_chunk_larger_than_its_block_is_corrupt() {
    // Block of 4 bytes whose chunk declares 1000 plain bytes.
    let chunk = snap::raw::Encoder::new()
        .compress_vec(&[7u8; 1000])
        .expect("compress");
    let mut bad = 4u32.to_be_bytes().to_vec();
    bad.extend(u32::try_from(chunk.len()).unwrap().to_be_bytes());
    bad.extend(chunk);
    assert!(unpack(Codec::Snappy, &bad, 3).is_err());
}

/// The suffixes each Spark line knows (observed with 3.5.9, 4.0.5 and 4.1.4).
#[test]
fn suffixes_name_codecs_by_spark_line() {
    use CompressionSuffixes::{AnyCase, Exact, Extended};
    let entry = &krishiv_common::config::IO_TEXT_COMPRESSION_SUFFIXES;
    assert_eq!(super::codec::SUFFIXES_CONF, entry.key);
    assert_eq!(
        entry.value_type,
        krishiv_common::config::ValueType::Enum(super::codec::SUFFIXES_VALUES)
    );
    for (path, want) in [
        ("a.gz", [Some(Codec::Gzip); 3]),
        ("a.LZ4", [None, Some(Codec::Lz4), Some(Codec::Lz4)]),
        ("a.Snappy", [None, Some(Codec::Snappy), Some(Codec::Snappy)]),
        ("a.gzip", [None, None, Some(Codec::Gzip)]),
        ("a.zst", [None, None, Some(Codec::Zstd)]),
        ("a.ZSTD", [None, None, Some(Codec::Zstd)]),
        ("a.bzip2", [None; 3]),
        ("a.zlib", [None; 3]),
        ("a.gz2", [None; 3]),
        ("gz", [None; 3]),
    ] {
        let got = [Exact, AnyCase, Extended].map(|r| Codec::of_path(path, r));
        assert_eq!(got, want, "{path}");
    }
}

/// One zstd frame of `plain`.
fn zst(plain: &[u8]) -> Vec<u8> {
    zstd::bulk::compress(plain, 3).expect("compress")
}

/// Everything `bytes` decodes to, pushed in `step`s, with the largest piece
/// one call handed over.
fn unzst(bytes: &[u8], step: usize) -> Result<(Vec<u8>, usize), FormatError> {
    let mut d = Decompressor::new(Codec::Zstd);
    let (mut out, mut largest) = (Vec::new(), 0);
    for c in bytes.chunks(step) {
        let mut piece = Vec::new();
        d.push(c, &mut piece)?;
        loop {
            largest = largest.max(piece.len());
            out.append(&mut piece);
            if !d.pending() {
                break;
            }
            d.resume(&mut piece)?;
        }
    }
    let mut tail = Vec::new();
    d.finish(&mut tail)?;
    out.extend(tail);
    Ok((out, largest))
}

#[test]
fn zstd_frames_decode_however_they_arrive() {
    let plain: Vec<u8> = (0..60_000u32)
        .flat_map(|i| format!("row {i},{}\n", i % 17).into_bytes())
        .collect();
    let mut two = zst(&plain);
    two.extend(zst(b"last\n"));
    let mut want = plain.clone();
    want.extend_from_slice(b"last\n");
    for step in [1, 7, 4096, two.len()] {
        assert_eq!(unzst(&two, step).expect("two frames").0, want, "{step}");
    }
    let one = zst(&plain);
    for bad in [&one[..one.len() - 1], &one[..3], b"not zstd at all"] {
        let e = unzst(bad, 5).expect_err("broken");
        assert!(matches!(e, FormatError::Corrupt(_)), "{e:?}");
    }
}

/// A very repetitive file inflates past what one push may hand over: it is
/// decoded in pieces, not refused.
#[test]
fn a_repetitive_zstd_chunk_is_decoded_in_pieces() {
    let plain = b"0123456789abcde\n".repeat(10 << 20);
    assert!(plain.len() > MAX_PUSH_OUTPUT);
    let packed = zst(&plain);
    assert!((packed.len() as u64) < COMPRESSED_CHUNK);
    let (out, largest) = unzst(&packed, packed.len()).expect("pieces");
    assert!(out == plain);
    assert!(largest <= 32 << 20, "{largest}");

    let size = packed.len() as u64;
    let mut r = LineReader::new(size, ByteRange::new(0, size), false, LineSep::Default)
        .compressed(Some(Codec::Zstd));
    let (mut lines, mut held) = (0_usize, 0);
    loop {
        match r.next_line().expect("line") {
            ReadChunk::End => break,
            ReadChunk::Line(line, _) => {
                assert_eq!(line, b"0123456789abcde");
                lines += 1;
                held = held.max(r.buffered_bytes());
            }
            ReadChunk::NeedData(range) => r
                .push(&range, &Bytes::copy_from_slice(&packed))
                .expect("push"),
        }
    }
    assert_eq!(lines, 10 << 20);
    assert!((1..64 << 20).contains(&held), "{held}");
}

/// One record, or one whole file, past the limit from a single chunk is a
/// bomb.
#[test]
fn a_zstd_bomb_is_an_error() {
    let packed = zst(&vec![b'a'; MAX_PUSH_OUTPUT + (16 << 20)]);
    let size = packed.len() as u64;
    let bytes = Bytes::from(packed);
    let mut lines = LineReader::new(size, ByteRange::new(0, size), false, LineSep::Default)
        .compressed(Some(Codec::Zstd));
    let e = loop {
        match lines.next_line() {
            Ok(ReadChunk::NeedData(range)) => lines.push(&range, &bytes).expect("push"),
            Ok(other) => panic!("{other:?}"),
            Err(e) => break e,
        }
    };
    assert!(matches!(e, FormatError::Corrupt(_)), "{e:?}");
    let mut whole = WholeReader::new(size, ByteRange::new(0, size)).compressed(Some(Codec::Zstd));
    let range = whole.need().expect("range");
    let e = whole.push(&range, &bytes).expect_err("bomb");
    assert!(matches!(e, FormatError::Corrupt(_)), "{e:?}");
}

#[test]
fn zstd_files_are_not_written() {
    let mut c = Compressor::new(Codec::Zstd);
    assert!(c.write(b"x").is_err());
    assert!(c.take().is_empty());
    assert_eq!(c.buffered(), 0);
    assert!(c.finish().is_err());
}
