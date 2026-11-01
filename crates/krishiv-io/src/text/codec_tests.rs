// SPDX-License-Identifier: Apache-2.0
use super::codec::{Codec, Compressor, Decompressor};

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
            assert_eq!(Codec::of_path(&name), Some(codec));
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
