// SPDX-License-Identifier: Apache-2.0
//! Which file suffixes the line sources decompress, per Spark line.
//!
//! Observed with Spark 3.5.9, 4.0.5, 4.1.4, 4.2.0, 4.2.1 and 4.3.0 (`spark.read.text` and the CSV and
//! JSON sources): 3.5 knows the codecs' own suffixes in lower case; 4.0 knows them in any case; from
//! 4.1 `.gzip` is gzip too and Zstandard is read as `.zst` and `.zstd`. Stock 3.5 and 4.0 fail on a
//! `.zst` file, which Krishiv refuses there, as it refuses `.zstd` (those lines read the compressed
//! bytes as text).
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

/// A Zstandard frame holding `data` as one raw block (under 256 bytes).
fn zstd_frame(data: &[u8]) -> Vec<u8> {
    let n = u8::try_from(data.len()).expect("a short file");
    // Magic; a single segment with a one-byte content size; the last block, raw.
    let mut out = vec![0x28, 0xB5, 0x2F, 0xFD, 0x20, n];
    let block = (u32::from(n) << 3) | 1;
    out.extend_from_slice(&block.to_le_bytes()[..3]);
    out.extend_from_slice(data);
    out
}

/// The gzip file the engine writes for the lines `x` and `y`.
fn gzip_bytes() -> Vec<u8> {
    let out = tempfile::tempdir().expect("temp dir");
    let out = out.path().join("t");
    let s = Session::builder().build().expect("session");
    block_on(async {
        s.sql(&format!(
            "INSERT OVERWRITE DIRECTORY '{}' USING text OPTIONS (compression 'gzip') \
             SELECT explode(array('x', 'y'))",
            out.display()
        ))
        .await?
        .collect()
        .await
    })
    .expect("write the file");
    let part = std::fs::read_dir(&out)
        .expect("list the output")
        .map(|e| e.expect("entry").path())
        .find(|p| p.extension().is_some_and(|e| e == "gz"))
        .expect("a part file");
    std::fs::read(part).expect("read the file")
}

/// A file called `name` holding `bytes`, in a directory of its own.
fn file(name: &str, bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).expect("write the file");
    (dir, path)
}

fn lines(profile: SparkProfile, name: &str, bytes: &[u8]) -> Result<Vec<String>, KrishivError> {
    let (_dir, path) = file(name, bytes);
    let s = Session::builder().profile(profile).build()?;
    let batches = block_on(async {
        s.sql(&format!("SELECT value FROM text.`{}`", path.display()))
            .await?
            .collect()
            .await
    })?;
    Ok(batches
        .iter()
        .flat_map(|b| {
            let values = b.columns()[0].as_string_view();
            (0..b.num_rows()).map(|i| values.value(i).to_owned())
        })
        .collect())
}

/// What a read gave: the two lines, the file's bytes as text, or the error class.
fn outcome(r: Result<Vec<String>, KrishivError>) -> String {
    match r {
        Ok(v) if v == ["x", "y"] => "decompressed".to_owned(),
        Ok(_) => "raw".to_owned(),
        Err(e) => e.error_class().name().to_owned(),
    }
}

#[test]
fn a_compressed_text_file_is_known_by_its_suffix_as_each_spark_line_knows_it() {
    const REFUSED: &str = "KRISHIV_UNSUPPORTED_FEATURE";
    const D: &str = "decompressed";
    let gzip = gzip_bytes();
    let zstd = zstd_frame(b"x\ny\n");
    // Suffix, bytes, then what 3.5, 4.0 and 4.1 or later give.
    let cases: [(&str, &[u8], [&str; 3]); 8] = [
        ("f.gz", &gzip, [D, D, D]),
        ("f.GZ", &gzip, ["raw", D, D]),
        ("f.gzip", &gzip, ["raw", "raw", D]),
        ("f.GZip", &gzip, ["raw", "raw", D]),
        ("f.zst", &zstd, [REFUSED, REFUSED, D]),
        ("f.ZST", &zstd, ["raw", REFUSED, D]),
        ("f.zstd", &zstd, [REFUSED, REFUSED, D]),
        ("f.Zstd", &zstd, ["raw", REFUSED, D]),
    ];
    let mut wrong = Vec::new();
    for profile in SparkProfile::ALL {
        let line = match profile {
            SparkProfile::V3_5 => 0,
            SparkProfile::V4_0 => 1,
            _ => 2,
        };
        for (name, bytes, want) in cases {
            let got = outcome(lines(profile, name, bytes));
            if got != want[line] {
                wrong.push(format!("{profile:?} {name}: {got}, want {}", want[line]));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_broken_zstd_file_fails_the_read() {
    let whole = zstd_frame(b"x\ny\n");
    for bytes in [&whole[..whole.len() - 1], b"nozstdhere".as_slice()] {
        let got = outcome(lines(SparkProfile::V4_3, "f.zst", bytes));
        assert_eq!(got, "FAILED_READ_FILE.NO_HINT", "{} bytes", bytes.len());
    }
    // Two frames in one file are one stream, as Hadoop's codec reads them.
    let two = [whole.clone(), whole].concat();
    let got = lines(SparkProfile::V4_3, "f.zst", &two).expect("two frames");
    assert_eq!(got, ["x", "y", "x", "y"]);
}
