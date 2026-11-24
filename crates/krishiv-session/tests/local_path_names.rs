// SPDX-License-Identifier: Apache-2.0
//! Local paths whose names object paths used to percent-encode (`~`, `%`, `#`,
//! `^`, `|`, `<`, `>`, `"`, non-ASCII text): a write lands in the directory the
//! path names and a read reads it back, as Spark does. An 8.3 Windows temp
//! directory (`RUNNER~1`) was read as `RUNNER%7E1`.
#![expect(clippy::unwrap_used, reason = "test code")]

use arrow_array::{Array, Int64Array};
use futures::executor::block_on;
use krishiv_session::Session;

/// Names with no glob character (`*`, `?`, `[`, `{` are globs in a read path).
const NAMES: [&str; 8] = [
    "RUNNER~1",
    "a%7Eb",
    "p=a%3Ab",
    "jos\u{e9}",
    "a#b",
    "x^y|z",
    "<q>\"",
    "a b",
];

fn count(s: &Session, sql: &str) -> i64 {
    let batches = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    let c = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(c.len(), 1);
    c.value(0)
}

#[test]
fn a_write_and_a_read_use_the_directory_the_path_names() {
    let base = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    for name in NAMES {
        let dir = base.path().join(name).join("t");
        let path = dir.display().to_string();
        let df = block_on(s.sql("SELECT id FROM range(0, 50, 1, 2)")).unwrap();
        block_on(df.write().parquet(&path)).unwrap();
        let written = std::fs::read_dir(&dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".parquet")
            })
            .count();
        assert!(written > 0, "{name}: no data file in {path}");
        assert_eq!(
            count(&s, &format!("SELECT count(*) FROM parquet.`{path}`")),
            50,
            "{name}"
        );
        // A path relative to the parent with `..` reaches the same directory.
        let dotted = base.path().join(name).join("t").join("..").join("t");
        let dotted = dotted.display().to_string();
        assert_eq!(
            count(&s, &format!("SELECT sum(id) FROM parquet.`{dotted}`")),
            (0..50).sum::<i64>(),
            "{name}"
        );
    }
    // Nothing was written under an encoded name beside them.
    let names: Vec<String> = std::fs::read_dir(base.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        !names
            .iter()
            .any(|n| n.contains("%7E1") || n.contains("%25")),
        "{names:?}"
    );
}

#[test]
fn a_table_located_below_such_a_directory_takes_inserts_and_reads_them() {
    let base = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    for (i, name) in NAMES.iter().enumerate() {
        let dir = base.path().join(name).join("loc");
        let path = dir.display().to_string();
        let t = format!("t{i}");
        block_on(s.sql(&format!(
            "CREATE TABLE {t} (id BIGINT) USING parquet LOCATION '{}'",
            path.replace('\'', "\\'")
        )))
        .unwrap();
        block_on(
            block_on(s.sql(&format!("INSERT INTO {t} SELECT id FROM range(7)")))
                .unwrap()
                .collect(),
        )
        .unwrap();
        block_on(
            block_on(s.sql(&format!("INSERT INTO {t} SELECT id FROM range(3)")))
                .unwrap()
                .collect(),
        )
        .unwrap();
        assert_eq!(
            count(&s, &format!("SELECT count(*) FROM {t}")),
            10,
            "{name}"
        );
        assert!(
            std::fs::read_dir(&dir).unwrap().count() > 0,
            "{name}: {path}"
        );
    }
}
