// SPDX-License-Identifier: Apache-2.0
//! `theta_sketch_agg`, `theta_union_agg`, `theta_intersection_agg` and the `theta_*` scalars through SQL (P3 PR-49
//! part 2): the result does not depend on the partition count, and sketches of 100,000 distinct values per input
//! type (estimation mode) agree with the ones Spark 4.3.0 wrote (`tests/spark-golden/functions/theta_interop/`).
#![forbid(unsafe_code)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn cell(b: &RecordBatch, c: usize, row: usize) -> String {
    let a = b.column(c);
    if a.is_null(row) {
        return "NULL".into();
    }
    if let Some(s) = a.as_string_opt::<i32>() {
        return s.value(row).to_owned();
    }
    if let Some(s) = a.as_string_view_opt() {
        return s.value(row).to_owned();
    }
    if let Some(s) = a.as_primitive_opt::<Int64Type>() {
        return s.value(row).to_string();
    }
    if let Some(s) = a.as_primitive_opt::<arrow_array::types::Int32Type>() {
        return s.value(row).to_string();
    }
    format!("<{}>", a.data_type())
}

fn rows(s: &Session, sql: &str) -> R<Vec<String>> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        let mut out = Vec::new();
        for b in &batches {
            for r in 0..b.num_rows() {
                out.push(
                    (0..b.num_columns())
                        .map(|c| cell(b, c, r))
                        .collect::<Vec<_>>()
                        .join("|"),
                );
            }
        }
        out.sort();
        Ok(out)
    })
}

#[test]
fn the_aggregates_do_not_depend_on_partitions() -> R {
    let s = Session::new()?;
    let from = |n: u32| format!("(SELECT /*+ REPARTITION({n}) */ id FROM range(0, 20000))");
    for n in [1_u32, 8] {
        let sk = format!(
            "SELECT theta_sketch_estimate(theta_sketch_agg(id % 997)) FROM {}",
            from(n)
        );
        let est: i64 = rows(&s, &sk)?[0].parse()?;
        assert_eq!(est, 997, "{n} partitions: exact below the nominal size");
        let un = format!(
            "SELECT theta_sketch_estimate(theta_union_agg(sk)) FROM \
             (SELECT g, theta_sketch_agg(id % 997) AS sk FROM (SELECT id, id % 5 AS g FROM {}) GROUP BY g)",
            from(n)
        );
        assert_eq!(rows(&s, &un)?[0].parse::<i64>()?, 997, "{n}: union");
    }
    // The same data gives the same estimate at 1 and 8 partitions even in estimation mode.
    let q = |n: u32| {
        format!(
            "SELECT theta_sketch_estimate(theta_sketch_agg(id, 10)) FROM {}",
            from(n)
        )
    };
    assert_eq!(rows(&s, &q(1))?, rows(&s, &q(8))?);
    Ok(())
}

/// `theta_intersection_agg` over a few sketches spread over 1 and 8 partitions (most of them empty, so the merge
/// skips their empty states), in exact and in estimation mode.
#[test]
fn the_intersection_does_not_depend_on_partitions() -> R {
    let s = Session::new()?;
    let q = |lg: u32, width: u32, n: u32| {
        let sketches = (0..3_u32)
            .map(|i| {
                let (lo, hi) = (i * width / 5, i * width / 5 + width);
                format!("SELECT theta_sketch_agg(id, {lg}) AS sk FROM range({lo}, {hi})")
            })
            .collect::<Vec<_>>()
            .join(" UNION ALL ");
        format!(
            "SELECT theta_sketch_estimate(theta_intersection_agg(sk)) \
             FROM (SELECT /*+ REPARTITION({n}) */ sk FROM ({sketches}))"
        )
    };
    // Exact: the three ranges share [2*width/5, width) = 60 of 100 values.
    assert_eq!(rows(&s, &q(12, 100, 1))?, ["60"]);
    assert_eq!(rows(&s, &q(12, 100, 8))?, ["60"]);
    // Estimation mode (lg 10 holds 1,024 entries; 50,000 values per sketch).
    let one = rows(&s, &q(10, 50_000, 1))?;
    assert_eq!(rows(&s, &q(10, 50_000, 8))?, one);
    let est: f64 = one[0].parse()?;
    assert!((est - 30_000.0).abs() < 30_000.0 * 0.2, "{est}");
    Ok(())
}

#[test]
fn lg_nom_entries_4_keeps_sixteen_hashes_as_java_does() -> R {
    // Java datasketches 6.2.0, longs 0..999 at lgNomEntries 4: estimate 945.5839687772802 over 16 retained hashes.
    let s = Session::new()?;
    for q in [
        "SELECT theta_sketch_estimate(theta_sketch_agg(id, 4)) FROM range(0, 1000)",
        "SELECT theta_sketch_estimate(theta_union_agg(sk, 4)) FROM (SELECT theta_sketch_agg(id) AS sk FROM range(0, 1000))",
        "SELECT theta_sketch_estimate(theta_union(theta_sketch_agg(id), theta_sketch_agg(id), 4)) FROM range(0, 1000)",
    ] {
        assert_eq!(rows(&s, q)?, ["946"], "{q}");
    }
    Ok(())
}

#[test]
fn the_intersection_of_two_groups_and_of_nothing() -> R {
    let s = Session::new()?;
    let q = "SELECT theta_sketch_estimate(theta_intersection_agg(sk)) FROM (\
             SELECT theta_sketch_agg(id) AS sk FROM range(0, 100) UNION ALL \
             SELECT theta_sketch_agg(id) AS sk FROM range(50, 150))";
    assert_eq!(rows(&s, q)?, ["50"]);
    // Spark 4.3.0 (theta_interop/fixture.json): a group with no sketch is the empty sketch.
    let none = "SELECT theta_sketch_estimate(theta_intersection_agg(s)) FROM VALUES (CAST(NULL AS BINARY)) t(s)";
    let want = fx_dir_json("fixture.json")?["intersection_of_nothing"]
        .as_i64()
        .ok_or("intersection_of_nothing")?
        .to_string();
    assert_eq!(rows(&s, none)?, [want]);
    Ok(())
}

const INTEROP_SRC: [(&str, &str); 8] = [
    ("int", "CAST(id AS INT)"),
    ("bigint", "id * 100000 + 7"),
    ("float", "CAST(id AS FLOAT) + CAST(0.5 AS FLOAT)"),
    ("double", "CAST(id AS DOUBLE) + 0.25"),
    ("string", "CONCAT('value-', CAST(id AS STRING))"),
    (
        "binary",
        "CAST(CONCAT('bin-', CAST(id AS STRING)) AS BINARY)",
    ),
    ("array_int", "ARRAY(CAST(id AS INT), CAST(id % 7 AS INT))"),
    ("array_bigint", "ARRAY(id * 100000 + 7, id % 7)"),
];

fn fx_dir_json(file: &str) -> R<serde_json::Value> {
    let path = format!(
        "{}/../../tests/spark-golden/functions/theta_interop/{file}",
        env!("CARGO_MANIFEST_DIR")
    );
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

/// Estimation-mode sketches written by Spark 4.3.0 (`fixture.json`): Krishiv's estimate of Spark's bytes is Spark's
/// printed number, its own sketch of the same data gives the same number (the update encoding of every input type
/// is Spark's), and the union of the two is that number again.
#[test]
fn spark_written_estimation_mode_sketches_agree_with_krishivs() -> R {
    let s = Session::new()?;
    let fx = fx_dir_json("fixture.json")?;
    let n = fx["distinct"].as_i64().ok_or("distinct")?;
    let mut dump = serde_json::Map::new();
    for (name, expr) in INTEROP_SRC {
        let want = fx["sketches"][name]["estimate"]
            .as_i64()
            .ok_or("estimate")?;
        let spark_hex = fx["sketches"][name]["hex"].as_str().ok_or("hex")?;
        let est = |e: &str| -> R<i64> {
            Ok(rows(&s, &format!("SELECT theta_sketch_estimate({e})"))?[0].parse()?)
        };
        assert_eq!(
            est(&format!("unhex('{spark_hex}')"))?,
            want,
            "{name}: Spark's bytes"
        );
        let own =
            format!("(SELECT theta_sketch_agg(v) FROM (SELECT {expr} AS v FROM range(0, {n})))");
        assert_eq!(
            est(&own)?,
            want,
            "{name}: Krishiv's sketch of the same data"
        );
        assert_eq!(
            est(&format!("theta_union(unhex('{spark_hex}'), {own}, 12)"))?,
            want,
            "{name}: the union"
        );
        let hex = rows(&s, &format!("SELECT hex({own})"))?.remove(0);
        // Exported before any golden is checked, so a changed encoding can still be dumped for gen.py.
        dump.insert(name.to_owned(), serde_json::Value::String(hex.clone()));
        if let Ok(path) = std::env::var("KRISHIV_THETA_INTEROP_DUMP") {
            std::fs::write(path, serde_json::to_string(&dump)?)?;
        }
        let back = fx_dir_json("reverse.json")?;
        let back = &back["krishiv_read_by_spark"][name];
        assert_eq!(back["estimate"].as_i64(), Some(want), "{name}: Spark reads");
        assert_eq!(
            back["union_with_spark"].as_i64(),
            Some(want),
            "{name}: Spark unions"
        );
        assert_eq!(
            fx_dir_json("krishiv-compact.json")?[name].as_str(),
            Some(hex.as_str()),
            "{name}: Krishiv's bytes changed; rerun gen.py with the new krishiv-compact.json"
        );
    }
    Ok(())
}
