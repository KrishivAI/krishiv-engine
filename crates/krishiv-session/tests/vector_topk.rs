// SPDX-License-Identifier: Apache-2.0
//! Exact top-k vector search (P3-T077): `ORDER BY <vector distance> LIMIT k` against Spark 4.2.0 goldens
//! (`tests/fixtures/spark-4.2.0/vector-topk-goldens.json`, `tests/scripts/gen_vector_topk_goldens.py`), against a
//! brute force over the kernels, and as a plan with no full sort.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::builder::{Float32Builder, ListBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int32Type};
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

mod vector_support;
use vector_support::random_vector;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "v",
            DataType::List(Arc::new(Field::new("element", DataType::Float32, true))),
            true,
        ),
    ]))
}

/// The table of the fixture's recipe (see the generator's docstring), `short_row`'s vector one element short.
fn table(n: i32, dim: usize, short_row: Option<i32>) -> RecordBatch {
    let field = Arc::new(Field::new("element", DataType::Float32, true));
    let mut vectors = ListBuilder::new(Float32Builder::new()).with_field(field);
    for i in 0..n {
        let mut v = random_vector(dim, 100 + u64::try_from(i).unwrap(), 0, 1.0);
        match i {
            _ if i % 97 == 5 => {
                vectors.append(false);
                continue;
            }
            13 => v[3] = Some(f32::NAN),
            29 => v = vec![Some(0.0); dim],
            41 => v[2] = None,
            57 => v[0] = Some(f32::INFINITY),
            _ => {}
        }
        if short_row == Some(i) {
            v.pop();
        }
        for x in v {
            vectors.values().append_option(x);
        }
        vectors.append(true);
    }
    let ids: ArrayRef = Arc::new(Int32Array::from_iter_values(0..n));
    RecordBatch::try_new(schema(), vec![ids, Arc::new(vectors.finish())]).unwrap()
}

fn literal(q: &[Option<f32>]) -> String {
    let items: Vec<String> = q
        .iter()
        .map(|x| format!("CAST({:?} AS FLOAT)", f64::from(x.unwrap())))
        .collect();
    format!("array({})", items.join(", "))
}

fn session_with(batch: RecordBatch) -> Session {
    let s = Session::new().unwrap();
    s.register_batches("vt", schema(), vec![batch]).unwrap();
    s
}

/// `(id, distance bits)` of every row of a two-column result.
fn rows(
    s: &Session,
    sql: &str,
) -> Result<Vec<(i32, Option<u32>)>, krishiv_common::error::KrishivError> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })?;
    let mut out = Vec::new();
    for b in &batches {
        let (ids, d) = (
            b.column(0).as_primitive::<Int32Type>(),
            b.column(1).as_primitive::<Float32Type>(),
        );
        for r in 0..b.num_rows() {
            out.push((ids.value(r), d.is_valid(r).then(|| d.value(r).to_bits())));
        }
    }
    Ok(out)
}

const FN: [(&str, &str, &str); 4] = [
    ("l2", "vector_l2_distance", "ASC"),
    ("l2_nulls_last", "vector_l2_distance", "ASC NULLS LAST"),
    ("cos", "vector_cosine_similarity", "DESC"),
    ("ip", "vector_inner_product", "DESC"),
];

/// The distances must equal Spark's in every position; ids must equal Spark's wherever the distance is unique in
/// the result. Rows of equal distance (the NULL rows above all) come in no defined order in Spark, so a run of
/// equal distances compares as a set, except the last run, which `LIMIT` may cut anywhere.
fn same_top_k(label: &str, got: &[(i32, Option<u32>)], want: &Value, bad: &mut Vec<String>) {
    let want: Vec<(i32, Option<u32>)> = want
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                i32::try_from(r[0].as_i64().unwrap()).unwrap(),
                r[1].as_u64().map(|b| u32::try_from(b).unwrap()),
            )
        })
        .collect();
    if got.len() != want.len() || got.iter().zip(&want).any(|(g, w)| g.1 != w.1) {
        bad.push(format!("{label}: distances {got:?} want {want:?}"));
        return;
    }
    let mut start = 0;
    while start < want.len() {
        let end = want[start..]
            .iter()
            .position(|w| w.1 != want[start].1)
            .map_or(want.len(), |p| start + p);
        if end < want.len() || end - start == 1 {
            let (mut g, mut w): (Vec<_>, Vec<_>) = (
                got[start..end].iter().map(|r| r.0).collect(),
                want[start..end].iter().map(|r| r.0).collect(),
            );
            g.sort_unstable();
            w.sort_unstable();
            if g != w {
                bad.push(format!("{label}: ids {g:?} want {w:?} at {start}"));
            }
        }
        start = end;
    }
}

#[test]
fn top_k_matches_spark_4_2_0() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/vector-topk-goldens.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let (mut compared, mut bad) = (0_usize, Vec::new());
    for case in fixture["cases"].as_array().unwrap() {
        let n = i32::try_from(case["n"].as_i64().unwrap()).unwrap();
        let dim = usize::try_from(case["dim"].as_u64().unwrap()).unwrap();
        let short = case["short_row"]
            .as_i64()
            .map(|r| i32::try_from(r).unwrap());
        let s = session_with(table(n, dim, short));
        let q = literal(&random_vector(dim, 999, 1, 1.0));
        for (name, function, order) in FN {
            for k in [1_usize, 10, 50] {
                let label = format!("n={n} dim={dim} {name}/{k}");
                let sql = format!(
                    "SELECT id, {function}(v, {q}) AS d FROM vt ORDER BY {function}(v, {q}) {order} LIMIT {k}"
                );
                let want = &case["results"][format!("{name}/{k}")];
                if want.get("error").is_some() {
                    let e = rows(&s, &sql).expect_err(&label);
                    assert!(
                        e.to_string().contains("VECTOR_DIMENSION_MISMATCH"),
                        "{label}: {e}"
                    );
                    assert_eq!(e.sql_state(), want["sqlstate"].as_str(), "{label}");
                } else {
                    same_top_k(&label, &rows(&s, &sql).unwrap(), want, &mut bad);
                }
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 36);
    assert!(
        bad.is_empty(),
        "{} of {compared} differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// `LIMIT k` is the first k rows of the full ordering, for the distances random data gives (all distinct), against
/// the kernels run row by row.
/// `ORDER BY` each distance over `n` random vectors of `dim` elements, `LIMIT k`, equals the first `k` rows of a sort
/// of the kernels' own values (ids and distance bits), for l2, cosine and inner product.
fn brute_force(n: i32, dim: usize, ks: &[usize]) {
    let s = Session::new().unwrap();
    let field = Arc::new(Field::new("element", DataType::Float32, true));
    let mut vectors = ListBuilder::new(Float32Builder::new()).with_field(field);
    let mut all = Vec::new();
    for i in 0..n {
        let v = random_vector(dim, 5000 + u64::try_from(i).unwrap(), 0, 1.0);
        for x in &v {
            vectors.values().append_option(*x);
        }
        vectors.append(true);
        all.push(v.into_iter().flatten().collect::<Vec<f32>>());
    }
    let ids: ArrayRef = Arc::new(Int32Array::from_iter_values(0..n));
    s.register_batches(
        "vt",
        schema(),
        vec![RecordBatch::try_new(schema(), vec![ids, Arc::new(vectors.finish())]).unwrap()],
    )
    .unwrap();
    let qv = random_vector(dim, 6000, 1, 1.0);
    let q: Vec<f32> = qv.iter().flatten().copied().collect();
    let lit = literal(&qv);
    type Kernel = fn(&[f32], &[f32]) -> f32;
    let cosine: Kernel = |a, b| krishiv_kernels::vector::cosine_similarity(a, b).unwrap();
    for (function, order, kernel) in [
        (
            "vector_l2_distance",
            "ASC",
            krishiv_kernels::vector::l2_distance as Kernel,
        ),
        ("vector_cosine_similarity", "DESC", cosine),
        (
            "vector_inner_product",
            "DESC",
            krishiv_kernels::vector::inner_product,
        ),
    ] {
        let mut expected: Vec<(i32, u32)> = all
            .iter()
            .enumerate()
            .map(|(i, v)| (i32::try_from(i).unwrap(), kernel(v, &q).to_bits()))
            .collect();
        // Ties are not guaranteed to keep row order in Spark; the random distances here never tie.
        expected.sort_by(|a, b| {
            let (x, y) = (f32::from_bits(a.1), f32::from_bits(b.1));
            let o = if order == "ASC" {
                x.total_cmp(&y)
            } else {
                y.total_cmp(&x)
            };
            o.then(a.0.cmp(&b.0))
        });
        for &k in ks {
            let sql = format!(
                "SELECT id, {function}(v, {lit}) AS d FROM vt ORDER BY {function}(v, {lit}) {order} LIMIT {k}"
            );
            let got = rows(&s, &sql).unwrap();
            let want: Vec<(i32, Option<u32>)> =
                expected[..k].iter().map(|r| (r.0, Some(r.1))).collect();
            assert_eq!(got, want, "{function} {order} k={k}");
        }
    }
}

#[test]
fn top_k_is_the_first_k_of_a_brute_force_ordering() {
    brute_force(3000, 96, &[1, 10, 1000]);
}

/// Design section 8.3 at full size: 100k x 768, k in {1, 10, 1000}. Slow in a debug build:
/// `cargo test --release -p krishiv-session --test vector_topk -- --ignored`.
#[test]
#[ignore = "100k x 768 vectors: run in release (design 8.3)"]
fn top_k_brute_force_at_100k_by_768() {
    brute_force(100_000, 768, &[1, 10, 1000]);
}

/// A NULL query vector gives NULL distances and the query still runs (design 8.1; golden:
/// tests/spark-golden/v4.2.0/inputs/vector-distance.sql:59).
#[test]
fn a_null_query_vector_gives_null_distances() {
    let s = session_with(table(50, 8, None));
    let got = rows(
        &s,
        "SELECT id, vector_l2_distance(v, CAST(NULL AS ARRAY<FLOAT>)) AS d FROM vt \
         ORDER BY vector_l2_distance(v, CAST(NULL AS ARRAY<FLOAT>)) LIMIT 5",
    )
    .unwrap();
    assert_eq!(got.len(), 5);
    assert!(got.iter().all(|r| r.1.is_none()), "{got:?}");
}

/// The plan is a partition-local top-k and a final top-k over the projected distance: no full sort.
#[test]
fn the_plan_has_a_top_n_and_no_sort() {
    let s = session_with(table(50, 8, None));
    let q = literal(&random_vector(8, 999, 1, 1.0));
    let sql = format!("EXPLAIN SELECT id FROM vt ORDER BY vector_l2_distance(v, {q}) LIMIT 5");
    let batches = block_on(async { s.sql(&sql).await?.collect().await }).unwrap();
    let text: String = batches
        .iter()
        .flat_map(|b| {
            let c = b.column(0).as_string_view();
            (0..c.len())
                .map(|i| c.value(i).to_owned())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    // The shape, top to bottom: the final TopN, the gather, the partition-local TopN; no full sort.
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.trim_start_matches([' ', '+', '-']))
        .collect();
    let at = |prefix: &str| lines.iter().position(|l| l.starts_with(prefix));
    let (top, gather, local) = (
        at("TopN 5"),
        at("Exchange SinglePartition"),
        at("LocalTopN 5"),
    );
    assert!(
        matches!((top, gather, local), (Some(t), Some(g), Some(l)) if t < g && g < l),
        "{text}"
    );
    assert!(
        lines.iter().all(|l| !l.starts_with("Sort")),
        "a full sort in the plan:\n{text}"
    );
}

/// `array(literals)` outside a vector call is still the constructor: its values, NULL elements, a column element and
/// the empty array are as they were.
#[test]
fn array_of_literals_keeps_its_values() {
    let s = Session::new().unwrap();
    let text = |sql: &str| -> String {
        let b = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
        arrow_cast::display::array_value_to_string(b[0].column(0), 0).unwrap()
    };
    assert_eq!(text("SELECT array(1, 2, 3)"), "[1, 2, 3]");
    assert_eq!(text("SELECT array(1, NULL, 3)"), "[1, , 3]");
    assert_eq!(text("SELECT array(CAST(1.5 AS FLOAT), 2)"), "[1.5, 2.0]");
    assert_eq!(text("SELECT array()"), "[]");
    assert_eq!(text("SELECT array(1, id) FROM (SELECT 7 AS id)"), "[1, 7]");
}
