// SPDX-License-Identifier: Apache-2.0
//! The `vector_*` functions against Spark 4.2.0, bit for bit
//! (`tests/fixtures/spark-4.2.0/vector-goldens.json`, made by
//! `tests/scripts/gen_vector_goldens.py` from a real Spark run). Spark sums in blocks of 8 in `float`
//! (`VectorFunctionImplUtils`); a plain fold differs from it from 8 dimensions up, which the
//! random 768-dimension cases show and the small-integer goldens cannot.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::builder::{Float32Builder, ListBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

mod vector_support;
use vector_support::random_vector;

fn from_bits(v: &Value) -> Option<Vec<Option<f32>>> {
    v.as_array().map(|a| {
        a.iter()
            .map(|e| {
                e.as_u64()
                    .map(|b| f32::from_bits(u32::try_from(b).unwrap()))
            })
            .collect()
    })
}

fn list(v: &[Option<f32>]) -> ArrayRef {
    let field = Arc::new(Field::new("element", DataType::Float32, true));
    let mut b = ListBuilder::new(Float32Builder::new()).with_field(field);
    for x in v {
        b.values().append_option(*x);
    }
    b.append(true);
    Arc::new(b.finish())
}

enum Got {
    Bits(u32),
    BitsList(Vec<u32>),
    Null,
    Error { state: Option<String>, text: String },
}

fn run(s: &Session, sql: &str) -> Got {
    let batches = match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(b) => b,
        Err(e) => {
            return Got::Error {
                state: e.sql_state().map(str::to_owned),
                text: format!("{e} {:?}", e.message_parameters()),
            };
        }
    };
    let col = batches[0].column(0);
    if col.is_null(0) {
        return Got::Null;
    }
    match col.data_type() {
        DataType::Float32 => Got::Bits(col.as_primitive::<Float32Type>().value(0).to_bits()),
        DataType::List(_) => Got::BitsList(
            col.as_list::<i32>()
                .value(0)
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .map(|x| x.to_bits())
                .collect(),
        ),
        other => panic!("{other:?}"),
    }
}

fn check(name: &str, cell: &str, got: &Got, want: &Value, bad: &mut Vec<String>) {
    let ok = match got {
        Got::Bits(b) => want["bits"].as_u64() == Some(u64::from(*b)),
        Got::BitsList(b) => want["bits_list"].as_array().is_some_and(|w| {
            w.len() == b.len()
                && w.iter()
                    .zip(b)
                    .all(|(w, b)| w.as_u64() == Some(u64::from(*b)))
        }),
        Got::Null => want["null"] == true,
        Got::Error { state, text } => want.get("error").is_some_and(|e| {
            text.contains(e.as_str().unwrap_or_default())
                && state.as_deref() == want["sqlstate"].as_str()
                && want["parameters"].as_object().is_some_and(|p| {
                    p.iter()
                        .all(|(_, v)| text.contains(v.as_str().unwrap_or_default()))
                })
        }),
    };
    if !ok {
        bad.push(format!("{name} {cell}: want {want}"));
    }
}

#[test]
fn every_vector_function_matches_spark_bit_for_bit() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/vector-goldens.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(fixture["spark"], "4.2.0");
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "a",
            DataType::List(Arc::new(Field::new("element", DataType::Float32, true))),
            true,
        ),
        Field::new(
            "b",
            DataType::List(Arc::new(Field::new("element", DataType::Float32, true))),
            true,
        ),
    ]));
    let (mut compared, mut bad) = (0_usize, Vec::new());
    for case in fixture["cases"].as_array().unwrap() {
        let (name, a, b) = if case["kind"] == "random" {
            let dim = usize::try_from(case["dim"].as_u64().unwrap()).unwrap();
            let (seed, scale) = (
                case["seed"].as_u64().unwrap(),
                case["scale"].as_f64().unwrap(),
            );
            (
                format!("random dim {dim} seed {seed} scale {scale}"),
                random_vector(dim, seed, 0, scale),
                random_vector(dim, seed, 1, scale),
            )
        } else {
            (
                case["name"].as_str().unwrap().to_owned(),
                from_bits(&case["a"]).unwrap(),
                from_bits(&case["b"]).unwrap(),
            )
        };
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![list(&a), list(&b)]).unwrap();
        s.register_batches("t", Arc::clone(&schema), vec![batch])
            .unwrap();
        for (cell, sql) in [
            ("ip", "SELECT vector_inner_product(a, b) FROM t"),
            ("l2", "SELECT vector_l2_distance(a, b) FROM t"),
            ("cos", "SELECT vector_cosine_similarity(a, b) FROM t"),
            ("norm1", "SELECT vector_norm(a, CAST(1.0 AS FLOAT)) FROM t"),
            ("norm2", "SELECT vector_norm(a, CAST(2.0 AS FLOAT)) FROM t"),
            (
                "norminf",
                "SELECT vector_norm(a, CAST('inf' AS FLOAT)) FROM t",
            ),
            (
                "normalize1",
                "SELECT vector_normalize(a, CAST(1.0 AS FLOAT)) FROM t",
            ),
            (
                "normalize2",
                "SELECT vector_normalize(a, CAST(2.0 AS FLOAT)) FROM t",
            ),
            (
                "normalizeinf",
                "SELECT vector_normalize(a, CAST('inf' AS FLOAT)) FROM t",
            ),
        ] {
            // The engine's default is Spark 4.3, which differs from these 4.2.0 goldens in exactly these
            // cells: its infinity norm propagates a NaN element and its normalizer divides by a subnormal
            // norm. The 4.3 behaviour is pinned by the unit tests in `krishiv-functions` and the 4.3 goldens.
            if matches!(cell, "norminf" | "normalizeinf") && name.starts_with("nan-")
                || name == "denormal-pair" && matches!(cell, "normalize1" | "normalizeinf")
            {
                continue;
            }
            check(&name, cell, &run(&s, sql), &case["result"][cell], &mut bad);
            compared += 1;
        }
    }
    assert!(compared > 400, "{compared}");
    assert!(
        bad.is_empty(),
        "{} of {compared} differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// Spark 4.3 rejects only a zero norm (`VectorFunctionImplUtils`): a subnormal norm divides, so the smallest
/// subnormal float under the infinity norm normalizes to `[1.0]`. A zero vector is still NULL.
#[test]
fn vector_normalize_on_4_3_divides_by_a_subnormal_norm() {
    let s = Session::new().unwrap();
    let got = run(
        &s,
        "SELECT vector_normalize(array(CAST(1.4E-45 AS FLOAT)), CAST('inf' AS FLOAT))",
    );
    assert!(
        matches!(got, Got::BitsList(ref b) if *b == vec![1.0_f32.to_bits()]),
        "a subnormal norm divides on 4.3"
    );
    let zero = run(
        &s,
        "SELECT vector_normalize(array(CAST(0.0 AS FLOAT)), CAST('inf' AS FLOAT))",
    );
    assert!(matches!(zero, Got::Null));
}
