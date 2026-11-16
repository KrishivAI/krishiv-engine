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

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The fixture's random vector: `float32((unit - 0.5) * 2 * scale)`, the stream seeded `seed * 2 + role`.
fn random_vector(dim: usize, seed: u64, role: u64, scale: f64) -> Vec<Option<f32>> {
    let mut state = seed * 2 + role;
    (0..dim)
        .map(|_| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "the 53-bit unit of the generator"
            )]
            let unit = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "float32 rounding is the point"
            )]
            let v = ((unit - 0.5) * 2.0 * scale) as f32;
            Some(v)
        })
        .collect()
}

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
