// SPDX-License-Identifier: Apache-2.0
//! Spark reads what Krishiv writes (design note P3-T021 sections 4.2 and 9): the tables under
//! `tests/fixtures/delta-4.4.1/krishiv-written/` are written by Krishiv
//! (`write_readback_tables`, run by hand with `KRISHIV_DELTA_READBACK_OUT=<dir>` and
//! `--ignored`), then read by Spark 4.2.0 + delta-spark 4.4.1 (`gen_delta_fixtures.py /out
//! readback /in`), which records each table's rows, `DESCRIBE HISTORY`, protocol and an append
//! of its own in `READBACK.json`. `spark_reads_krishiv_written_tables` compares Spark's rows
//! with Krishiv's.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::{Int32Builder, ListBuilder, MapBuilder, StringBuilder};
use arrow_array::{
    ArrayRef, Date32Array, Int32Array, Int64Array, RecordBatch, StringArray, StructArray,
};
use arrow_schema::{DataType as ArrowType, Field, Fields, Schema, SchemaRef};
use krishiv_exec::planner::WriteMode;
use krishiv_plan::{Expr, Literal};
use serde_json::Value;

use super::super::NewDeltaTable;
use super::{Env, adds, check_field_ids, fixtures, footer_schema};

fn written() -> PathBuf {
    fixtures().join("krishiv-written")
}

fn plain_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int64, true),
        Field::new("v", ArrowType::Utf8, true),
        Field::new("p", ArrowType::Utf8, true),
        Field::new("d", ArrowType::Date32, true),
    ]))
}

fn plain(rows: &[(i64, &str, Option<&str>, i32)]) -> Vec<RecordBatch> {
    let id: ArrayRef = Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0)));
    let v: ArrayRef = Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.1)));
    let p: ArrayRef = Arc::new(StringArray::from(
        rows.iter().map(|r| r.2).collect::<Vec<_>>(),
    ));
    let d: ArrayRef = Arc::new(Date32Array::from_iter_values(rows.iter().map(|r| r.3)));
    vec![RecordBatch::try_new(plain_schema(), vec![id, v, p, d]).unwrap()]
}

fn nested_schema() -> SchemaRef {
    let s = Fields::from(vec![
        Field::new("x", ArrowType::Int32, true),
        Field::new("y", ArrowType::Utf8, true),
    ]);
    Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int64, true),
        Field::new("s", ArrowType::Struct(s), true),
        Field::new(
            "a",
            ArrowType::List(Arc::new(Field::new("element", ArrowType::Int32, true))),
            true,
        ),
        Field::new(
            "m",
            ArrowType::Map(
                Arc::new(Field::new(
                    "key_value",
                    ArrowType::Struct(Fields::from(vec![
                        Field::new("key", ArrowType::Utf8, false),
                        Field::new("value", ArrowType::Int32, true),
                    ])),
                    false,
                )),
                false,
            ),
            true,
        ),
    ]))
}

fn nested(ids: &[i64]) -> Vec<RecordBatch> {
    let schema = nested_schema();
    let id: ArrayRef = Arc::new(Int64Array::from_iter_values(ids.iter().copied()));
    let ArrowType::Struct(s_fields) = schema.field(1).data_type().clone() else {
        unreachable!()
    };
    let x: ArrayRef = Arc::new(Int32Array::from_iter_values(
        ids.iter().map(|i| i32::try_from(*i).unwrap() * 10),
    ));
    let y: ArrayRef = Arc::new(StringArray::from_iter_values(
        ids.iter().map(|i| format!("y{i}")),
    ));
    let s: ArrayRef = Arc::new(StructArray::new(s_fields, vec![x, y], None));
    let mut a = ListBuilder::new(Int32Builder::new()).with_field(Arc::new(Field::new(
        "element",
        ArrowType::Int32,
        true,
    )));
    let mut m = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    for i in ids {
        let i = i32::try_from(*i).unwrap();
        a.values().append_value(i);
        a.values().append_value(i + 1);
        a.append(true);
        m.keys().append_value(format!("k{i}"));
        m.values().append_value(i);
        m.append(true).unwrap();
    }
    let a: ArrayRef = Arc::new(a.finish());
    let m: ArrayRef = Arc::new(m.finish());
    let m = arrow_cast::cast(&m, schema.field(3).data_type()).unwrap();
    vec![RecordBatch::try_new(schema, vec![id, s, a, m]).unwrap()]
}

fn create(
    env: &Env,
    schema: &SchemaRef,
    partitions: &[&str],
    properties: &[(&str, &str)],
) -> NewDeltaTable {
    NewDeltaTable::new(
        env.engine.clone(),
        schema,
        partitions.iter().map(|p| (*p).to_owned()).collect(),
        properties
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect::<BTreeMap<_, _>>(),
        None,
    )
    .unwrap()
}

fn eq(column: &str, value: &str) -> Arc<Expr> {
    Arc::new(Expr::function(
        "==",
        vec![
            Expr::column([column]),
            Expr::literal(Literal::String(value.to_owned())),
        ],
    ))
}

/// An env whose table root is `dir` (made empty).
fn env_at(dir: &Path) -> Env {
    drop(std::fs::remove_dir_all(dir));
    std::fs::create_dir_all(dir).unwrap();
    Env::at(tempfile::tempdir().unwrap(), dir.to_path_buf())
}

/// Writes the read-back tables under `$KRISHIV_DELTA_READBACK_OUT`.
#[test]
#[ignore = "generates fixtures: KRISHIV_DELTA_READBACK_OUT=<dir>"]
fn write_readback_tables() {
    let out = PathBuf::from(std::env::var("KRISHIV_DELTA_READBACK_OUT").unwrap());
    // An unpartitioned create, then an append.
    let env = env_at(&out.join("append"));
    let t = create(&env, &plain_schema(), &[], &[]);
    env.write(
        &t,
        WriteMode::Append,
        plain(&[(1, "a", Some("x"), 19_000)]),
        &[],
    )
    .unwrap();
    env.write(
        &env.open().unwrap(),
        WriteMode::Append,
        plain(&[
            (2, "b", None, 19_001),
            (3, &"z".repeat(40), Some("y"), 19_002),
        ]),
        &[],
    )
    .unwrap();
    // A truncate.
    let env = env_at(&out.join("overwrite"));
    let t = create(&env, &plain_schema(), &[], &[]);
    env.write(
        &t,
        WriteMode::Append,
        plain(&[(1, "a", Some("x"), 19_000)]),
        &[],
    )
    .unwrap();
    env.write(
        &env.open().unwrap(),
        WriteMode::Truncate,
        plain(&[(4, "d", Some("y"), 19_003)]),
        &[],
    )
    .unwrap();
    // Partitioned by a string with NULL and a date; replaceWhere and a dynamic overwrite.
    let env = env_at(&out.join("partitioned"));
    let t = create(&env, &plain_schema(), &["p", "d"], &[]);
    env.write(
        &t,
        WriteMode::Append,
        plain(&[
            (1, "a", Some("x"), 19_000),
            (2, "b", Some("y"), 19_000),
            (3, "c", None, 19_001),
            (4, "\u{1f600}", Some("a b/c"), 19_002),
        ]),
        &[],
    )
    .unwrap();
    env.write(
        &env.open().unwrap(),
        WriteMode::OverwriteByFilter(eq("p", "x")),
        plain(&[(5, "e", Some("x"), 19_005)]),
        &[("__krishiv_delta_predicate", "p = 'x'")],
    )
    .unwrap();
    env.write(
        &env.open().unwrap(),
        WriteMode::OverwriteDynamic,
        plain(&[(6, "f", None, 19_001)]),
        &[],
    )
    .unwrap();
    // Column mapping by name with a struct, an array and a map.
    let env = env_at(&out.join("column-mapping"));
    let t = create(
        &env,
        &nested_schema(),
        &[],
        &[("delta.columnMapping.mode", "name")],
    );
    env.write(&t, WriteMode::Append, nested(&[1, 2]), &[])
        .unwrap();
    env.write(&env.open().unwrap(), WriteMode::Append, nested(&[3]), &[])
        .unwrap();
    // A CTAS (one create commit with rows), partitioned.
    let env = env_at(&out.join("ctas"));
    let t = create(&env, &plain_schema(), &["p"], &[]);
    env.write(
        &t,
        WriteMode::Append,
        plain(&[(1, "a", Some("x"), 19_000), (2, "b", Some("y"), 19_001)]),
        &[],
    )
    .unwrap();
    // A Krishiv checkpoint (interval 2).
    let env = env_at(&out.join("checkpointed"));
    let t = create(
        &env,
        &plain_schema(),
        &["p"],
        &[("delta.checkpointInterval", "2")],
    );
    env.write(
        &t,
        WriteMode::Append,
        plain(&[(1, "a", Some("x"), 19_000)]),
        &[],
    )
    .unwrap();
    for i in 2..=3 {
        env.write(
            &env.open().unwrap(),
            WriteMode::Append,
            plain(&[(i, "n", Some("y"), 19_000)]),
            &[],
        )
        .unwrap();
    }
    let last = env.dir.join("_delta_log/_last_checkpoint");
    for _ in 0..200 {
        if last.exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(last.exists());
}

/// Every row of `batches` as Spark's `to_json(struct(*))` parses (NULL fields left out).
fn json_rows(batches: &[RecordBatch]) -> Vec<Value> {
    let mut out = Vec::new();
    for b in batches {
        let b = conform_strings(b);
        let mut w = arrow_json::ArrayWriter::new(Vec::new());
        w.write(&b).unwrap();
        w.finish().unwrap();
        let rows: Vec<Value> = serde_json::from_slice(&w.into_inner()).unwrap_or_default();
        out.extend(rows);
    }
    out.sort_by_key(|r| r["id"].as_i64());
    out
}

/// `b` with string views as strings (for the JSON writer).
fn conform_strings(b: &RecordBatch) -> RecordBatch {
    let columns: Vec<ArrayRef> = b
        .columns()
        .iter()
        .map(|c| match c.data_type() {
            ArrowType::Utf8View => arrow_cast::cast(c, &ArrowType::Utf8).unwrap(),
            _ => Arc::clone(c),
        })
        .collect();
    let fields: Vec<Field> = b
        .schema()
        .fields()
        .iter()
        .zip(&columns)
        .map(|(f, c)| Field::new(f.name(), c.data_type().clone(), f.is_nullable()))
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

/// Spark 4.2.0 + delta-spark 4.4.1 read every Krishiv-written table: the same rows as Krishiv
/// reads, a history entry per Krishiv commit, and an append of its own on top.
#[test]
fn spark_reads_krishiv_written_tables() {
    let readback: Value =
        serde_json::from_slice(&std::fs::read(written().join("READBACK.json")).unwrap()).unwrap();
    let tables = readback["tables"].as_object().unwrap();
    assert!(tables.len() >= 6, "{:?}", tables.keys());
    for (name, theirs) in tables {
        assert!(theirs["error"].is_null(), "{name}: {}", theirs["error"]);
        let env = Env::fixture(&format!("krishiv-written/{name}"));
        let ours = json_rows(&env.read(&env.open().unwrap()));
        let spark: Vec<Value> = theirs["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| serde_json::from_str(r.as_str().unwrap()).unwrap())
            .collect();
        assert_eq!(ours, spark, "{name}: the rows");
        assert_eq!(
            theirs["history"].as_array().unwrap().len() as u64,
            env.latest() + 1,
            "{name}: a history entry per commit"
        );
        assert!(
            theirs["spark_append"]["error"].is_null(),
            "{name}: Spark appends: {}",
            theirs["spark_append"]
        );
        assert_eq!(
            theirs["spark_append"]["rows"].as_u64(),
            Some(spark.len() as u64 + 1),
            "{name}"
        );
    }
}

/// The Krishiv-created column-mapping table: every file has each field's physical name and id,
/// in the struct and through the array and the map, at every level.
#[test]
fn krishiv_created_column_mapping_files_carry_ids_at_every_level() {
    let env = Env::fixture("krishiv-written/column-mapping");
    let schema = env.open().unwrap().snapshot().schema();
    for version in 0..=env.latest() {
        for add in adds(&env.log(version)) {
            let footer = footer_schema(&env, add["path"].as_str().unwrap());
            let mut checked = Vec::new();
            check_field_ids(&schema, &footer, "", &mut checked);
            for name in ["id", "s", "s.x", "s.y", "a", "m"] {
                assert!(checked.iter().any(|c| c == name), "{name}: {checked:?}");
            }
        }
    }
}
