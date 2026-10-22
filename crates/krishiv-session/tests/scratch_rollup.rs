//! scratch
#![cfg(test)]
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType as T, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

#[test]
fn scratch() {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", T::Int32, false),
        Field::new("b", T::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3])),
            Arc::new(Int32Array::from(vec![1, 2, 1, 2, 1, 2])),
        ],
    )
    .unwrap();
    s.register_batches("td", schema, vec![batch]).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("course", T::Utf8, false),
        Field::new("year", T::Int32, false),
        Field::new("earnings", T::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["dotNET", "Java", "dotNET", "dotNET", "Java"])),
            Arc::new(Int32Array::from(vec![2012, 2012, 2012, 2013, 2013])),
            Arc::new(Int32Array::from(vec![10000, 20000, 5000, 48000, 30000])),
        ],
    )
    .unwrap();
    s.register_batches("cs", schema, vec![batch]).unwrap();
    for q in std::fs::read_to_string("/home/krishiv/krishiv-data/tmp/s-rollup/q.sql")
        .unwrap()
        .split(";\n")
        .filter(|q| !q.trim().is_empty())
    {
        match block_on(s.sql(q)) {
            Err(e) => println!("ERR {q}\n  {e}"),
            Ok(df) => match block_on(df.collect()) {
                Err(e) => println!("RUNERR {q}\n  {e}"),
                Ok(b) => {
                    println!("OK {q}");
                    for b in &b {
                        println!("{}", arrow_cast_fmt(b));
                    }
                }
            },
        }
    }
}

fn arrow_cast_fmt(b: &RecordBatch) -> String {
    let mut out = format!("{:?}\n", b.schema().fields().iter().map(|f| format!("{}:{}{}", f.name(), f.data_type(), if f.is_nullable() {"?"} else {""})).collect::<Vec<_>>());
    for r in 0..b.num_rows() {
        let cells: Vec<String> = b
            .columns()
            .iter()
            .map(|c| {
                if c.is_null(r) {
                    "NULL".to_owned()
                } else {
                    let d = arrow_array::cast::AsArray::as_primitive_opt::<arrow_array::types::Int32Type>(c)
                        .map(|a| a.value(r).to_string())
                        .or_else(|| arrow_array::cast::AsArray::as_primitive_opt::<arrow_array::types::Int64Type>(c).map(|a| a.value(r).to_string()))
                        .or_else(|| arrow_array::cast::AsArray::as_primitive_opt::<arrow_array::types::Int8Type>(c).map(|a| a.value(r).to_string()))
                        .or_else(|| arrow_array::cast::AsArray::as_string_opt::<i32>(c).map(|a| a.value(r).to_owned()));
                    d.unwrap_or_else(|| "?".to_owned())
                }
            })
            .collect();
        out.push_str(&cells.join("\t"));
        out.push('\n');
    }
    out
}
