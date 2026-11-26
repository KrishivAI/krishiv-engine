// SPDX-License-Identifier: Apache-2.0
//! A column handed in as a `FixedSizeList` of floats runs as a vector carrier (design P3-T075): expressions and the
//! caller see Spark's `array<float>` (a plain list), a bare column passes through, and a fixed-size list that is not
//! a vector (other elements, too long) is an ordinary list from the start.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int32Type};
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, Int32Array, RecordBatch};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

/// The three-row vector table, its elements nullable or not (`containsNull`).
fn vectors_with(nullable: bool) -> (Arc<Schema>, RecordBatch) {
    let item = Arc::new(Field::new("item", DataType::Float32, nullable));
    let values = Float32Array::from(vec![
        1.0, 2.0, 3.0, //
        0.0, 0.0, 0.0, // the null row's slot
        7.0, 8.0, 9.0,
    ]);
    let v: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(
            Arc::clone(&item),
            3,
            Arc::new(values),
            Some(NullBuffer::from(vec![true, false, true])),
        )
        .unwrap(),
    );
    let id: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("v", DataType::FixedSizeList(item, 3), true),
    ]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![id, v]).unwrap();
    (schema, batch)
}

fn session() -> Session {
    session_with(true)
}

fn session_with(nullable: bool) -> Session {
    let s = Session::new().unwrap();
    let (schema, batch) = vectors_with(nullable);
    s.register_batches("t", schema, vec![batch]).unwrap();
    s
}

fn rows(s: &Session, sql: &str) -> Vec<RecordBatch> {
    block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap()
}

/// What a caller reads is Spark's plain list, with the same values and the null row.
#[test]
fn a_caller_reads_a_vector_column_as_a_plain_list() {
    let s = session();
    let df = block_on(s.sql("SELECT id, v FROM t ORDER BY id")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert!(
        matches!(schema.field(1).data_type(), DataType::List(f) if f.data_type() == &DataType::Float32),
        "{schema:?}"
    );
    let batches = block_on(df.collect()).unwrap();
    let v = batches[0].column(1).as_list::<i32>();
    assert!(v.is_valid(0) && v.is_null(1) && v.is_valid(2));
    assert_eq!(
        v.value(2).as_primitive::<Float32Type>().values(),
        &[7.0, 8.0, 9.0]
    );
}

/// Functions over arrays read a vector column as they read a list: the result is the one a list column gives.
#[test]
fn array_functions_over_a_vector_column_match_a_list_column() {
    twin_check(true);
}

/// A vector of non-null elements (`containsNull = false`, design §9.1) behaves as the plain list of it too.
#[test]
fn array_functions_over_a_vector_of_non_null_elements_match_a_list_column() {
    twin_check(false);
}

fn twin_check(nullable: bool) {
    let s = session_with(nullable);
    let (_, batch) = vectors_with(nullable);
    let plain = krishiv_exec::carrier::recarried_array(
        batch.column(1),
        &DataType::List(Arc::new(Field::new("element", DataType::Float32, nullable))),
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("v", plain.data_type().clone(), true),
    ]));
    let as_list = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::clone(batch.column(0)), plain],
    )
    .unwrap();
    s.register_batches("l", schema, vec![as_list]).unwrap();
    for sql in [
        "SELECT size(v) FROM {} ORDER BY id",
        "SELECT array_max(v) FROM {} ORDER BY id",
        "SELECT v[1] FROM {} ORDER BY id",
        "SELECT id FROM {} WHERE array_contains(v, 8.0) ORDER BY id",
        "SELECT id, explode(v) FROM {} ORDER BY id, 2",
        "SELECT count(v), max(size(v)) FROM {}",
        "SELECT id, v FROM {} ORDER BY id DESC",
        // Hash, equality, grouping and joining read the value, not the carrier (design P3-T075, hash and sort keys).
        // A bare vector column through the operators that move rows (design P3-T075 §9.2).
        "SELECT id, v FROM {} WHERE id > 1 ORDER BY id",
        "SELECT id, v FROM {} ORDER BY id LIMIT 2",
        "SELECT id, v FROM (SELECT id, v FROM {} DISTRIBUTE BY id) ORDER BY id",
        "SELECT id, v FROM {} UNION ALL SELECT id, v FROM {} ORDER BY id",
        "SELECT id, v FROM (SELECT id, v FROM {} ORDER BY id DESC LIMIT 2) ORDER BY id",
        // Aggregate, window and subquery inputs read the plain list too.
        "SELECT collect_list(v) FROM {}",
        "SELECT first(v), last(v) FROM {}",
        "SELECT max(v) FROM {}",
        "SELECT id, lag(v) OVER (ORDER BY id) FROM {} ORDER BY id",
        "SELECT (SELECT v FROM {} WHERE id = 1)",
        "SELECT hash(v), xxhash64(v) FROM {} ORDER BY id",
        "SELECT v = v, v <=> v FROM {} ORDER BY id",
        "SELECT count(DISTINCT v) FROM {}",
        "SELECT count(*) FROM (SELECT v FROM {} GROUP BY v)",
        "SELECT a.id, b.id FROM {} a JOIN {} b ON a.v = b.v ORDER BY a.id, b.id",
    ] {
        // A query the engine refuses for a list column is refused for a vector column with the same error class.
        let render = |table: &str| {
            let sql = sql.replace("{}", table);
            match block_on(s.sql(&sql)).and_then(|df| block_on(df.collect())) {
                Ok(batches) => batches
                    .iter()
                    .map(|b| format!("{:?}", b.columns()))
                    .collect::<Vec<_>>(),
                Err(e) => vec![format!("{:?}", e.error_class())],
            }
        };
        assert_eq!(render("t"), render("l"), "{sql}");
    }
}

/// A fixed-size list of anything but floats, or too long to be a vector, enters as a plain list.
#[test]
fn a_fixed_size_list_that_is_not_a_vector_is_an_ordinary_list() {
    let s = Session::new().unwrap();
    let item = Arc::new(Field::new("item", DataType::Int32, true));
    let v: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(
            Arc::clone(&item),
            2,
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            None,
        )
        .unwrap(),
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::FixedSizeList(item, 2),
        false,
    )]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![v]).unwrap();
    s.register_batches("ints", schema, vec![batch]).unwrap();
    let out = rows(&s, "SELECT v FROM ints");
    let list = out[0].column(0).as_list::<i32>();
    assert_eq!(list.value(1).as_primitive::<Int32Type>().values(), &[3, 4]);
}

fn vector_table(s: &Session, name: &str, n: i32, rows: &[&[f32]]) {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let v: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(
            Arc::clone(&item),
            n,
            Arc::new(Float32Array::from(flat)),
            None,
        )
        .unwrap(),
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::FixedSizeList(item, n),
        true,
    )]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![v]).unwrap();
    s.register_batches(name, schema, vec![batch]).unwrap();
}

fn lists(batches: &[RecordBatch]) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    for b in batches {
        let l = b.column(0).as_list::<i32>();
        for i in 0..l.len() {
            out.push(l.value(i).as_primitive::<Float32Type>().values().to_vec());
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

/// A union of vectors of one length is a vector column, of two lengths or with a plain list a list column
/// (design P3-T075 §4): the values are right either way.
#[test]
fn a_union_of_vectors_keeps_values_whatever_the_lengths() {
    let s = Session::new().unwrap();
    vector_table(&s, "a", 2, &[&[1.0, 2.0], &[3.0, 4.0]]);
    vector_table(&s, "b", 2, &[&[5.0, 6.0]]);
    vector_table(&s, "c", 3, &[&[7.0, 8.0, 9.0]]);
    assert_eq!(
        lists(&rows(&s, "SELECT v FROM a UNION ALL SELECT v FROM b")),
        vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]
    );
    assert_eq!(
        lists(&rows(&s, "SELECT v FROM a UNION ALL SELECT v FROM c")),
        vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![7.0, 8.0, 9.0]]
    );
    assert_eq!(
        lists(&rows(
            &s,
            "SELECT v FROM a UNION ALL SELECT array(CAST(0.5 AS FLOAT), CAST(1.5 AS FLOAT))"
        )),
        vec![vec![0.5, 1.5], vec![1.0, 2.0], vec![3.0, 4.0]]
    );
}

/// Written to Parquet and read back by path (which keeps the file's carrier) and through a table declared as
/// `array<float>` (which reads the plain list), a vector column gives the same values (design P3-T075 §4, §6).
#[test]
fn a_vector_column_through_parquet_reads_the_same_by_path_and_by_table() {
    let s = session();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vectors");
    let path = path.to_str().unwrap();
    block_on(
        block_on(s.sql("SELECT id, v FROM t"))
            .unwrap()
            .write()
            .parquet(path),
    )
    .unwrap();
    let by_path = rows(
        &s,
        &format!("SELECT id, size(v), v FROM parquet.`{path}` ORDER BY id"),
    );
    block_on(s.sql(&format!(
        "CREATE TABLE vt (id INT, v ARRAY<FLOAT>) USING parquet LOCATION '{}'",
        // `/` separators: a Windows path's `\` would be read as a SQL escape.
        path.replace('\\', "/")
    )))
    .map(|df| block_on(df.collect()))
    .unwrap()
    .unwrap();
    let by_table = rows(&s, "SELECT id, size(v), v FROM vt ORDER BY id");
    let render = |b: &[RecordBatch]| {
        b.iter()
            .map(|b| format!("{:?}", b.columns()))
            .collect::<Vec<_>>()
    };
    assert_eq!(render(&by_path), render(&by_table));
    assert_eq!(by_path[0].num_rows(), 3);
}

/// `carrier_kind(v)`: 1 when the argument array a UDF receives is a plain list, 2 when it is a fixed-size list.
#[derive(Debug)]
struct CarrierKind;

impl krishiv_session::ScalarUdf for CarrierKind {
    fn name(&self) -> &str {
        "carrier_kind"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &krishiv_session::Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<krishiv_session::Signature> = OnceLock::new();
        SIG.get_or_init(|| {
            krishiv_session::Signature::Exact(vec![krishiv_session::UdfInputType::Any])
        })
    }

    fn return_field(
        &self,
        _: &[krishiv_session::ArgField],
    ) -> Result<krishiv_session::ReturnField, krishiv_session::UdfError> {
        Ok(krishiv_session::ReturnField {
            data_type: krishiv_types::DataType::Integer,
            nullable: false,
        })
    }

    fn invoke(
        &self,
        args: &[krishiv_session::UdfArg<'_>],
        rows: usize,
        _: &krishiv_session::InvokeContext,
    ) -> Result<ArrayRef, krishiv_session::UdfError> {
        let kind = match args[0].array().data_type() {
            DataType::List(_) => 1,
            DataType::FixedSizeList(..) => 2,
            _ => 0,
        };
        Ok(Arc::new(Int32Array::from(vec![kind; rows])))
    }
}

/// A UDF sees Spark's plain list for a vector column (the session edge), never the fixed-size carrier.
#[test]
fn a_udf_receives_a_vector_column_as_a_plain_list() {
    let s = session();
    s.register_udf(CarrierKind).unwrap();
    let batches = rows(&s, "SELECT carrier_kind(v) FROM t ORDER BY id");
    let kinds: Vec<i32> = batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect();
    assert_eq!(kinds, [1, 1, 1]);
}

/// A vector column carried through a sort under a memory limit small enough to spill (design P3-T075, review M2):
/// the rows come back whole and in order. A list is no grouping or join key in the engine, so the carrier is
/// never hashed as one (`array_functions_over_a_vector_column_match_a_list_column` checks the refusal).
#[test]
fn a_vector_column_survives_a_spilling_sort() {
    const ROWS: u16 = 60_000;
    const N: usize = 16;
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("spill"); // the session creates it, mode 0700
    let s = Session::builder()
        .config("krishiv.spill.dir", dir.to_str().unwrap())
        .config("krishiv.memory.limit", "3m")
        .build()
        .unwrap();
    // Row i starts with ROWS - i, so the sort reverses the input.
    let data: Vec<Vec<f32>> = (0..ROWS)
        .map(|i| {
            let mut r = vec![0.0; N];
            r[0] = f32::from(ROWS - i);
            r
        })
        .collect();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    vector_table(&s, "big", i32::try_from(N).unwrap(), &refs);

    let got: Vec<Vec<f32>> = rows(&s, "SELECT v FROM big ORDER BY v[0]")
        .iter()
        .flat_map(|b| {
            let l = b.column(0).as_list::<i32>();
            (0..l.len())
                .map(|i| l.value(i).as_primitive::<Float32Type>().values().to_vec())
                .collect::<Vec<_>>()
        })
        .collect();
    let mut want = data.clone();
    want.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(got, want);
}
