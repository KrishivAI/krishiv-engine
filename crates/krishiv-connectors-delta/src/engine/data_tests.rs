// SPDX-License-Identifier: Apache-2.0
//! `EngineData` / `GetData` tests (cases after the kernel's `arrow_data.rs` tests).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{ListBuilder, MapBuilder, StringBuilder};
use arrow_array::{
    ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray, ListArray, NullArray,
    RecordBatch, StringArray, StringViewArray, StructArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType as ArrowType, Field, Fields, Schema};
use delta_kernel::EngineData;
use delta_kernel::engine_data::{GetData, RowVisitor, TypedGetData as _};
use delta_kernel::expressions::{ArrayData, ColumnName, Scalar};
use delta_kernel::schema::{ArrayType, DataType, MapType, StructField, StructType};

use super::*;

/// Records what a visitor saw, as strings, for each requested leaf.
struct Collect {
    names: &'static [ColumnName],
    types: &'static [DataType],
    rows: usize,
    seen: Vec<Vec<String>>,
}

fn columns(spec: &[(&str, DataType)]) -> Collect {
    let names: Vec<ColumnName> = spec
        .iter()
        .map(|(n, _)| ColumnName::new(n.split('.')))
        .collect();
    let types: Vec<DataType> = spec.iter().map(|(_, t)| t.clone()).collect();
    Collect {
        names: Box::leak(names.into_boxed_slice()),
        types: Box::leak(types.into_boxed_slice()),
        rows: 0,
        seen: Vec::new(),
    }
}

fn show<'a>(g: &'a dyn GetData<'a>, t: &DataType, row: usize) -> KernelResult<String> {
    let name = "c";
    Ok(match t {
        &DataType::BOOLEAN => format!("{:?}", g.get_bool(row, name)?),
        &DataType::BYTE => format!("{:?}", g.get_byte(row, name)?),
        &DataType::SHORT => format!("{:?}", g.get_short(row, name)?),
        &DataType::INTEGER => format!("{:?}", g.get_int(row, name)?),
        &DataType::LONG => format!("{:?}", g.get_long(row, name)?),
        &DataType::FLOAT => format!("{:?}", g.get_float(row, name)?),
        &DataType::DOUBLE => format!("{:?}", g.get_double(row, name)?),
        &DataType::DATE => format!("{:?}", g.get_date(row, name)?),
        &DataType::TIMESTAMP | &DataType::TIMESTAMP_NTZ => {
            format!("{:?}", g.get_timestamp(row, name)?)
        }
        &DataType::STRING => format!("{:?}", g.get_str(row, name)?),
        &DataType::BINARY => format!("{:?}", g.get_binary(row, name)?),
        DataType::Primitive(_) => format!("{:?}", g.get_decimal(row, name)?),
        DataType::Array(_) => {
            let v: Option<Vec<String>> = g.get_opt(row, name)?;
            format!("{v:?}")
        }
        DataType::Map(_) => {
            let v: Option<HashMap<String, String>> = g.get_opt(row, name)?;
            let mut v: Option<Vec<(String, String)>> = v.map(|m| m.into_iter().collect());
            if let Some(v) = v.as_mut() {
                v.sort();
            }
            format!("{v:?}")
        }
        other => format!("unsupported {other}"),
    })
}

impl RowVisitor for Collect {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        (self.names, self.types)
    }

    fn visit<'a>(&mut self, row_count: usize, getters: &[&'a dyn GetData<'a>]) -> KernelResult<()> {
        self.rows = row_count;
        for (g, t) in getters.iter().zip(self.types) {
            let mut col = Vec::new();
            for row in 0..row_count {
                col.push(show(*g, t, row)?);
            }
            self.seen.push(col);
        }
        Ok(())
    }
}

fn batch(cols: Vec<(&str, ArrayRef)>) -> RecordBatch {
    RecordBatch::try_from_iter(cols).unwrap()
}

fn visit(data: &KrishivEngineData, spec: &[(&str, DataType)]) -> KernelResult<Vec<Vec<String>>> {
    let mut v = columns(spec);
    v.visit_rows_of(data)?;
    assert_eq!(v.rows, data.len());
    Ok(v.seen)
}

#[test]
fn every_getter_reads_its_type_and_nulls() {
    let data = KrishivEngineData::new(batch(vec![
        (
            "bo",
            Arc::new(BooleanArray::from(vec![Some(true), None])) as ArrayRef,
        ),
        ("b", Arc::new(Int8Array::from(vec![Some(1), None]))),
        ("sh", Arc::new(Int16Array::from(vec![Some(2), None]))),
        ("i", Arc::new(Int32Array::from(vec![Some(3), None]))),
        ("l", Arc::new(Int64Array::from(vec![Some(4), None]))),
        ("f", Arc::new(Float32Array::from(vec![Some(0.5), None]))),
        ("d", Arc::new(Float64Array::from(vec![Some(1.5), None]))),
        ("da", Arc::new(Date32Array::from(vec![Some(19000), None]))),
        (
            "ts",
            Arc::new(TimestampMicrosecondArray::from(vec![Some(7), None]).with_timezone("UTC")),
        ),
        (
            "ntz",
            Arc::new(TimestampMicrosecondArray::from(vec![Some(8), None])),
        ),
        (
            "dec",
            Arc::new(
                Decimal128Array::from(vec![Some(12345), None])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
        ),
        ("s", Arc::new(StringArray::from(vec![Some("a"), None]))),
        (
            "ls",
            Arc::new(LargeStringArray::from(vec![Some("b"), None])),
        ),
        ("vs", Arc::new(StringViewArray::from(vec![Some("c"), None]))),
        (
            "bi",
            Arc::new(BinaryArray::from(vec![Some(&b"x"[..]), None])),
        ),
        ("n", Arc::new(NullArray::new(2))),
    ]));
    let spec = [
        ("bo", DataType::BOOLEAN),
        ("b", DataType::BYTE),
        ("sh", DataType::SHORT),
        ("i", DataType::INTEGER),
        ("l", DataType::LONG),
        ("f", DataType::FLOAT),
        ("d", DataType::DOUBLE),
        ("da", DataType::DATE),
        ("ts", DataType::TIMESTAMP),
        ("ntz", DataType::TIMESTAMP_NTZ),
        ("dec", DataType::decimal(10, 2).unwrap()),
        ("s", DataType::STRING),
        ("ls", DataType::STRING),
        ("vs", DataType::STRING),
        ("bi", DataType::BINARY),
        ("n", DataType::LONG),
    ];
    let seen = visit(&data, &spec).unwrap();
    let first: Vec<&str> = seen.iter().map(|c| c[0].as_str()).collect();
    assert_eq!(
        first,
        [
            "Some(true)",
            "Some(1)",
            "Some(2)",
            "Some(3)",
            "Some(4)",
            "Some(0.5)",
            "Some(1.5)",
            "Some(19000)",
            "Some(7)",
            "Some(8)",
            "Some(12345)",
            "Some(\"a\")",
            "Some(\"b\")",
            "Some(\"c\")",
            "Some([120])",
            "None"
        ]
    );
    assert!(seen.iter().all(|c| c[1] == "None"));
}

#[test]
fn a_type_mismatch_or_missing_column_is_an_error() {
    let data = KrishivEngineData::new(batch(vec![(
        "i",
        Arc::new(Int32Array::from(vec![1])) as ArrayRef,
    )]));
    let e = visit(&data, &[("i", DataType::LONG)]).unwrap_err();
    assert!(e.to_string().contains("Type mismatch on i"), "{e}");
    assert!(visit(&data, &[("j", DataType::LONG)]).is_err());
    assert!(visit(&data, &[("i", DataType::INTEGER), ("i", DataType::INTEGER)]).is_err());
    assert!(
        visit(
            &data,
            &[("i", DataType::INTEGER), ("i.x", DataType::INTEGER)]
        )
        .is_err()
    );
}

fn nested() -> KrishivEngineData {
    // add: struct<path: string, size: long, partitionValues: map<string,string>, tags: list<string>>
    let mut map = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    map.keys().append_value("p");
    map.values().append_value("1");
    map.keys().append_value("q");
    map.values().append_null();
    map.append(true).unwrap();
    map.append(false).unwrap();
    let map = map.finish();
    let mut list = ListBuilder::new(StringBuilder::new());
    list.values().append_value("t1");
    list.values().append_value("t2");
    list.append(true);
    list.append(false);
    let list = list.finish();
    let add = StructArray::from(vec![
        (
            Arc::new(Field::new("path", ArrowType::Utf8, true)),
            Arc::new(StringArray::from(vec![Some("a.parquet"), None])) as ArrayRef,
        ),
        (
            Arc::new(Field::new("size", ArrowType::Int64, true)),
            Arc::new(Int64Array::from(vec![Some(10), None])) as ArrayRef,
        ),
        (
            Arc::new(Field::new("partitionValues", map.data_type().clone(), true)),
            Arc::new(map) as ArrayRef,
        ),
        (
            Arc::new(Field::new("tags", list.data_type().clone(), true)),
            Arc::new(list) as ArrayRef,
        ),
    ]);
    KrishivEngineData::new(batch(vec![
        ("add", Arc::new(add) as ArrayRef),
        ("v", Arc::new(Int32Array::from(vec![1, 2]))),
    ]))
}

#[test]
fn nested_leaves_lists_and_maps() {
    let data = nested();
    let seen = visit(
        &data,
        &[
            ("v", DataType::INTEGER),
            ("add.path", DataType::STRING),
            ("add.size", DataType::LONG),
            (
                "add.partitionValues",
                MapType::new(DataType::STRING, DataType::STRING, true).into(),
            ),
            ("add.tags", ArrayType::new(DataType::STRING, false).into()),
        ],
    )
    .unwrap();
    assert_eq!(seen[0], ["Some(1)", "Some(2)"]);
    assert_eq!(seen[1], ["Some(\"a.parquet\")", "None"]);
    assert_eq!(seen[2], ["Some(10)", "None"]);
    // `materialize` drops null values, as the kernel's does.
    assert_eq!(seen[3], ["Some([(\"p\", \"1\")])", "None"]);
    assert_eq!(seen[4], ["Some([\"t1\", \"t2\"])", "None"]);
    assert!(data.has_field(&ColumnName::new(["add", "size"])));
    assert!(!data.has_field(&ColumnName::new(["add", "nope"])));
    assert!(!data.has_field(&ColumnName::new(["v", "x"])));
    assert!(!data.has_field(&ColumnName::new(Vec::<String>::new())));
}

/// Visits element structs through `get_struct_list`.
struct Outer {
    inner: Vec<Vec<String>>,
}

static OUTER_NAMES: std::sync::LazyLock<[ColumnName; 1]> =
    std::sync::LazyLock::new(|| [ColumnName::new(["l"])]);
static OUTER_TYPES: std::sync::LazyLock<[DataType; 1]> = std::sync::LazyLock::new(|| {
    [ArrayType::new(
        StructType::try_new([StructField::nullable("x", DataType::INTEGER)]).unwrap(),
        false,
    )
    .into()]
});

impl RowVisitor for Outer {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        (&*OUTER_NAMES, &*OUTER_TYPES)
    }

    fn visit<'a>(&mut self, row_count: usize, getters: &[&'a dyn GetData<'a>]) -> KernelResult<()> {
        for row in 0..row_count {
            match getters[0].get_struct_list(row, "l")? {
                None => self.inner.push(vec!["null".to_owned()]),
                Some(list) => {
                    let mut v = columns(&[("x", DataType::INTEGER)]);
                    list.visit_with(&mut v)?;
                    self.inner
                        .push(v.seen.into_iter().next().unwrap_or_default());
                }
            }
        }
        Ok(())
    }
}

#[test]
fn a_list_of_structs_visits_its_elements() {
    let elements = StructArray::from(vec![(
        Arc::new(Field::new("x", ArrowType::Int32, true)),
        Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
    )]);
    let field = Arc::new(Field::new("item", elements.data_type().clone(), false));
    let list = ListArray::new(
        field,
        arrow_buffer::OffsetBuffer::from_lengths([2, 0, 1]),
        Arc::new(elements),
        Some(vec![true, false, true].into()),
    );
    let data = KrishivEngineData::new(batch(vec![("l", Arc::new(list) as ArrayRef)]));
    let mut v = Outer { inner: Vec::new() };
    v.visit_rows_of(&data).unwrap();
    assert_eq!(
        v.inner,
        [vec!["Some(1)", "Some(2)"], vec!["null"], vec!["Some(3)"]]
    );
    // A declared nullable element is refused up front.
    let nullable = ArrayType::new(
        StructType::try_new([StructField::nullable("x", DataType::INTEGER)]).unwrap(),
        true,
    );
    assert!(visit(&data, &[("l", nullable.into())]).is_err());
}

#[test]
fn append_columns_adds_kernel_arrays() {
    let data = KrishivEngineData::new(batch(vec![(
        "a",
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
    )]));
    let schema =
        Arc::new(StructType::try_new([StructField::nullable("b", DataType::STRING)]).unwrap());
    let column = ArrayData::try_new(
        ArrayType::new(DataType::STRING, true),
        [Scalar::from("x"), Scalar::Null(DataType::STRING)],
    )
    .unwrap();
    let out = data
        .append_columns(Arc::clone(&schema), vec![column])
        .unwrap();
    let out = KrishivEngineData::batch_of(out.as_ref()).unwrap();
    assert_eq!(
        out.schema().field(1),
        &Field::new("b", ArrowType::Utf8, true)
    );
    assert_eq!(out.column(1).null_count(), 1);
    // A column per field, of the data's length.
    assert!(data.append_columns(Arc::clone(&schema), vec![]).is_err());
    let short =
        ArrayData::try_new(ArrayType::new(DataType::STRING, true), [Scalar::from("x")]).unwrap();
    assert!(data.append_columns(schema, vec![short]).is_err());
}

#[test]
fn a_short_selection_vector_keeps_the_rest_and_a_long_one_fails() {
    let rows = || {
        Box::new(KrishivEngineData::new(batch(vec![(
            "a",
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])) as ArrayRef,
        )])))
    };
    let kept = |data: Box<dyn EngineData>| -> Vec<i32> {
        KrishivEngineData::batch_of(data.as_ref())
            .unwrap()
            .column(0)
            .as_primitive::<Int32Type>()
            .values()
            .to_vec()
    };
    assert_eq!(
        kept(rows().apply_selection_vector(vec![false, true]).unwrap()),
        [2, 3, 4]
    );
    assert_eq!(
        kept(rows().apply_selection_vector(vec![]).unwrap()),
        [1, 2, 3, 4]
    );
    assert_eq!(
        kept(
            rows()
                .apply_selection_vector(vec![true, false, true, false])
                .unwrap()
        ),
        [1, 3]
    );
    let e = rows().apply_selection_vector(vec![true; 5]).err().unwrap();
    assert!(matches!(e, KernelError::InvalidSelectionVector(_)), "{e}");
}

#[test]
fn engine_data_round_trips_through_a_box() {
    let b = batch(vec![("a", Arc::new(Int32Array::from(vec![1])) as ArrayRef)]);
    let boxed: Box<dyn EngineData> = Box::new(KrishivEngineData::from(b.clone()));
    assert_eq!(
        KrishivEngineData::from_engine_data(boxed)
            .unwrap()
            .into_batch(),
        b
    );
    let st = StructArray::from(b.clone());
    assert_eq!(KrishivEngineData::from(st).batch(), &b);
    let _ = Schema::new(Fields::empty());
}

/// A null struct row hides its children, even where they hold values (valid Arrow): an absent
/// action must not look present. Two levels: `a` is null at row 1, `a.b` at row 0.
#[test]
fn a_null_struct_hides_its_children() {
    let mut map = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    for _ in 0..3 {
        map.keys().append_value("k");
        map.values().append_value("v");
        map.append(true).unwrap();
    }
    let map = map.finish();
    let mut list = ListBuilder::new(StringBuilder::new());
    for _ in 0..3 {
        list.values().append_value("t");
        list.append(true);
    }
    let list = list.finish();
    let b = StructArray::new(
        Fields::from(vec![
            Field::new("s", ArrowType::Utf8, true),
            Field::new("n", ArrowType::Int64, true),
        ]),
        vec![
            Arc::new(StringArray::from(vec!["x", "y", "z"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
        Some(vec![false, true, true].into()),
    );
    let a = StructArray::new(
        Fields::from(vec![
            Field::new("b", b.data_type().clone(), true),
            Field::new("m", map.data_type().clone(), true),
            Field::new("l", list.data_type().clone(), true),
            Field::new("i", ArrowType::Int32, true),
        ]),
        vec![
            Arc::new(b) as ArrayRef,
            Arc::new(map),
            Arc::new(list),
            Arc::new(Int32Array::from(vec![7, 8, 9])),
        ],
        Some(vec![true, false, true].into()),
    );
    let data = KrishivEngineData::new(batch(vec![("a", Arc::new(a) as ArrayRef)]));
    let seen = visit(
        &data,
        &[
            ("a.b.s", DataType::STRING),
            ("a.b.n", DataType::LONG),
            (
                "a.m",
                MapType::new(DataType::STRING, DataType::STRING, true).into(),
            ),
            ("a.l", ArrayType::new(DataType::STRING, true).into()),
            ("a.i", DataType::INTEGER),
        ],
    )
    .unwrap();
    assert_eq!(seen[0], ["None", "None", "Some(\"z\")"]);
    assert_eq!(seen[1], ["None", "None", "Some(3)"]);
    assert_eq!(
        seen[2],
        ["Some([(\"k\", \"v\")])", "None", "Some([(\"k\", \"v\")])"]
    );
    assert_eq!(seen[3], ["Some([\"t\"])", "None", "Some([\"t\"])"]);
    assert_eq!(seen[4], ["Some(7)", "None", "Some(9)"]);
}
