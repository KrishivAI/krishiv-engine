// SPDX-License-Identifier: Apache-2.0
//! Type promotion and schema evolution through SQL (P3-T026, PR-25c; design
//! note §4): tables whose files were written under an earlier schema, read
//! with the current one. iceberg-rust 0.11 cannot promote, rename or reorder
//! columns, so each table's snapshot and schemas are written here: one data
//! manifest of the given files and a metadata file whose current schema is
//! the last one given.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Decimal128Array, Float32Array, Float64Array, Int32Array, Int64Array};
use arrow_schema::{DataType as A, Field};
use iceberg::io::FileIO;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, ManifestListWriter,
    ManifestWriterBuilder, NestedField, PartitionSpec, PrimitiveType, Schema, Struct, Type,
};

use super::provider_tests::{Env, batch, env, id_field, provider_session, write_parquet};
use super::table::IcebergTable;
use crate::plan::VersionSpec;

/// A data file `name` of `columns` (fields with ids), with bounds by field
/// id, written under the schema `schema` (an index into the table's schemas).
struct File {
    name: &'static str,
    schema: usize,
    fields: Vec<Field>,
    columns: Vec<ArrayRef>,
    lower: HashMap<i32, Datum>,
    upper: HashMap<i32, Datum>,
    nans: HashMap<i32, u64>,
    /// An equality delete file on these field ids (data sequence number 2;
    /// data files have 1).
    equality: Option<Vec<i32>>,
}

impl File {
    fn new(name: &'static str, fields: Vec<Field>, columns: Vec<ArrayRef>) -> Self {
        Self {
            name,
            schema: 0,
            fields,
            columns,
            lower: HashMap::new(),
            upper: HashMap::new(),
            nans: HashMap::new(),
            equality: None,
        }
    }

    /// An equality delete file on `ids`.
    fn deletes(mut self, ids: &[i32]) -> Self {
        self.equality = Some(ids.to_vec());
        self
    }

    fn bounds(mut self, id: i32, lower: Datum, upper: Datum) -> Self {
        self.lower.insert(id, lower);
        self.upper.insert(id, upper);
        self
    }

    /// Written under the schema with index `schema`.
    fn under(mut self, schema: usize) -> Self {
        self.schema = schema;
        self
    }

    fn nans(mut self, id: i32, n: u64) -> Self {
        self.nans.insert(id, n);
        self
    }
}

/// A table created with `schemas[0]` whose current schema is the last of
/// `schemas` (ids 0, 1, ...), with one snapshot holding `files`: one data
/// manifest per schema the files were written under, carrying that schema,
/// as a writer of the time wrote it (a manifest's bounds are decoded with
/// its own schema, so an `int` bound stays 4 bytes).
fn evolved(env: &Env, schemas: &[Schema], files: Vec<File>) -> IcebergTable {
    let t = env.create("t", schemas[0].clone(), None);
    let io: FileIO = env.reg.file_io();
    let data: Vec<(usize, bool, DataFile)> = files
        .into_iter()
        .map(|f| {
            let path = env.at(f.name);
            let rows = f.columns[0].len();
            let size = write_parquet(&path, &batch(f.fields, f.columns));
            let mut builder = DataFileBuilder::default();
            if let Some(ids) = &f.equality {
                builder
                    .content(DataContentType::EqualityDeletes)
                    .equality_ids(Some(ids.clone()));
            } else {
                builder.content(DataContentType::Data);
            }
            let file = builder
                .file_path(path)
                .file_format(DataFileFormat::Parquet)
                .file_size_in_bytes(size)
                .record_count(u64::try_from(rows).unwrap())
                .partition(Struct::empty())
                .partition_spec_id(0)
                .lower_bounds(f.lower)
                .upper_bounds(f.upper)
                .nan_value_counts(f.nans)
                .build()
                .unwrap();
            (f.schema, f.equality.is_some(), file)
        })
        .collect();
    let snapshot_id = 1000;
    let list = env.at("snap.avro");
    env.rt.block_on(async {
        let mut manifests = Vec::new();
        for (i, schema) in schemas.iter().enumerate() {
            for deletes in [false, true] {
                let files: Vec<_> = data
                    .iter()
                    .filter(|(s, d, _)| *s == i && *d == deletes)
                    .collect();
                if files.is_empty() {
                    continue;
                }
                let b = ManifestWriterBuilder::new(
                    io.new_output(env.at(&format!("m{i}-{deletes}.avro")))
                        .unwrap(),
                    Some(snapshot_id),
                    Arc::new(schema.clone()),
                    PartitionSpec::unpartition_spec(),
                );
                let mut w = if deletes {
                    b.build_v2_deletes()
                } else {
                    b.build_v2_data()
                };
                for (_, _, d) in files {
                    w.add_file(d.clone(), if deletes { 2 } else { 1 }).unwrap();
                }
                manifests.push(w.write_manifest_file().await.unwrap());
            }
        }
        let mut lw = ManifestListWriter::v2(
            io.new_output(&list).unwrap().writer().await.unwrap(),
            snapshot_id,
            None,
            2,
        );
        lw.add_manifests(manifests.into_iter()).unwrap();
        lw.close().await.unwrap();
    });
    let location = t.metadata_location().unwrap().to_owned();
    let text = env
        .rt
        .block_on(async { io.new_input(&location).unwrap().read().await.unwrap() });
    let mut json: serde_json::Value = serde_json::from_slice(&text).unwrap();
    let ts = json["last-updated-ms"].as_i64().unwrap() + 1;
    let current_id = schemas.len() - 1;
    json["schemas"] = serde_json::Value::Array(
        schemas
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mut v = serde_json::to_value(s).unwrap();
                v["schema-id"] = serde_json::json!(i);
                v
            })
            .collect(),
    );
    json["current-schema-id"] = serde_json::json!(current_id);
    json["last-column-id"] =
        serde_json::json!(schemas.iter().map(Schema::highest_field_id).max().unwrap());
    json["snapshots"] = serde_json::json!([{
        "snapshot-id": snapshot_id,
        "sequence-number": 2,
        "timestamp-ms": ts,
        "manifest-list": list,
        "summary": {"operation": "append"},
        "schema-id": current_id
    }]);
    json["current-snapshot-id"] = serde_json::json!(snapshot_id);
    json["last-sequence-number"] = serde_json::json!(2);
    json["refs"] = serde_json::json!({"main": {"snapshot-id": snapshot_id, "type": "branch"}});
    json["snapshot-log"] = serde_json::json!([{"snapshot-id": snapshot_id, "timestamp-ms": ts}]);
    let edited = env.at("99-evolved.metadata.json");
    env.rt.block_on(async {
        io.new_output(&edited)
            .unwrap()
            .write(bytes::Bytes::from(serde_json::to_vec(&json).unwrap()))
            .await
            .unwrap();
    });
    env.rt
        .block_on(IcebergTable::load(&env.reg, &edited, "db", "t"))
        .unwrap()
}

fn schema(fields: Vec<NestedField>) -> Schema {
    Schema::builder()
        .with_fields(fields.into_iter().map(Arc::new))
        .build()
        .unwrap()
}

fn opt(id: i32, name: &str, ty: PrimitiveType) -> NestedField {
    NestedField::optional(id, name, Type::Primitive(ty))
}

/// The sorted rows of `sql` over `table`.
fn rows(env: &Env, table: &IcebergTable, sql: &str) -> Vec<String> {
    let p = env.provider(table, &VersionSpec::Current).unwrap();
    let mut out = provider_session(p).rows(sql);
    out.sort();
    out
}

/// iceberg-rust #3482 (design note §4): a column promoted from `int` to
/// `long`, with a file written under each type, each with bounds in its own
/// width. Equality, range and `IN` filters on values of the old file return
/// its rows: the old file's 4-byte bounds are read as `int`, never as part
/// of a `long`, so pruning keeps it.
#[test]
fn promoted_column_filter_returns_all_rows() {
    let env = env();
    let old = id_field("v", A::Int32, 1);
    let new = id_field("v", A::Int64, 1);
    let table = evolved(
        &env,
        &[
            schema(vec![opt(1, "v", PrimitiveType::Int)]),
            schema(vec![opt(1, "v", PrimitiveType::Long)]),
        ],
        vec![
            File::new(
                "old.parquet",
                vec![old],
                vec![Arc::new(Int32Array::from(vec![5, 6, 7]))],
            )
            .bounds(1, Datum::int(5), Datum::int(7)),
            File::new(
                "new.parquet",
                vec![new],
                vec![Arc::new(Int64Array::from(vec![8, 9, 7_000_000_000]))],
            )
            .bounds(1, Datum::long(8_i64), Datum::long(7_000_000_000_i64))
            .under(1),
        ],
    );
    // Pruning alone (the filter above the scan would hide a wrongly skipped
    // file only by returning fewer rows): every comparison a row of the old
    // file passes opens it.
    use krishiv_io::{CmpOp, ColumnPath, PushdownExpr, ScalarValue};
    let p = env.provider(&table, &VersionSpec::Current).unwrap();
    let cmp = |op, v| PushdownExpr::cmp(ColumnPath::new("v"), op, ScalarValue::Int64(v));
    assert_eq!(p.files_read(cmp(CmpOp::Eq, 6)), 1);
    assert_eq!(p.files_read(cmp(CmpOp::LtEq, 5)), 1);
    assert_eq!(p.files_read(cmp(CmpOp::GtEq, 7)), 2);
    assert_eq!(p.files_read(cmp(CmpOp::Lt, 5)), 0);
    assert_eq!(p.files_read(cmp(CmpOp::Gt, 7_000_000_000)), 0);
    let q = |w: &str| rows(&env, &table, &format!("SELECT v FROM t WHERE {w}"));
    assert_eq!(q("v = 6"), ["6"]);
    assert_eq!(q("v = 5"), ["5"]);
    assert_eq!(q("v BETWEEN 6 AND 8"), ["6", "7", "8"]);
    assert_eq!(q("v > 6"), ["7", "7000000000", "8", "9"]);
    assert_eq!(q("v IN (7, 7000000000)"), ["7", "7000000000"]);
    assert_eq!(q("v < 5"), Vec::<String>::new());
}

/// `float` promoted to `double`, the old file holding a NaN (its bounds
/// exclude NaN, so it is never pruned by them): `=`, `>` and NaN itself.
#[test]
fn promoted_float_column_keeps_nan_rows() {
    let env = env();
    let table = evolved(
        &env,
        &[
            schema(vec![opt(1, "x", PrimitiveType::Float)]),
            schema(vec![opt(1, "x", PrimitiveType::Double)]),
        ],
        vec![
            File::new(
                "old.parquet",
                vec![id_field("x", A::Float32, 1)],
                vec![Arc::new(Float32Array::from(vec![1.5, f32::NAN, 2.5]))],
            )
            .bounds(1, Datum::float(1.5_f32), Datum::float(2.5_f32))
            .nans(1, 1),
            File::new(
                "new.parquet",
                vec![id_field("x", A::Float64, 1)],
                vec![Arc::new(Float64Array::from(vec![1.5, 3.25]))],
            )
            .bounds(1, Datum::double(1.5), Datum::double(3.25))
            .under(1),
        ],
    );
    let q = |w: &str| rows(&env, &table, &format!("SELECT x FROM t WHERE {w}"));
    assert_eq!(q("x = 1.5D"), ["1.5", "1.5"]);
    assert_eq!(q("x > 2.0D"), ["2.5", "3.25", "NaN"]);
    assert_eq!(q("x = double('NaN')"), ["NaN"]);
    assert_eq!(q("isnan(x)"), ["NaN"]);
}

/// `decimal(9,2)` promoted to `decimal(18,2)` (precision only).
#[test]
fn promoted_decimal_column_reads_both_files() {
    let env = env();
    let dec = |p: u8, v: Vec<i128>| -> ArrayRef {
        Arc::new(
            Decimal128Array::from(v)
                .with_precision_and_scale(p, 2)
                .unwrap(),
        )
    };
    let table = evolved(
        &env,
        &[
            schema(vec![opt(
                1,
                "d",
                PrimitiveType::Decimal {
                    precision: 9,
                    scale: 2,
                },
            )]),
            schema(vec![opt(
                1,
                "d",
                PrimitiveType::Decimal {
                    precision: 18,
                    scale: 2,
                },
            )]),
        ],
        vec![
            File::new(
                "old.parquet",
                vec![id_field("d", A::Decimal128(9, 2), 1)],
                vec![dec(9, vec![125, 250])],
            ),
            File::new(
                "new.parquet",
                vec![id_field("d", A::Decimal128(18, 2), 1)],
                vec![dec(18, vec![125, 9_999_999_999_999])],
            )
            .under(1),
        ],
    );
    let q = |w: &str| rows(&env, &table, &format!("SELECT d FROM t WHERE {w}"));
    assert_eq!(q("d = 1.25"), ["125e-2", "125e-2"]);
    assert_eq!(q("d > 2"), ["250e-2", "9999999999999e-2"]);
}

/// Design note §4: schema evolution is by field id end to end. A file
/// written under the first schema is read with a second that renames a
/// column, reorders the top level, drops a struct field and adds one, adds a
/// column, drops a column and adds another under its old name (a new id, so
/// old files read null), and makes a required column optional.
#[test]
fn iceberg_schema_evolution_matrix() {
    use arrow_array::{StringArray, StructArray};
    use arrow_schema::Fields;
    use iceberg::spec::StructType as IceStruct;

    let env = env();
    let first = Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(opt(2, "name", PrimitiveType::String)),
            Arc::new(NestedField::optional(
                3,
                "s",
                Type::Struct(IceStruct::new(vec![
                    Arc::new(opt(4, "a", PrimitiveType::Int)),
                    Arc::new(opt(5, "b", PrimitiveType::String)),
                ])),
            )),
            Arc::new(opt(9, "gone", PrimitiveType::Int)),
        ])
        .build()
        .unwrap();
    let second = schema(vec![
        NestedField::optional(
            3,
            "s",
            Type::Struct(IceStruct::new(vec![
                Arc::new(opt(4, "a", PrimitiveType::Int)),
                Arc::new(opt(6, "c", PrimitiveType::Int)),
            ])),
        ),
        opt(2, "label", PrimitiveType::String),
        opt(1, "id", PrimitiveType::Long),
        opt(7, "extra", PrimitiveType::Int),
        opt(8, "gone", PrimitiveType::String),
    ]);
    let old_s = Fields::from(vec![id_field("a", A::Int32, 4), id_field("b", A::Utf8, 5)]);
    let old = File::new(
        "old.parquet",
        vec![
            id_field("id", A::Int64, 1),
            id_field("name", A::Utf8, 2),
            id_field("s", A::Struct(old_s.clone()), 3),
            id_field("gone", A::Int32, 9),
        ],
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(StringArray::from(vec!["x"])),
            Arc::new(StructArray::new(
                old_s,
                vec![
                    Arc::new(Int32Array::from(vec![10])),
                    Arc::new(StringArray::from(vec!["dropped"])),
                ],
                None,
            )),
            Arc::new(Int32Array::from(vec![99])),
        ],
    );
    let new_s = Fields::from(vec![id_field("a", A::Int32, 4), id_field("c", A::Int32, 6)]);
    let new = File::new(
        "new.parquet",
        vec![
            id_field("s", A::Struct(new_s.clone()), 3),
            id_field("label", A::Utf8, 2),
            id_field("id", A::Int64, 1),
            id_field("extra", A::Int32, 7),
            id_field("gone", A::Utf8, 8),
        ],
        vec![
            Arc::new(StructArray::new(
                new_s,
                vec![
                    Arc::new(Int32Array::from(vec![20])),
                    Arc::new(Int32Array::from(vec![30])),
                ],
                None,
            )),
            Arc::new(StringArray::from(vec!["y"])),
            Arc::new(Int64Array::from(vec![2])),
            Arc::new(Int32Array::from(vec![7])),
            Arc::new(StringArray::from(vec!["back"])),
        ],
    )
    .under(1);
    let table = evolved(&env, &[first, second], vec![old, new]);
    assert_eq!(
        rows(
            &env,
            &table,
            "SELECT id, label, s.a, s.c, extra, gone FROM t ORDER BY id"
        ),
        ["1|x|10|NULL|NULL|NULL", "2|y|20|30|7|back"]
    );
    // The table's order of columns is the current schema's.
    let p = env.provider(&table, &VersionSpec::Current).unwrap();
    let names: Vec<String> = p
        .snapshot()
        .schema
        .fields()
        .iter()
        .map(|f| f.name().to_owned())
        .collect();
    assert_eq!(names, ["s", "label", "id", "extra", "gone"]);
    // A filter on the renamed column finds the old file's row by id.
    assert_eq!(
        rows(&env, &table, "SELECT id FROM t WHERE label = 'x'"),
        ["1"]
    );
}

/// Equality keys end to end (design note §3.3): a key written as `int` and
/// compared with the column promoted to `long`, a string key, a key nested
/// in a struct, and a key column added after the data file was written
/// (read as null there, so a non-null key never matches it).
#[test]
fn equality_keys_of_every_shape_delete_rows() {
    use arrow_array::{StringArray, StructArray};
    use arrow_schema::Fields;
    use iceberg::spec::StructType as IceStruct;

    let env = env();
    let s_type = || {
        Type::Struct(IceStruct::new(vec![Arc::new(opt(
            4,
            "a",
            PrimitiveType::Int,
        ))]))
    };
    let first = schema(vec![
        opt(1, "v", PrimitiveType::Int),
        opt(2, "name", PrimitiveType::String),
        NestedField::optional(3, "s", s_type()),
    ]);
    let second = schema(vec![
        opt(1, "v", PrimitiveType::Long),
        opt(2, "name", PrimitiveType::String),
        NestedField::optional(3, "s", s_type()),
        opt(5, "c", PrimitiveType::Int),
    ]);
    let s_fields = Fields::from(vec![id_field("a", A::Int32, 4)]);
    let s_col = |v: Vec<i32>| -> ArrayRef {
        Arc::new(StructArray::new(
            s_fields.clone(),
            vec![Arc::new(Int32Array::from(v))],
            None,
        ))
    };
    let old = File::new(
        "old.parquet",
        vec![
            id_field("v", A::Int32, 1),
            id_field("name", A::Utf8, 2),
            id_field("s", A::Struct(s_fields.clone()), 3),
        ],
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
            s_col(vec![10, 20, 30, 40, 50]),
        ],
    );
    let new = File::new(
        "new.parquet",
        vec![
            id_field("v", A::Int64, 1),
            id_field("name", A::Utf8, 2),
            id_field("s", A::Struct(s_fields.clone()), 3),
            id_field("c", A::Int32, 5),
        ],
        vec![
            Arc::new(Int64Array::from(vec![6, 7])),
            Arc::new(StringArray::from(vec!["f", "g"])),
            s_col(vec![60, 70]),
            Arc::new(Int32Array::from(vec![Some(1), None])),
        ],
    )
    .under(1);
    let deletes = vec![
        // Written when `v` was an int: deletes 2 (and 7 is not in it).
        File::new(
            "eq-v.parquet",
            vec![id_field("v", A::Int32, 1)],
            vec![Arc::new(Int32Array::from(vec![2]))],
        )
        .deletes(&[1]),
        // A string key: deletes "c".
        File::new(
            "eq-name.parquet",
            vec![id_field("name", A::Utf8, 2)],
            vec![Arc::new(StringArray::from(vec!["c", "zz"]))],
        )
        .deletes(&[2])
        .under(1),
        // A nested key: deletes s.a = 40.
        File::new(
            "eq-s.parquet",
            vec![id_field("s", A::Struct(s_fields.clone()), 3)],
            vec![s_col(vec![40])],
        )
        .deletes(&[4])
        .under(1),
        // `c` = 1: only the new file's first row; the old file has no `c`.
        File::new(
            "eq-c.parquet",
            vec![id_field("c", A::Int32, 5)],
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .deletes(&[5])
        .under(1),
    ];
    let mut files = vec![old, new];
    files.extend(deletes);
    let table = evolved(&env, &[first, second], files);
    assert_eq!(
        rows(&env, &table, "SELECT v, name FROM t"),
        ["1|a", "5|e", "7|g"]
    );
}
