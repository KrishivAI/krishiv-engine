// SPDX-License-Identifier: Apache-2.0
//! Nanosecond timestamps read from Parquet (Spark 4.3, P3-T090) at the format level.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_schema::DataType;

use super::ParquetFormat;
use super::tests::Env;
use crate::format::MetadataNeeds;
use crate::request::ScanRequest;

/// `nanosAsLong` types every `INT64 TIMESTAMP(NANOS)` leaf `bigint` at any depth, as Spark's
/// `ParquetToSparkSchemaConverter` does per leaf, also in a file that holds an `INT96` column (which
/// arrow-rs decodes as nanoseconds too, and which stays a timestamp).
#[test]
fn nanos_as_long_retypes_each_nanosecond_leaf_beside_an_int96_column() {
    use parquet::data_type::{Int64Type, Int96, Int96Type};
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;
    let message = "message m {
        required int96 i96;
        required group s { required int64 t (TIMESTAMP(NANOS,false)); }
        required int64 top (TIMESTAMP(NANOS,true));
    }";
    let schema = Arc::new(parse_message_type(message).unwrap());
    let mut buf = Vec::new();
    let mut w = SerializedFileWriter::new(&mut buf, schema, Arc::default()).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    let mut i96 = Int96::new();
    // 2020-01-01: Julian day 2_458_850, zero nanoseconds into it.
    i96.set_data(0, 0, 2_458_850);
    col.typed::<Int96Type>()
        .write_batch(&[i96], None, None)
        .unwrap();
    col.close().unwrap();
    for _ in 0..2 {
        let mut col = rg.next_column().unwrap().unwrap();
        col.typed::<Int64Type>()
            .write_batch(&[1_577_836_800_123_456_789], None, None)
            .unwrap();
        col.close().unwrap();
    }
    rg.close().unwrap();
    w.close().unwrap();

    let env = Env::new();
    let file = env.put("int96-nested.parquet", buf);
    let o = env.options(&[("nanosAsLong", "true")]);
    let schema = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
        )
        .unwrap()
        .schema();
    assert!(
        matches!(
            schema.field(0).data_type(),
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _)
        ),
        "{schema:?}"
    );
    let DataType::Struct(s) = schema.field(1).data_type() else {
        panic!("{schema:?}")
    };
    assert_eq!(s[0].data_type(), &DataType::Int64, "{schema:?}");
    assert_eq!(schema.field(2).data_type(), &DataType::Int64, "{schema:?}");
    // The decode retypes the nested leaf too: the stored nanoseconds.
    let out = env.scan(&file, &ScanRequest::all(schema), &o).unwrap();
    let nested = out[0].column(1).as_struct().column(0);
    assert_eq!(
        nested
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(0),
        1_577_836_800_123_456_789
    );
}

/// `nanosTypesKnown` changes how the footer decodes, so it keys the metadata cache: a Spark 4.2
/// profile (`false`, the footer's nanosecond types dropped, `nanosAsLong` makes them `bigint`) and a
/// 4.3 one (the footer's types kept) over the same file and `IoContext` never share an entry.
#[test]
fn nanos_types_known_keys_the_metadata_cache() {
    let bytes = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/parquet/nanos/spark-flat.parquet"),
    )
    .unwrap();
    let env = Env::new();
    let file = env.put("spark-flat.parquet", bytes);
    let n9 = |kv: &[(&str, &str)]| {
        let o = env.options(kv);
        let schema = env
            .rt
            .block_on(
                env.io
                    .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
            )
            .unwrap()
            .schema();
        schema.field_with_name("n9").unwrap().data_type().clone()
    };
    let older = n9(&[("nanosAsLong", "true"), ("nanosTypesKnown", "false")]);
    assert_eq!(older, DataType::Int64);
    let newer = n9(&[("nanosAsLong", "true")]);
    assert!(
        matches!(newer, DataType::Timestamp(..)),
        "a 4.3 read took the 4.2 profile's cached metadata: {newer:?}"
    );
}
