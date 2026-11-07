// SPDX-License-Identifier: Apache-2.0
//! `GEOMETRY` and `GEOGRAPHY` columns in Parquet files (P2b row 4, `krishiv-io` `parquet/geo.rs`),
//! against a real Spark 4.2.0 run (`research/2026-11-07-spark-parquet-geo.md`): the logical
//! type and CRS string a written column gets, the SRID a CRS spelling reads as, and the errors.
#![expect(clippy::unwrap_used, reason = "test code")]

use std::fs::File;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_session::Session;
use parquet::basic::{LogicalType, Repetition, Type as Physical};
use parquet::data_type::ByteArray;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::FileReader;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::Type;

const POINT: &str = "X'0101000000000000000000F03F0000000000000040'";

fn rows(s: &Session, sql: &str) -> Result<Vec<String>, String> {
    let df = block_on(s.sql(sql)).map_err(|e| e.to_string())?;
    let batches = block_on(df.collect()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for b in &batches {
        let c = b.column(0);
        for i in 0..b.num_rows() {
            if c.is_null(i) {
                out.push("NULL".to_owned());
            } else if let Some(st) = c.as_struct_opt() {
                let srid = st.column(0).as_primitive::<Int32Type>().value(i);
                let wkb = st.column(1).as_binary_view().value(i);
                out.push(krishiv_types::geo::to_ewkt(srid, wkb).map_err(|e| e.to_string())?);
            } else {
                out.push(format!("<{}>", c.data_type()));
            }
        }
    }
    Ok(out)
}

fn first_file(dir: &std::path::Path) -> std::path::PathBuf {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|x| x == "parquet"))
        .unwrap()
}

fn schema_text(file: &std::path::Path) -> String {
    let reader =
        parquet::file::reader::SerializedFileReader::new(File::open(file).unwrap()).unwrap();
    let mut out = Vec::new();
    parquet::schema::printer::print_schema(
        &mut out,
        reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .root_schema(),
    );
    String::from_utf8(out).unwrap()
}

#[test]
fn written_columns_carry_spark_s_logical_type_and_crs() {
    let s = Session::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out");
    let df = block_on(s.sql(&format!(
        "SELECT ST_GeomFromWKB({POINT}) g0, ST_GeomFromWKB({POINT}, 3857) g3857, \
         ST_GeomFromWKB({POINT}, 4326) g4326, ST_GeomFromWKB({POINT}, 4269) g4269, \
         ST_GeogFromWKB({POINT}) h4326"
    )))
    .unwrap();
    block_on(df.write().parquet(out.to_str().unwrap())).unwrap();
    let text = schema_text(&first_file(&out));
    for want in [
        "GEOMETRY(SRID:0)",
        "GEOMETRY(EPSG:3857)",
        "GEOMETRY(OGC:CRS84)",
        "GEOMETRY(OGC:CRS83)",
        "GEOGRAPHY(SPHERICAL, OGC:CRS84)",
    ] {
        assert!(text.contains(want), "{want} not in {text}");
    }
}

#[test]
fn a_written_column_reads_back_with_its_srid_and_nulls() {
    let s = Session::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out");
    let df = block_on(s.sql(&format!(
        "SELECT CASE WHEN i = 1 THEN NULL ELSE ST_GeomFromWKB({POINT}, 3857) END g \
         FROM range(3) t(i) ORDER BY i"
    )))
    .unwrap();
    block_on(df.write().parquet(out.to_str().unwrap())).unwrap();
    let got = rows(&s, &format!("SELECT g FROM parquet.`{}` ", out.display())).unwrap();
    assert_eq!(
        got,
        ["SRID=3857;POINT(1 2)", "NULL", "SRID=3857;POINT(1 2)"]
    );
    let geog = dir.path().join("geog");
    let df = block_on(s.sql(&format!("SELECT ST_GeogFromWKB({POINT}) g"))).unwrap();
    block_on(df.write().parquet(geog.to_str().unwrap())).unwrap();
    let got = rows(&s, &format!("SELECT g FROM parquet.`{}`", geog.display())).unwrap();
    assert_eq!(got, ["SRID=4326;POINT(1 2)"]);
}

#[test]
fn an_any_srid_column_is_refused_as_spark_refuses_it() {
    let s = Session::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let df = block_on(s.sql(&format!(
        "SELECT CAST(ST_GeomFromWKB({POINT}, 4326) AS GEOMETRY(ANY)) g"
    )))
    .unwrap();
    let e = block_on(df.write().parquet(dir.path().join("o").to_str().unwrap())).unwrap_err();
    assert!(
        e.to_string()
            .contains("UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE"),
        "{e}"
    );
}

/// A file with one optional BINARY column `g` annotated `logical`, holding `values`.
fn crafted(path: &std::path::Path, logical: Option<LogicalType>, values: &[Option<&[u8]>]) {
    let mut column = Type::primitive_type_builder("g", Physical::BYTE_ARRAY)
        .with_repetition(Repetition::OPTIONAL);
    if let Some(l) = logical {
        column = column.with_logical_type(Some(l));
    }
    let schema = Type::group_type_builder("m")
        .with_fields(vec![Arc::new(column.build().unwrap())])
        .build()
        .unwrap();
    let mut w = SerializedFileWriter::new(
        File::create(path).unwrap(),
        Arc::new(schema),
        Arc::new(WriterProperties::builder().build()),
    )
    .unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    let defs: Vec<i16> = values.iter().map(|v| i16::from(v.is_some())).collect();
    let data: Vec<ByteArray> = values
        .iter()
        .flatten()
        .map(|v| ByteArray::from(v.to_vec()))
        .collect();
    col.typed::<parquet::data_type::ByteArrayType>()
        .write_batch(&data, Some(&defs), None)
        .unwrap();
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();
}

fn point() -> Vec<u8> {
    let mut v = vec![1_u8, 1, 0, 0, 0];
    v.extend_from_slice(&1.0_f64.to_le_bytes());
    v.extend_from_slice(&2.0_f64.to_le_bytes());
    v
}

fn read_crafted(
    logical: Option<LogicalType>,
    values: &[Option<&[u8]>],
) -> Result<Vec<String>, String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let file = dir.path().join("f.parquet");
    crafted(&file, logical, values);
    let s = Session::new().map_err(|e| e.to_string())?;
    rows(&s, &format!("SELECT g FROM parquet.`{}`", file.display()))
}

#[test]
fn a_crs_spelling_reads_as_spark_reads_it() {
    let p = point();
    let one = [Some(p.as_slice())];
    for (crs, want) in [
        (None, "SRID=4326;POINT(1 2)"),
        (Some("EPSG:3857"), "SRID=3857;POINT(1 2)"),
        (Some("EPSG:4326"), "SRID=4326;POINT(1 2)"),
        (Some("OGC:CRS84"), "SRID=4326;POINT(1 2)"),
        (Some("SRID:0"), "POINT(1 2)"),
    ] {
        let got = read_crafted(Some(LogicalType::geometry(crs.map(str::to_owned))), &one);
        assert_eq!(got, Ok(vec![want.to_owned()]), "{crs:?}");
    }
    // Case-sensitive; `srid:N` and PROJJSON are no registry names.
    for crs in [
        "srid:0",
        "srid:3857",
        "srid:4326",
        "{\"type\":\"ProjectedCRS\",\"name\":\"x\"}",
    ] {
        let e = read_crafted(Some(LogicalType::geometry(Some(crs.to_owned()))), &one).unwrap_err();
        assert!(e.contains("ST_INVALID_CRS_VALUE"), "{crs}: {e}");
    }
    // A geography needs a geographic system.
    let ok = read_crafted(
        Some(LogicalType::geography(Some("OGC:CRS84".to_owned()), None)),
        &one,
    );
    assert_eq!(ok, Ok(vec!["SRID=4326;POINT(1 2)".to_owned()]));
    for crs in ["EPSG:3857", "SRID:0"] {
        let e = read_crafted(
            Some(LogicalType::geography(Some(crs.to_owned()), None)),
            &one,
        )
        .unwrap_err();
        assert!(e.contains("ST_INVALID_CRS_VALUE"), "{crs}: {e}");
    }
}

#[test]
fn nulls_big_endian_and_malformed_values() {
    let p = point();
    let mut big = vec![0_u8, 0, 0, 0, 1];
    big.extend_from_slice(&1.0_f64.to_be_bytes());
    big.extend_from_slice(&2.0_f64.to_be_bytes());
    let got = read_crafted(
        Some(LogicalType::geometry(None)),
        &[Some(p.as_slice()), None, Some(big.as_slice())],
    );
    assert_eq!(
        got,
        Ok(vec![
            "SRID=4326;POINT(1 2)".to_owned(),
            "NULL".to_owned(),
            "SRID=4326;POINT(1 2)".to_owned()
        ])
    );
    let e = read_crafted(
        Some(LogicalType::geometry(None)),
        &[Some(&[0xff_u8, 0][..])],
    )
    .unwrap_err();
    // Spark: `FAILED_READ_FILE.NO_HINT` caused by `WKB_PARSE_ERROR`.
    assert!(e.contains("FAILED_READ_FILE.NO_HINT"), "{e}");
}

#[test]
fn a_plain_binary_column_stays_binary() {
    let p = point();
    let got = read_crafted(None, &[Some(p.as_slice())]).unwrap();
    assert_eq!(got, ["<BinaryView>"]);
}
