// SPDX-License-Identifier: Apache-2.0
//! `st_geomfromwkb`, `st_geogfromwkb`, `st_asbinary`, `st_srid`,
//! `st_setsrid` and the casts between the spatial types, checked against
//! the Spark 4.2.0 `st-functions` golden.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

const POINT: &str = "X'0101000000000000000000F03F0000000000000040'";

fn run(sql: &str) -> Result<Option<String>, String> {
    let s = Session::new().map_err(|e| e.to_string())?;
    let df = block_on(s.sql(sql)).map_err(|e| e.to_string())?;
    let b = block_on(df.collect()).map_err(|e| e.to_string())?;
    let c = &b[0].columns()[0];
    let Some(strings) = c.as_string_view_opt() else {
        return Err(format!("it returned a {}", c.data_type()));
    };
    Ok((!c.is_null(0)).then(|| strings.value(0).to_owned()))
}

fn text(sql: &str) -> String {
    run(&format!("SELECT CAST(({sql}) AS STRING)"))
        .unwrap()
        .unwrap()
}

fn error(sql: &str) -> String {
    run(&format!("SELECT {sql}")).unwrap_err()
}

#[test]
fn the_constructors_type_their_result() {
    assert_eq!(
        text(&format!("typeof(ST_GeomFromWKB({POINT}))")),
        "geometry(0)"
    );
    assert_eq!(
        text(&format!("typeof(ST_GeomFromWKB({POINT}, 4326))")),
        "geometry(4326)"
    );
    assert_eq!(
        text(&format!("typeof(ST_GeogFromWKB({POINT}))")),
        "geography(4326)"
    );
    assert_eq!(
        text(&format!(
            "typeof(ST_SetSrid(ST_GeomFromWKB({POINT}), 3857))"
        )),
        "geometry(3857)"
    );
    assert_eq!(text("typeof(ST_Srid(NULL))"), "int");
}

#[test]
fn srid_and_binary_round_trip() {
    assert_eq!(text(&format!("ST_Srid(ST_GeomFromWKB({POINT}))")), "0");
    assert_eq!(text(&format!("ST_Srid(ST_GeogFromWKB({POINT}))")), "4326");
    assert_eq!(
        text(&format!(
            "ST_Srid(ST_SetSrid(ST_GeomFromWKB({POINT}), 3857))"
        )),
        "3857"
    );
    assert_eq!(
        text(&format!("hex(ST_AsBinary(ST_GeomFromWKB({POINT}, 3857)))")),
        "0101000000000000000000F03F0000000000000040"
    );
    // Big-endian in, little-endian out; `XDR` writes it back.
    let xdr = "X'00000000013FF00000000000004000000000000000'";
    assert_eq!(
        text(&format!("hex(ST_AsBinary(ST_GeomFromWKB({xdr}), 'NDR'))")),
        "0101000000000000000000F03F0000000000000040"
    );
    assert_eq!(
        text(&format!("hex(ST_AsBinary(ST_GeomFromWKB({xdr}), 'XDR'))")),
        "00000000013FF00000000000004000000000000000"
    );
    assert_eq!(text("ST_AsBinary(ST_GeogFromWKB(NULL)) IS NULL"), "true");
    assert_eq!(text("ST_Srid(NULL) IS NULL"), "true");
}

#[test]
fn endianness_may_be_a_column() {
    let g = format!("ST_GeomFromWKB({POINT})");
    let from = |v: &str| format!("(VALUES ({v})) t(e)");
    let q = |v: &str| format!("SELECT hex(ST_AsBinary({g}, e)) FROM {}", from(v));
    assert_eq!(
        run(&q("'xdr'")).unwrap().unwrap(),
        "00000000013FF00000000000004000000000000000"
    );
    assert_eq!(
        run(&q("'NDR'")).unwrap().unwrap(),
        "0101000000000000000000F03F0000000000000040"
    );
    assert_eq!(run(&q("CAST(NULL AS STRING)")).unwrap(), None);
    let e = run(&q("'big'")).unwrap_err();
    assert!(e.contains("ST_INVALID_ENDIANNESS_VALUE"), "{e}");
    // A NULL spatial row is NULL even when its endianness is invalid.
    let null_row = format!(
        "SELECT hex(ST_AsBinary(IF(i = 0, CAST(NULL AS GEOMETRY(4326)), {g}), \
         IF(i = 0, 'bad', 'NDR'))) FROM range(2) t(i) ORDER BY i"
    );
    assert_eq!(run(&null_row).unwrap(), None);
}

#[test]
fn bad_input_is_refused_as_spark_refuses_it() {
    let e = error("ST_GeogFromWKB(X'6F')");
    assert!(e.contains("WKB_PARSE_ERROR"), "{e}");
    assert!(e.contains("Unexpected end of WKB buffer"), "{e}");
    for srid in ["-1", "9999"] {
        let e = error(&format!("ST_GeomFromWKB({POINT}, {srid})"));
        assert!(e.contains("ST_INVALID_SRID_VALUE"), "{e}");
    }
    let e = error(&format!("ST_SetSrid(ST_GeogFromWKB({POINT}), 3857)"));
    assert!(e.contains("ST_INVALID_SRID_VALUE"), "{e}");
    for end in ["''", "'ABC'", "'big-endian'"] {
        let e = error(&format!("ST_AsBinary(ST_GeogFromWKB({POINT}), {end})"));
        assert!(e.contains("ST_INVALID_ENDIANNESS_VALUE"), "{e}");
    }
}

#[test]
fn casts_follow_the_spatial_rules() {
    let g = format!("ST_GeogFromWKB({POINT})");
    let m = format!("ST_GeomFromWKB({POINT})");
    let hex = |e: &str| text(&format!("hex(ST_AsBinary({e}))"));
    let point = "0101000000000000000000F03F0000000000000040";
    assert_eq!(hex(&format!("CAST({g} AS GEOGRAPHY(ANY))")), point);
    assert_eq!(hex(&format!("CAST({g} AS GEOMETRY(4326))")), point);
    assert_eq!(hex(&format!("CAST({m} AS GEOMETRY(ANY))")), point);
    assert_eq!(
        hex(&format!(
            "CAST(ST_GeomFromWKB({POINT}, 4326) AS GEOGRAPHY(4326))"
        )),
        point
    );
    for bad in [
        format!("CAST({g} AS STRING)"),
        format!("CAST({POINT} AS GEOMETRY(4326))"),
        format!("CAST(CAST({g} AS GEOGRAPHY(ANY)) AS GEOGRAPHY(4326))"),
        format!("CAST({g} AS GEOMETRY(ANY))"),
        format!("CAST(CAST({m} AS GEOMETRY(ANY)) AS GEOMETRY(4326))"),
        format!("CAST({m} AS GEOGRAPHY(4326))"),
    ] {
        let e = error(&bad);
        assert!(
            e.contains("DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION"),
            "{bad}: {e}"
        );
    }
    // The coordinates must be geographic.
    let e = error(
        "CAST(ST_GeomFromWKB(X'010100000000000000000069400000000000005940', 4326) AS GEOGRAPHY(4326))",
    );
    assert!(e.contains("Invalid coordinate value found"), "{e}");
    // TRY_CAST gives NULL for the invalid value and keeps the valid ones.
    let bad = "ST_GeomFromWKB(X'010100000000000000000069400000000000005940', 4326)";
    assert!(
        run(&format!(
            "SELECT hex(ST_AsBinary(TRY_CAST({bad} AS GEOGRAPHY(4326))))"
        ))
        .unwrap()
        .is_none()
    );
    let ok = "ST_GeomFromWKB(X'0101000000000000000000F03F0000000000000040', 4326)";
    assert_eq!(
        text(&format!(
            "hex(ST_AsBinary(TRY_CAST({ok} AS GEOGRAPHY(4326))))"
        )),
        point
    );
}

#[test]
fn mixed_srids_meet_at_any() {
    let g = format!("ST_GeogFromWKB({POINT})");
    assert_eq!(
        text(&format!("typeof(array({g}, {g}::GEOGRAPHY(ANY)))")),
        "array<geography(any)>"
    );
    assert_eq!(
        text(&format!("typeof(coalesce({g}, {g}::GEOGRAPHY(ANY)))")),
        "geography(any)"
    );
}

#[test]
fn spatial_values_are_no_keys() {
    let m = format!("ST_GeomFromWKB({POINT})");
    let e = error(&format!("{m} = {m}"));
    assert!(e.contains("INVALID_ORDERING_TYPE"), "{e}");
    for q in [
        format!("max({m})"),
        format!("min({m})"),
        format!("count(DISTINCT {m})"),
        format!("collect_set({m})"),
    ] {
        let e = error(&q);
        assert!(e.contains("GEOMETRY or GEOGRAPHY"), "{q}: {e}");
    }
    // A spatial field read out of a struct is no key either.
    let field = format!("(SELECT named_struct('g', {m}) s) t");
    for q in [
        format!("SELECT count(*) FROM {field} GROUP BY s.g"),
        format!("SELECT count(DISTINCT s.g) FROM {field}"),
    ] {
        let e = run(&q).unwrap_err();
        assert!(e.contains("GEOMETRY"), "{q}: {e}");
    }
    for q in [
        format!("SELECT DISTINCT {m}"),
        format!("SELECT {m} g GROUP BY {m}"),
    ] {
        let e = run(&q).unwrap_err();
        assert!(e.contains("GEOMETRY"), "{q}: {e}");
    }
}

#[test]
fn spatial_values_are_not_written_to_files() {
    let s = Session::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let df = block_on(s.sql(&format!("SELECT ST_GeomFromWKB({POINT}) g"))).unwrap();
    let e = block_on(df.write().parquet(dir.path().join("out").to_str().unwrap())).unwrap_err();
    assert!(e.to_string().contains("GEOMETRY or GEOGRAPHY"), "{e}");
}

#[test]
fn an_srid_outside_int_is_named_as_written() {
    let e = error(&format!("ST_GeomFromWKB({POINT}, 5000000000)"));
    assert!(
        e.contains("ST_INVALID_SRID_VALUE") || e.contains("DATATYPE_MISMATCH"),
        "{e}"
    );
    assert!(!e.contains("2147483648"), "{e}");
}

#[test]
fn codex_review_fixes() {
    let m = format!("ST_GeomFromWKB({POINT})");
    // A NULL endianness gives NULL.
    assert_eq!(
        run(&format!(
            "SELECT CAST(hex(ST_AsBinary({m}, CAST(NULL AS STRING))) AS STRING)"
        )),
        Ok(None)
    );
    // Collection functions refuse spatial elements.
    for q in [
        format!("sort_array(array({m}))"),
        format!("array_distinct(array({m}))"),
        format!("array_contains(array({m}), {m})"),
    ] {
        let e = error(&q);
        assert!(!e.is_empty(), "{q}");
    }
    // Nested geography casts check coordinates.
    let bad = "ST_GeomFromWKB(X'010100000000000000000069400000000000005940', 4326)";
    let e = error(&format!("CAST(array({bad}) AS ARRAY<GEOGRAPHY(4326)>)"));
    assert!(e.contains("Invalid coordinate value found"), "{e}");
    // An ordinary struct shaped like the carrier is written.
    let s = Session::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let df = block_on(s.sql("SELECT named_struct('srid', 1, 'wkb', X'01') v")).unwrap();
    block_on(df.write().parquet(dir.path().join("out").to_str().unwrap())).unwrap();
}
