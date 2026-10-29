// SPDX-License-Identifier: Apache-2.0
//! Struct functions end to end: `named_struct` and `struct` construction,
//! `STRUCT(...)` syntax, field access under `spark.sql.caseSensitive` and
//! struct comparison. Construction, field access and error classes follow
//! the matrix and golden fixtures (real Spark 4.2 runs); comparison treats a
//! null field as the smallest value and never returns NULL for a null field
//! (checked against a Spark 4.2 run).
//! `MAX`/`MIN` over structs follow the comparison order (field by field,
//! NULL fields smallest); window `MAX`/`MIN` over structs too.
//! Not here: star-except in `STRUCT`.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::functions::{call_function, col, count_star, lit};
use krishiv_session::{Column, DataFrame, ExplainMode, Session};

fn session() -> Session {
    Session::new().unwrap()
}

/// Every row of `sql`'s first column, rendered as Spark renders values
/// (`{"a":1}`, `NULL` at the top level).
fn rows(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for batch in block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}")) {
        let col = batch.column(0);
        for row in 0..batch.num_rows() {
            out.push(cell(col, row));
        }
    }
    out
}

fn error(s: &Session, sql: &str) -> ErrorClass {
    match block_on(s.sql(sql)) {
        Ok(_) => panic!("{sql} resolved"),
        Err(e) => e.error_class(),
    }
}

/// One value of `col` at `row`, as Spark renders it (`{"a":1}`, `NULL` at
/// the top level, `null` for a NULL field).
fn cell(col: &arrow_array::ArrayRef, row: usize) -> String {
    use arrow_array::cast::AsArray as _;

    if col.is_null(row) {
        return "NULL".to_owned();
    }
    match col.data_type() {
        arrow_schema::DataType::Null => "null".to_owned(),
        arrow_schema::DataType::Boolean => col.as_boolean().value(row).to_string(),
        arrow_schema::DataType::Int32 => col
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(row)
            .to_string(),
        arrow_schema::DataType::Int64 => col
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(row)
            .to_string(),
        arrow_schema::DataType::Utf8View => col.as_string_view().value(row).to_owned(),
        arrow_schema::DataType::Utf8 => col.as_string::<i32>().value(row).to_owned(),
        arrow_schema::DataType::Float64 => format!(
            "{:?}",
            col.as_primitive::<arrow_array::types::Float64Type>()
                .value(row)
        ),
        arrow_schema::DataType::Decimal128(_, scale) => {
            let v = col
                .as_primitive::<arrow_array::types::Decimal128Type>()
                .value(row);
            dec_text(v, *scale)
        }
        arrow_schema::DataType::Struct(_) => {
            let s = col.as_struct();
            let fields = s
                .columns()
                .iter()
                .enumerate()
                .map(|(i, child)| {
                    let name = s.column_names().get(i).cloned().unwrap_or_default();
                    let value = if child.is_null(row) {
                        "null".to_owned()
                    } else {
                        match child.data_type() {
                            arrow_schema::DataType::Utf8View => {
                                format!("\"{}\"", child.as_string_view().value(row))
                            }
                            arrow_schema::DataType::Utf8 => {
                                format!("\"{}\"", child.as_string::<i32>().value(row))
                            }
                            _ => cell(child, row),
                        }
                    };
                    format!("\"{name}\":{value}")
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{fields}}}")
        }
        other => panic!("unrendered test value of {other}"),
    }
}

/// The unscaled decimal `v` at `scale`, as Spark prints it.
fn dec_text(v: i128, scale: i8) -> String {
    let scale = usize::try_from(scale).unwrap_or(0);
    let digits = v.unsigned_abs().to_string();
    let sign = if v < 0 { "-" } else { "" };
    if scale == 0 {
        return format!("{sign}{digits}");
    }
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (int, frac) = padded.split_at(padded.len() - scale);
    format!("{sign}{int}.{frac}")
}

/// Every row of `sql`, all columns joined by `|` (top-level NULLs are
/// `NULL`).
fn rows_all(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for batch in block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}")) {
        for row in 0..batch.num_rows() {
            out.push(
                (0..batch.num_columns())
                    .map(|c| cell(batch.column(c), row))
                    .collect::<Vec<_>>()
                    .join("|"),
            );
        }
    }
    out
}

#[test]
fn named_struct_and_struct_construct_structs() {
    let s = session();
    assert_eq!(
        rows(&s, "SELECT named_struct('a', 1, 'b', 'x')"),
        ["{\"a\":1,\"b\":\"x\"}"]
    );
    assert_eq!(
        rows(&s, "SELECT struct(1, 'a')"),
        ["{\"col1\":1,\"col2\":\"a\"}"]
    );
    assert_eq!(
        rows(&s, "SELECT STRUCT('alpha', 'beta')"),
        ["{\"col1\":\"alpha\",\"col2\":\"beta\"}"]
    );
    assert_eq!(
        rows(&s, "SELECT STRUCT('alpha' AS A, 'beta' AS B)"),
        ["{\"A\":\"alpha\",\"B\":\"beta\"}"]
    );
    // Null values are fields, not null structs.
    assert_eq!(
        rows(&s, "SELECT named_struct('a', NULL, 'b', 2)"),
        ["{\"a\":null,\"b\":2}"]
    );
    // Nesting.
    assert_eq!(
        rows(&s, "SELECT named_struct('a', named_struct('x', 1))"),
        ["{\"a\":{\"x\":1}}"]
    );
    // No arguments: the empty struct.
    assert_eq!(rows(&s, "SELECT struct()"), ["{}"]);
}

#[test]
fn struct_field_access_reads_by_name() {
    let s = session();
    assert_eq!(rows(&s, "SELECT named_struct('a', 1, 'b', 'x').b"), ["x"]);
    assert_eq!(rows(&s, "SELECT STRUCT(1 AS a, 2 AS b).b"), ["2"]);
    // A null struct gives a null field; a null field stays null.
    assert_eq!(rows(&s, "SELECT CAST(NULL AS STRUCT<a: INT>).a"), ["NULL"]);
    assert_eq!(
        rows(&s, "SELECT named_struct('a', CAST(NULL AS INT)).a"),
        ["NULL"]
    );
}

#[test]
fn struct_field_names_follow_case_sensitivity() {
    let s = session();
    assert_eq!(rows(&s, "SELECT named_struct('A', 1).a"), ["1"]);
    s.conf().set("spark.sql.caseSensitive", "true").unwrap();
    assert_eq!(
        error(&s, "SELECT named_struct('A', 1).a"),
        ErrorClass::FieldNotFound
    );
    assert_eq!(rows(&s, "SELECT named_struct('A', 1).A"), ["1"]);
}

#[test]
fn struct_errors_name_their_classes() {
    let s = session();
    // Odd positions take only foldable strings.
    assert_eq!(
        error(&s, "SELECT named_struct(1, 2)"),
        ErrorClass::DatatypeMismatchCreateNamedStructWithoutFoldableString
    );
    assert_eq!(
        error(&s, "SELECT named_struct('a', 1, true, 2)"),
        ErrorClass::DatatypeMismatchCreateNamedStructWithoutFoldableString
    );
    // A null name is unexpected.
    assert_eq!(
        error(&s, "SELECT named_struct(CAST(NULL AS STRING), 1)"),
        ErrorClass::DatatypeMismatchUnexpectedNull
    );
    // An untyped null is not a string at all.
    assert_eq!(
        error(&s, "SELECT named_struct(NULL, 1)"),
        ErrorClass::DatatypeMismatchCreateNamedStructWithoutFoldableString
    );
    // Odd counts are an arity error.
    assert_eq!(
        error(&s, "SELECT named_struct('a')"),
        ErrorClass::WrongNumArgsWithoutSuggestion
    );
    // Field access on a non-struct.
    assert_eq!(
        error(&s, "SELECT (1).a"),
        ErrorClass::InvalidExtractBaseFieldType
    );
    assert_eq!(
        error(&s, "SELECT named_struct('a', 1).b"),
        ErrorClass::FieldNotFound
    );
}

#[test]
fn struct_comparison_follows_field_order() {
    let s = session();
    assert_eq!(
        rows(&s, "SELECT named_struct('a', 1) = named_struct('a', 1)"),
        ["true"]
    );
    assert_eq!(
        rows(&s, "SELECT named_struct('a', 1) = named_struct('a', 2)"),
        ["false"]
    );
    assert_eq!(
        rows(&s, "SELECT named_struct('a', 1) <> named_struct('a', 2)"),
        ["true"]
    );
    // A null field is a value like any other for equality (it equals only
    // another null), never a NULL result (Spark 4.2).
    assert_eq!(
        rows(
            &s,
            "SELECT named_struct('a', CAST(NULL AS INT)) = named_struct('a', 1)"
        ),
        ["false"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT named_struct('a', CAST(NULL AS INT), 'b', 1) = named_struct('a', 2, 'b', 1)"
        ),
        ["false"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT named_struct('a', CAST(NULL AS INT), 'b', 1) = named_struct('a', 2, 'b', 2)"
        ),
        ["false"]
    );
    // Ordering is lexicographic.
    assert_eq!(
        rows(&s, "SELECT named_struct('a', 1) < named_struct('a', 2)"),
        ["true"]
    );
    assert_eq!(
        rows(&s, "SELECT named_struct('a', 2) < named_struct('a', 1)"),
        ["false"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT named_struct('a', 1, 'b', 2) < named_struct('a', 1, 'b', 3)"
        ),
        ["true"]
    );
    // A null field orders before every value and equals another null.
    for (sql, want) in [
        (
            "SELECT named_struct('a', 1, 'b', CAST(NULL AS INT)) < named_struct('a', 1, 'b', 2)",
            "true",
        ),
        (
            "SELECT named_struct('a', CAST(NULL AS INT), 'b', 1) > named_struct('a', 1, 'b', 1)",
            "false",
        ),
        (
            "SELECT named_struct('a', 1, 'b', 2) > named_struct('a', 1, 'b', CAST(NULL AS INT))",
            "true",
        ),
        (
            "SELECT named_struct('a', 1, 'b', CAST(NULL AS INT)) = named_struct('a', 1, 'b', CAST(NULL AS INT))",
            "true",
        ),
        // Equal null fields defer to the next field (Spark 4.2.0 run).
        (
            "SELECT named_struct('a', CAST(NULL AS INT), 'b', 1) < named_struct('a', CAST(NULL AS INT), 'b', 2)",
            "true",
        ),
        // Untyped NULL fields (a `NullArray` underneath) behave the same.
        (
            "SELECT named_struct('a', NULL) = named_struct('a', NULL)",
            "true",
        ),
        (
            "SELECT named_struct('a', NULL) < named_struct('a', NULL)",
            "false",
        ),
        (
            "SELECT named_struct('a', NULL) > named_struct('a', NULL)",
            "false",
        ),
        (
            "SELECT named_struct('a', NULL, 'b', 1) < named_struct('a', NULL, 'b', 2)",
            "true",
        ),
    ] {
        assert_eq!(rows(&s, sql), [want], "{sql}");
    }
}

/// `struct(...)` over a qualified star and over an aggregate (Spark 4.2.0:
/// the star's columns keep their names; a non-named child is `colN`).
#[test]
fn struct_of_a_star_and_of_an_aggregate() {
    let s = session();
    assert_eq!(
        rows(&s, "SELECT STRUCT(t.*) FROM (SELECT 1 AS a, 'x' AS b) AS t"),
        ["{\"a\":1,\"b\":\"x\"}"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT struct(k, sum(v)) FROM VALUES (1, 2), (1, 3) AS t(k, v) GROUP BY k"
        ),
        ["{\"k\":1,\"col2\":5}"]
    );
}

/// `Column.getField`, `Column.withField` and `Column.dropFields` end to
/// end: the PySpark 4.2 doc examples (checked against the documented
/// outputs, not a live Spark run), case sensitivity, duplicate field names,
/// nulls, nesting and every operator that takes a column (select, filter,
/// group-by keys, withColumn). The analyzer rewrites updates to the
/// `named_struct` the SQL path builds, so errors name the same classes the
/// SQL path raises.
fn collected(df: &DataFrame) -> Vec<String> {
    let mut out = Vec::new();
    for batch in block_on(df.collect()).unwrap() {
        let column = batch.column(0);
        for row in 0..batch.num_rows() {
            out.push(cell(column, row));
        }
    }
    out
}

fn column_error(df: &DataFrame, c: Column) -> ErrorClass {
    match block_on(df.select([c]).collect()) {
        Ok(_) => panic!("resolved"),
        Err(e) => e.error_class(),
    }
}

/// `getField`: the PySpark doc example, nulls, and the SQL path's errors.
#[test]
fn column_get_field_reads_by_name() {
    let s = session();
    let df = block_on(s.sql("SELECT named_struct('a', 1, 'b', 'x') AS r, 1 AS x")).unwrap();
    assert_eq!(collected(&df.select([col("r").get_field("b")])), ["x"]);
    assert_eq!(collected(&df.select([col("r").get_field("a")])), ["1"]);
    // A null struct gives a null field; a null field stays null.
    let nulls = block_on(s.sql(
        "SELECT CAST(NULL AS STRUCT<a: INT>) AS r, named_struct('a', CAST(NULL AS INT)) AS z",
    ))
    .unwrap();
    assert_eq!(
        collected(&nulls.select([col("r").get_field("a")])),
        ["NULL"]
    );
    assert_eq!(
        collected(&nulls.select([col("z").get_field("a")])),
        ["NULL"]
    );
    // Missing fields and non-structs fail as the SQL path fails.
    assert_eq!(
        column_error(&df, col("r").get_field("zzz")),
        ErrorClass::FieldNotFound
    );
    assert_eq!(
        column_error(&df, col("x").get_field("a")),
        ErrorClass::InvalidExtractBaseFieldType
    );
}

/// `getField` follows `spark.sql.caseSensitive`, as `s.f` does.
#[test]
fn column_get_field_follows_case_sensitivity() {
    let s = session();
    let df = block_on(s.sql("SELECT named_struct('A', 1) AS r")).unwrap();
    assert_eq!(collected(&df.select([col("r").get_field("a")])), ["1"]);
    s.conf().set("spark.sql.caseSensitive", "true").unwrap();
    assert_eq!(
        column_error(&df, col("r").get_field("a")),
        ErrorClass::FieldNotFound
    );
    assert_eq!(collected(&df.select([col("r").get_field("A")])), ["1"]);
}

/// `withField`: the PySpark doc examples (replace, then add).
#[test]
fn column_with_field_replaces_and_appends() {
    let s = session();
    let df = block_on(s.sql("SELECT named_struct('b', 1, 'c', 2) AS a")).unwrap();
    assert_eq!(
        collected(&df.select([col("a").with_field("b", lit(3))])),
        ["{\"b\":3,\"c\":2}"]
    );
    assert_eq!(
        collected(&df.select([col("a").with_field("d", lit(4))])),
        ["{\"b\":1,\"c\":2,\"d\":4}"]
    );
    // A replacement may change the field's type; values are never cast.
    assert_eq!(
        collected(&df.select([col("a").with_field("b", lit("x"))])),
        ["{\"b\":\"x\",\"c\":2}"]
    );
    let replaced = df.select([col("a").with_field("b", lit(3)).get_field("b")]);
    assert_eq!(collected(&replaced), ["3"]);
    let added = df.select([col("a").with_field("d", lit(4)).get_field("d")]);
    assert_eq!(collected(&added), ["4"]);
}

/// `dropFields`: the PySpark doc examples (one field, then several).
#[test]
fn column_drop_fields_drops_one_and_several() {
    let s = session();
    let df = block_on(
        s.sql("SELECT named_struct('b', 1, 'c', 2, 'd', 3, 'e', named_struct('f', 4, 'g', 5, 'h', 6)) AS a"),
    )
    .unwrap();
    assert_eq!(
        collected(&df.select([col("a").drop_fields(vec!["b".to_owned()])])),
        ["{\"c\":2,\"d\":3,\"e\":{\"f\":4,\"g\":5,\"h\":6}}"]
    );
    assert_eq!(
        collected(&df.select([col("a").drop_fields(vec!["b".to_owned(), "c".to_owned()])])),
        ["{\"d\":3,\"e\":{\"f\":4,\"g\":5,\"h\":6}}"]
    );
    // Dropping a missing field is a no-op.
    assert_eq!(
        collected(&df.select([col("a").drop_fields(vec!["zzz".to_owned()])])),
        ["{\"b\":1,\"c\":2,\"d\":3,\"e\":{\"f\":4,\"g\":5,\"h\":6}}"]
    );
}

/// Dropping every field and updating a non-struct are errors.
#[test]
fn column_struct_updates_name_their_errors() {
    let s = session();
    let df = block_on(s.sql("SELECT named_struct('b', 1, 'c', 2) AS a, 1 AS x")).unwrap();
    let error = |c: Column| match block_on(df.select([c]).collect()) {
        Ok(_) => panic!("resolved"),
        Err(e) => e.error_class(),
    };
    assert_eq!(
        error(col("a").drop_fields(vec!["b".to_owned(), "c".to_owned()])),
        ErrorClass::DatatypeMismatchCannotDropAllFields
    );
    assert_eq!(
        error(col("x").with_field("b", lit(1))),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert_eq!(
        error(col("x").drop_fields(vec!["b".to_owned()])),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
}

/// `withField`/`dropFields` replace and drop every match under a
/// case-insensitive session, and only exact matches under a sensitive one.
#[test]
fn column_struct_updates_match_like_field_access() {
    let s = session();
    let df = block_on(s.sql("SELECT named_struct('A', 1, 'a', 2, 'b', 3) AS s")).unwrap();
    // Case-insensitive (the default): every match is replaced or dropped.
    assert_eq!(
        collected(&df.select([col("s").with_field("a", lit(9))])),
        ["{\"a\":9,\"a\":9,\"b\":3}"]
    );
    assert_eq!(
        collected(&df.select([col("s").drop_fields(vec!["a".to_owned()])])),
        ["{\"b\":3}"]
    );
    s.conf().set("spark.sql.caseSensitive", "true").unwrap();
    assert_eq!(
        collected(&df.select([col("s").with_field("a", lit(9))])),
        ["{\"A\":1,\"a\":9,\"b\":3}"]
    );
    assert_eq!(
        collected(&df.select([col("s").drop_fields(vec!["a".to_owned()])])),
        ["{\"A\":1,\"b\":3}"]
    );
}

/// A null struct stays null, with the new struct's type.
#[test]
fn column_struct_updates_keep_a_null_struct_null() {
    let s = session();
    let df = block_on(s.sql("SELECT CAST(NULL AS STRUCT<b: INT, c: INT>) AS s")).unwrap();
    assert_eq!(
        collected(&df.select([col("s").with_field("b", lit(1))])),
        ["NULL"]
    );
    assert_eq!(
        collected(&df.select([col("s").drop_fields(vec!["b".to_owned()])])),
        ["NULL"]
    );
    for (c, names) in [
        (col("s").with_field("d", lit("x")), vec!["b", "c", "d"]),
        (col("s").drop_fields(vec!["b".to_owned()]), vec!["c"]),
    ] {
        let batches = block_on(df.select([c]).collect()).unwrap();
        let fields = match batches[0].schema().field(0).data_type() {
            arrow_schema::DataType::Struct(fields) => fields.clone(),
            other => panic!("not a struct: {other}"),
        };
        assert_eq!(
            fields.iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
            names
        );
    }
}

/// Nested updates: the inner update is the outer update's value.
#[test]
fn column_struct_updates_nest() {
    let s = session();
    let df = block_on(s.sql("SELECT named_struct('b', 1, 'e', named_struct('f', 4, 'g', 5)) AS a"))
        .unwrap();
    let inner = col("a").get_field("e").with_field("g", lit(9));
    assert_eq!(
        collected(&df.select([col("a").with_field("e", inner)])),
        ["{\"b\":1,\"e\":{\"f\":4,\"g\":9}}"]
    );
}

/// Updates work in `select`, `filter`, group-by keys and `withColumn`.
#[test]
fn column_struct_updates_run_in_every_clause() {
    let s = session();
    let df = block_on(
        s.sql("SELECT named_struct('b', b, 'c', c) AS s FROM VALUES (1, 10), (2, 20) AS t(b, c)"),
    )
    .unwrap();
    // `select` and `withColumn`.
    assert_eq!(
        collected(&df.select([col("s").with_field("d", lit(0))])),
        ["{\"b\":1,\"c\":10,\"d\":0}", "{\"b\":2,\"c\":20,\"d\":0}"]
    );
    let widened = df.with_column("t", col("s").with_field("d", col("s").get_field("b")));
    assert_eq!(
        collected(&widened.select([col("t").get_field("d")])),
        ["1", "2"]
    );
    // `filter` on an updated field.
    let kept = df.filter(col("s").with_field("b", lit(7)).get_field("b").eq(lit(7)));
    assert_eq!(
        collected(&kept.select([col("s").get_field("c")])),
        ["10", "20"]
    );
    // Group-by keys on a field.
    let groups = df.group_by([col("s").get_field("b")]).agg([count_star()]);
    let mut rows = Vec::new();
    for batch in block_on(groups.collect()).unwrap() {
        for row in 0..batch.num_rows() {
            rows.push(cell(batch.column(0), row) + "|" + &cell(batch.column(1), row));
        }
    }
    rows.sort();
    assert_eq!(rows, ["1|1", "2|1"]);
}

/// The analyzed plan holds no `UpdateFields` node: the update is the
/// `named_struct` rewrite, with the expected field order and types.
#[test]
fn column_struct_updates_resolve_to_named_struct() {
    let s = session();
    let df = block_on(s.sql("SELECT named_struct('b', 1, 'c', 'x') AS s")).unwrap();
    let updated = df.select([col("s").with_field("a", lit(true)).alias("u")]);
    let text = block_on(updated.explain(ExplainMode::Extended)).unwrap_or_else(|e| panic!("{e}"));
    let (parsed, analyzed) = text
        .split_once("== Analyzed Logical Plan ==")
        .unwrap_or((text.as_str(), ""));
    // The parsed plan still holds the update; analysis rewrote it.
    assert!(parsed.contains("update_fields"), "{text}");
    assert!(!analyzed.contains("update_fields"), "{text}");
    assert!(!analyzed.contains("UpdateFields"), "{text}");
    assert!(analyzed.contains("named_struct"), "{text}");
    // The output names the new struct type in field order.
    let batches = block_on(updated.collect()).unwrap();
    let fields = match batches[0].schema().field(0).data_type() {
        arrow_schema::DataType::Struct(fields) => fields.clone(),
        other => panic!("not a struct: {other}"),
    };
    assert_eq!(
        fields
            .iter()
            .map(|f| (f.name().as_str(), f.data_type().clone()))
            .collect::<Vec<_>>(),
        [
            ("b", arrow_schema::DataType::Int32),
            ("c", arrow_schema::DataType::Utf8View),
            ("a", arrow_schema::DataType::Boolean),
        ]
    );
}

/// A nondeterministic base is refused: the rewrite would evaluate it once
/// per kept field, reading a different value each time.
#[test]
fn column_struct_update_of_a_nondeterministic_base_is_unsupported() {
    let s = session();
    let base = call_function("named_struct", [lit("b"), call_function("rand", [])]);
    let df = block_on(s.sql("SELECT 1 AS x")).unwrap();
    let Err(got) = block_on(df.select([base.with_field("c", lit(1))]).collect()) else {
        panic!("resolved")
    };
    assert_eq!(got.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // The same shape with a deterministic base resolves.
    let steady = call_function("named_struct", [lit("b"), lit(1)]);
    assert_eq!(
        collected(&df.select([steady.with_field("c", lit(1))])),
        ["{\"b\":1,\"c\":1}"]
    );
}
/// `MIN`/`MAX` over structs (Spark 4.2: field-by-field ordering, NULL
/// fields smallest and equal to one another, NaN the largest float; a NULL
/// struct row is skipped like any NULL input, all-NULL gives NULL).
#[test]
fn min_and_max_over_structs() {
    let s = session();
    // The golden `struct.sql` statement: schema and rows.
    assert_eq!(
        rows_all(&s, "SELECT MAX(STRUCT(1 AS a)), 2 AS b, b AS c GROUP BY b"),
        ["{\"a\":1}|2|2"]
    );
    // Simple, grouped and global.
    assert_eq!(
        rows(&s, "SELECT MAX(STRUCT(1 AS a, 2 AS b))"),
        ["{\"a\":1,\"b\":2}"]
    );
    assert_eq!(
        rows_all(
            &s,
            "SELECT k, MIN(s), MAX(s) FROM VALUES (1, named_struct('a', 1, 'b', 'y')), (1, named_struct('a', 1, 'b', 'x')), (1, named_struct('a', 2, 'b', 'a')) AS t(k, s) GROUP BY k"
        ),
        ["1|{\"a\":1,\"b\":\"x\"}|{\"a\":2,\"b\":\"a\"}"]
    );
    // A NULL field sorts before every value; NULL structs are skipped.
    assert_eq!(
        rows_all(
            &s,
            "SELECT MIN(s), MAX(s) FROM VALUES (named_struct('a', 1, 'b', CAST(NULL AS STRING))), (named_struct('a', 1, 'b', 'x')), (CAST(NULL AS STRUCT<a: INT, b: STRING>)) AS t(s)"
        ),
        ["{\"a\":1,\"b\":null}|{\"a\":1,\"b\":\"x\"}"]
    );
    // Field by field with NULLs first (Spark 4.2.0): a NULL first field is
    // the smallest, whatever the later fields hold.
    assert_eq!(
        rows_all(
            &s,
            "SELECT MAX(c), MIN(c) FROM VALUES (named_struct('a', 1, 'b', 'x')), (named_struct('a', 1, 'b', CAST(NULL AS STRING))), (named_struct('a', CAST(NULL AS INT), 'b', 'z')), (CAST(NULL AS STRUCT<a: INT, b: STRING>)) AS t(c)"
        ),
        ["{\"a\":1,\"b\":\"x\"}|{\"a\":null,\"b\":\"z\"}"]
    );
    assert_eq!(
        rows_all(
            &s,
            "SELECT MAX(c), MIN(c) FROM VALUES (named_struct('a', 2, 'b', 1)), (named_struct('a', 2, 'b', CAST(NULL AS INT))) AS t(c)"
        ),
        ["{\"a\":2,\"b\":1}|{\"a\":2,\"b\":null}"]
    );
    // Only NULL structs: NULL.
    assert_eq!(
        rows_all(
            &s,
            "SELECT MIN(s), MAX(s) FROM VALUES (CAST(NULL AS STRUCT<a: INT>)) AS t(s)"
        ),
        ["NULL|NULL"]
    );
    // Nested structs, strings, decimals and floats with NaN.
    assert_eq!(
        rows_all(
            &s,
            "SELECT MIN(s), MAX(s) FROM (SELECT named_struct('n', named_struct('x', x), 'd', d, 't', t) AS s FROM VALUES (CAST('NaN' AS DOUBLE), CAST(1.00 AS DECIMAL(10, 2)), 'a'), (1.0, CAST(2.00 AS DECIMAL(10, 2)), 'z'), (CAST(NULL AS DOUBLE), CAST(0.50 AS DECIMAL(10, 2)), 'm') AS v(x, d, t))"
        ),
        ["{\"n\":{\"x\":null},\"d\":0.50,\"t\":\"m\"}|{\"n\":{\"x\":NaN},\"d\":1.00,\"t\":\"a\"}"]
    );
    // `MAX` and `MIN` over a struct with a map field is an ordering error
    // (Spark's `DATATYPE_MISMATCH.INVALID_ORDERING_TYPE`).
    for sql in [
        "SELECT MAX(CAST(NULL AS STRUCT<m: MAP<STRING, INT>>)) FROM VALUES (1) AS t(x)",
        "SELECT MIN(CAST(NULL AS STRUCT<m: MAP<STRING, INT>>)) FROM VALUES (1) AS t(x)",
    ] {
        assert_eq!(
            error(&s, sql),
            ErrorClass::DatatypeMismatchInvalidOrderingType
        );
    }
}

/// Window `MIN`/`MAX` over structs: whole partitions, running frames and
/// `ROWS` frames all use the aggregate's struct comparison.
#[test]
fn min_and_max_over_structs_in_windows() {
    let s = session();
    let input = "VALUES (1, 1, named_struct('a', 1)), (1, 2, CAST(NULL AS STRUCT<a: INT>)), (1, 3, named_struct('a', 2)), (2, 1, CAST(NULL AS STRUCT<a: INT>)) AS t(k, o, s)";
    let mut whole = rows_all(
        &s,
        &format!("SELECT k, o, MAX(s) OVER (PARTITION BY k) FROM {input}"),
    );
    whole.sort();
    assert_eq!(
        whole,
        [
            "1|1|{\"a\":2}",
            "1|2|{\"a\":2}",
            "1|3|{\"a\":2}",
            "2|1|NULL",
        ]
    );
    // Running peers (the default frame): NULL structs are skipped.
    let mut running = rows_all(
        &s,
        &format!("SELECT k, o, MIN(s) OVER (PARTITION BY k ORDER BY o) FROM {input}"),
    );
    running.sort();
    assert_eq!(
        running,
        [
            "1|1|{\"a\":1}",
            "1|2|{\"a\":1}",
            "1|3|{\"a\":1}",
            "2|1|NULL",
        ]
    );
    // Running rows: each row's frame is itself and its predecessor (the
    // NULL row contributes nothing).
    let mut rows_frame = rows_all(
        &s,
        &format!(
            "SELECT k, MAX(s) OVER (PARTITION BY k ORDER BY o ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM {input}"
        ),
    );
    rows_frame.sort();
    assert_eq!(
        rows_frame,
        ["1|{\"a\":1}", "1|{\"a\":1}", "1|{\"a\":2}", "2|NULL"]
    );
    // A `ROWS` frame around the row.
    let mut frame = rows_all(
        &s,
        &format!(
            "SELECT k, o, MIN(s) OVER (PARTITION BY k ORDER BY o ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM {input}"
        ),
    );
    frame.sort();
    assert_eq!(
        frame,
        [
            "1|1|{\"a\":1}",
            "1|2|{\"a\":1}",
            "1|3|{\"a\":2}",
            "2|1|NULL",
        ]
    );
}

/// `withField` and `dropFields` read a dotted name as a path into nested
/// structs, a NULL parent stays NULL, a missing parent is `FIELD_NOT_FOUND`,
/// and dropping every field of a nested struct is `CANNOT_DROP_ALL_FIELDS`
/// (all against Spark 4.2.0 through Connect).
#[test]
fn column_struct_updates_follow_dotted_paths() {
    let s = session();
    let df = block_on(s.sql(
        "SELECT named_struct('a', named_struct('b', 1, 'c', 2), 'd', 3) AS n, \
         named_struct('a', CAST(NULL AS STRUCT<b: INT, c: INT>), 'd', 3) AS m",
    ))
    .unwrap();
    let one = |c: Column| collected(&df.select([c]));
    assert_eq!(
        one(col("n").with_field("a.b", lit(9))),
        ["{\"a\":{\"b\":9,\"c\":2},\"d\":3}"]
    );
    assert_eq!(
        one(col("n").with_field("a.z", lit(1))),
        ["{\"a\":{\"b\":1,\"c\":2,\"z\":1},\"d\":3}"]
    );
    assert_eq!(
        one(col("n").drop_fields(vec!["a.b".to_owned()])),
        ["{\"a\":{\"c\":2},\"d\":3}"]
    );
    assert_eq!(
        one(col("m").with_field("a.b", lit(1))),
        ["{\"a\":null,\"d\":3}"]
    );
    assert_eq!(
        column_error(&df, col("n").with_field("x.y", lit(1))),
        ErrorClass::FieldNotFound
    );
    assert_eq!(
        column_error(
            &df,
            col("n").drop_fields(vec!["a.b".to_owned(), "a.c".to_owned()])
        ),
        ErrorClass::DatatypeMismatchCannotDropAllFields
    );
}
