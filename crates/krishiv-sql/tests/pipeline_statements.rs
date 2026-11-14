// SPDX-License-Identifier: Apache-2.0
//! `CREATE MATERIALIZED VIEW`, `CREATE STREAMING TABLE`, `CREATE FLOW`, `LOAD DATA` and
//! `INSERT OVERWRITE [LOCAL] DIRECTORY`, pinned to a real Spark 4.2.0 run (`parsePlan` of each statement).
#![expect(clippy::panic, clippy::expect_used, reason = "test code")]

use krishiv_common::error::ErrorClass;
use krishiv_sql::ast::{DirectoryFormat, PipelineDatasetKind, Statement};
use krishiv_sql::{ParserOptions, parse_statement};

fn parse(sql: &str) -> Statement {
    parse_statement(sql, &ParserOptions::default()).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn fails(sql: &str) -> (ErrorClass, String, usize) {
    let e = parse_statement(sql, &ParserOptions::default()).expect_err(sql);
    (e.error.error_class(), e.error.to_string(), e.span.start)
}

fn fail_end(sql: &str) -> usize {
    parse_statement(sql, &ParserOptions::default())
        .expect_err(sql)
        .span
        .end
}

#[test]
fn materialized_view_and_streaming_table() {
    let Statement::CreatePipelineDataset(c) = parse("CREATE MATERIALIZED VIEW mv AS SELECT 1")
    else {
        panic!("not a pipeline dataset")
    };
    assert_eq!(c.kind, PipelineDatasetKind::MaterializedView);
    assert!(c.table.as_query.is_some());
    // CreateStreamingTableAsSelect, with the CREATE TABLE clauses.
    let Statement::CreatePipelineDataset(c) = parse(
        "CREATE MATERIALIZED VIEW IF NOT EXISTS mv (a INT COMMENT 'c') COMMENT 'x' \
         PARTITIONED BY (a) TBLPROPERTIES ('k'='v') AS SELECT 1 AS a",
    ) else {
        panic!("not a pipeline dataset")
    };
    assert!(c.table.if_not_exists);
    assert_eq!(c.table.elements.len(), 1);
    assert_eq!(c.table.comment.as_deref(), Some("x"));
    assert_eq!(c.table.partitioned_by.len(), 1);
    assert_eq!(c.table.tblproperties.len(), 1);
    let Statement::CreatePipelineDataset(c) =
        parse("CREATE MATERIALIZED VIEW mv USING parquet AS SELECT 1")
    else {
        panic!("not a pipeline dataset")
    };
    assert!(c.table.using.is_some());
    let Statement::CreatePipelineDataset(c) = parse("CREATE STREAMING TABLE st AS SELECT 1") else {
        panic!("not a pipeline dataset")
    };
    assert_eq!(c.kind, PipelineDatasetKind::StreamingTable);
    // CreateStreamingTable: no query.
    let Statement::CreatePipelineDataset(c) = parse("CREATE STREAMING TABLE st") else {
        panic!("not a pipeline dataset")
    };
    assert!(c.table.as_query.is_none());
}

#[test]
fn pipeline_dataset_errors() {
    // ParseException "Operation not allowed: Unable to find query for CREATE MATERIALIZED VIEW statement."
    let (class, message, start) = fails("CREATE MATERIALIZED VIEW mv");
    assert_eq!(class, ErrorClass::LegacyErrorTemp0035);
    assert!(
        message.contains(
            "Operation not allowed: Unable to find query for CREATE MATERIALIZED VIEW statement.."
        ),
        "{message}"
    );
    assert_eq!(start, 0);
    // PARSE_SYNTAX_ERROR at 'REFRESH' (pos 10) and 'MATERIALIZED' (pos 17).
    for (sql, pos) in [
        ("CREATE OR REFRESH MATERIALIZED VIEW mv AS SELECT 1", 10),
        ("CREATE OR REFRESH STREAMING TABLE st AS SELECT 1", 10),
        ("CREATE TEMPORARY MATERIALIZED VIEW mv AS SELECT 1", 17),
    ] {
        let (class, _, start) = fails(sql);
        assert_eq!(class, ErrorClass::ParseSyntaxError, "{sql}");
        assert_eq!(start, pos, "{sql}");
    }
}

#[test]
fn flow_and_load_data() {
    assert!(matches!(
        parse("CREATE FLOW f AS INSERT INTO t BY NAME SELECT 1"),
        Statement::CreateFlow(_)
    ));
    let Statement::LoadData(l) = parse("LOAD DATA INPATH 'x' INTO TABLE nat") else {
        panic!("not LOAD DATA")
    };
    assert_eq!((l.local, l.overwrite, l.path.as_str()), (false, false, "x"));
    let Statement::LoadData(l) = parse("LOAD DATA LOCAL INPATH 'x' OVERWRITE INTO TABLE nat")
    else {
        panic!("not LOAD DATA")
    };
    assert_eq!((l.local, l.overwrite), (true, true));
    assert!(matches!(
        parse("LOAD DATA INPATH 'x' INTO TABLE nat PARTITION (p = 1)"),
        Statement::LoadData(_)
    ));
}

#[test]
fn insert_overwrite_directory() {
    for sql in [
        "INSERT OVERWRITE DIRECTORY '/tmp/o1' USING parquet SELECT a, count(*) c FROM nat GROUP BY 1",
        "INSERT OVERWRITE DIRECTORY USING parquet OPTIONS ('path' '/tmp/o2') SELECT a FROM nat",
        "INSERT OVERWRITE LOCAL DIRECTORY '/tmp/o3' USING parquet SELECT a FROM nat",
        "INSERT OVERWRITE DIRECTORY '/tmp/o4' USING csv OPTIONS ('header' 'true') SELECT a FROM nat",
        "INSERT OVERWRITE DIRECTORY '/tmp/o5' STORED AS parquet SELECT a FROM nat",
        "INSERT OVERWRITE DIRECTORY '/tmp/o6' ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' SELECT a FROM nat",
        "INSERT OVERWRITE DIRECTORY '/tmp/o10' USING parquet VALUES (1)",
        "INSERT OVERWRITE DIRECTORY '/tmp/o11' USING parquet TABLE nat",
        "INSERT OVERWRITE DIRECTORY '/tmp/o12' USING parquet FROM nat SELECT a",
    ] {
        assert!(matches!(parse(sql), Statement::InsertDirectory(_)), "{sql}");
    }
    let Statement::InsertDirectory(d) =
        parse("INSERT OVERWRITE LOCAL DIRECTORY '/tmp/o3' USING parquet SELECT a FROM nat")
    else {
        panic!("not a directory insert")
    };
    assert!(d.local);
    assert_eq!(d.path.as_deref(), Some("/tmp/o3"));
    assert!(matches!(d.format, DirectoryFormat::Provider { .. }));
    let Statement::InsertDirectory(d) =
        parse("INSERT OVERWRITE DIRECTORY USING parquet OPTIONS ('path' '/tmp/o2') SELECT 1")
    else {
        panic!("not a directory insert")
    };
    assert!(d.path.is_none() && d.has_path_option());
}

#[test]
fn insert_directory_errors() {
    // The path given both ways or neither: _LEGACY_ERROR_TEMP_0049 at the statement.
    for sql in [
        "INSERT OVERWRITE DIRECTORY USING parquet SELECT a FROM nat",
        "INSERT OVERWRITE DIRECTORY '/tmp/o7' USING parquet OPTIONS ('path' '/tmp/o8') SELECT a FROM nat",
    ] {
        let (class, message, start) = fails(sql);
        assert_eq!(class, ErrorClass::LegacyErrorTemp0049, "{sql}");
        assert!(
            message.contains(
                "Directory path and 'path' in OPTIONS should be specified one, but not both."
            ),
            "{message}"
        );
        assert_eq!(start, 0);
    }
    // INSERT INTO DIRECTORY: syntax error at the path (pos 22).
    let (class, _, start) =
        fails("INSERT INTO DIRECTORY '/tmp/o9' USING parquet SELECT a FROM nat");
    assert_eq!(class, ErrorClass::ParseSyntaxError);
    assert_eq!(start, 22);
}

/// A table may be named `directory` (non-reserved) or `local`: those stay table inserts.
#[test]
fn a_table_named_directory_is_still_a_table() {
    for sql in [
        "INSERT OVERWRITE directory SELECT 1",
        "INSERT OVERWRITE directory VALUES (1)",
        "INSERT OVERWRITE directory (a) SELECT 1",
        "INSERT OVERWRITE TABLE directory SELECT 1",
        "INSERT OVERWRITE local SELECT 1",
        "INSERT INTO directory SELECT 1",
    ] {
        assert!(matches!(parse(sql), Statement::Insert(_)), "{sql}");
    }
}

/// Spark's context for the one-path rule is the insert header (the probe's carets: 40 and 77 characters).
#[test]
fn the_path_rule_covers_the_header_only() {
    assert_eq!(
        fail_end("INSERT OVERWRITE DIRECTORY USING parquet SELECT a FROM nat"),
        40
    );
    assert_eq!(
        fail_end(
            "INSERT OVERWRITE DIRECTORY '/tmp/o7' USING parquet OPTIONS ('path' '/tmp/o8') SELECT a FROM nat"
        ),
        77
    );
    // An empty literal counts as no path, as in Spark.
    assert!(matches!(
        parse("INSERT OVERWRITE DIRECTORY '' USING parquet OPTIONS ('path' '/x') SELECT 1"),
        Statement::InsertDirectory(_)
    ));
    assert_eq!(
        fails("INSERT OVERWRITE DIRECTORY '' USING parquet SELECT 1").0,
        ErrorClass::LegacyErrorTemp0049
    );
}

/// The directory is a write target whether written outside `OPTIONS` or as its `path` (any case).
#[test]
fn the_directory_is_a_write_target() {
    use krishiv_sql::{TargetObject, statement_targets};
    for sql in [
        "INSERT OVERWRITE DIRECTORY '/x' USING parquet SELECT 1",
        "INSERT OVERWRITE DIRECTORY USING parquet OPTIONS ('PATH' '/x') SELECT 1",
    ] {
        let statement = parse(sql);
        let targets = statement_targets(&statement);
        assert!(
            targets.iter().any(|t| t.object == TargetObject::Path("/x")),
            "{sql}"
        );
    }
}

/// `identity(a)` partitioning and the comment of the Spark tree.
#[test]
fn partition_columns_and_comment_are_kept() {
    let Statement::CreatePipelineDataset(c) = parse(
        "CREATE MATERIALIZED VIEW mv (a INT) COMMENT 'x' PARTITIONED BY (a) AS SELECT 1 AS a",
    ) else {
        panic!("not a pipeline dataset")
    };
    assert_eq!(c.table.comment.as_deref(), Some("x"));
    assert_eq!(c.table.partitioned_by.len(), 1);
}
