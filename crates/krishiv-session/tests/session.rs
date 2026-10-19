// SPDX-License-Identifier: Apache-2.0
//! Sessions end to end: SQL and DataFrames through analysis, the optimizer,
//! the physical planner and the executor; configuration snapshots and
//! analysis reuse; temporary tables and views; reads by path.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::StreamExt;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::analyzer::BuiltinFunctions;
use krishiv_session::functions::{
    avg, coalesce, col, count, count_distinct, count_star, lit, max, min, sum,
};
use krishiv_session::{ExplainMode, JoinType, Session};

fn people() -> (SchemaRef, RecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int32, false),
        Field::new("name", ArrowType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![3, 1, 2])),
            Arc::new(StringArray::from(vec![Some("c"), None, Some("b")])),
        ],
    )
    .unwrap();
    (schema, batch)
}

fn session() -> Session {
    let s = Session::new().unwrap();
    let (schema, batch) = people();
    s.register_batches("people", schema, vec![batch]).unwrap();
    s
}

fn ids(batches: &[RecordBatch]) -> Vec<i32> {
    batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect()
}

fn class<T: std::fmt::Debug>(r: Result<T, KrishivError>) -> ErrorClass {
    r.unwrap_err().error_class()
}

#[test]
fn sql_and_dataframes_run_end_to_end() {
    let s = session();
    let df = block_on(s.sql("SELECT id FROM people ORDER BY id DESC LIMIT 2")).unwrap();
    assert_eq!(ids(&block_on(df.collect()).unwrap()), [3, 2]);
    // The same through the DataFrame API, and the same plan shape.
    let df = s
        .table("people")
        .select([col("id")])
        .order_by([col("id").desc()])
        .limit(2);
    assert_eq!(ids(&block_on(df.collect()).unwrap()), [3, 2]);
    assert_eq!(block_on(df.count()).unwrap(), 2);
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(schema.field(0).name(), "id");
    assert!(!schema.field(0).is_nullable());
    // Names resolve case-insensitively, as Spark's default.
    let df = s.table("PEOPLE").select([col("ID").alias("x")]);
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(schema.field(0).name(), "x");
    // A stream dropped early cancels the query without hanging.
    let mut stream = block_on(s.table("people").execute_stream()).unwrap();
    let first = block_on(stream.next()).unwrap().unwrap();
    assert_eq!(first.num_columns(), 2);
    drop(stream);
}

#[test]
fn actions_run_from_a_tokio_runtime() {
    let s = session();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let rows = rt
        .block_on(async { s.table("people").count().await })
        .unwrap();
    assert_eq!(rows, 3);
}

#[test]
fn errors_surface_where_spark_raises_them() {
    let s = session();
    // `sql` parses and analyzes at once.
    assert_eq!(
        class(block_on(s.sql("SELECT FROM WHERE"))),
        ErrorClass::ParseSyntaxError
    );
    assert_eq!(
        class(block_on(s.sql("SELECT nope FROM people"))),
        ErrorClass::UnresolvedColumnWithSuggestion
    );
    let e = block_on(s.sql("EXPLAIN SELECT 1")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("EXPLAIN statements"), "{e}");
    // DataFrames fail at their first action.
    let df = s.table("people").select([col("nope")]);
    assert_eq!(
        class(block_on(df.schema())),
        ErrorClass::UnresolvedColumnWithSuggestion
    );
    // The built-in registry resolves functions by default.
    let df = s.table("people").agg([sum(col("id")).alias("total")]);
    assert_eq!(block_on(df.schema()).unwrap().field(0).name(), "total");
    let df = s.table("people").select([(col("id") + 1).alias("next")]);
    assert_eq!(block_on(df.schema()).unwrap().field(0).name(), "next");
    // Filters run now that the physical planner has them.
    let df = s.table("people").filter(col("id").gt(lit(1)));
    assert_eq!(ids(&block_on(df.collect()).unwrap()), [3, 2]);
}

#[test]
fn dataframe_columns_resolve_in_their_own_frame() {
    let s = Session::builder()
        .functions(Arc::new(BuiltinFunctions::new()))
        .build()
        .unwrap();
    let (schema, batch) = people();
    s.register_batches("people", schema, vec![batch]).unwrap();
    let a = s.table("people");
    let b = s.table("people");
    // A self-join: `a.col("id")` and `b.col("id")` are different columns.
    let joined = a
        .join(&b, a.col("id").eq(b.col("id")), JoinType::Inner)
        .select([a.col("id"), b.col("name")]);
    let schema = block_on(joined.schema()).unwrap();
    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        ["id", "name"]
    );
    let using = a.join_using(&b, ["id"], JoinType::LeftOuter);
    assert_eq!(block_on(using.schema()).unwrap().fields().len(), 3);
    let cross = a.cross_join(&b);
    assert_eq!(block_on(cross.schema()).unwrap().fields().len(), 4);
    // Aggregates with a registry: keys first, then the aggregates.
    let grouped = s
        .table("people")
        .group_by(["name"])
        .agg([count_star().alias("n"), sum(col("id")).alias("total")]);
    let schema = block_on(grouped.schema()).unwrap();
    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        ["name", "n", "total"]
    );
    let aliased = s
        .table("people")
        .alias("p")
        .where_(col("p.id").is_not_null());
    assert_eq!(block_on(aliased.schema()).unwrap().fields().len(), 2);
    let with = s.table("people").with_column("id", col("id") * 10);
    assert_eq!(block_on(with.schema()).unwrap().fields().len(), 2);
}

#[test]
fn configuration_is_snapshotted_per_query() {
    let s = session();
    let conf = s.conf();
    assert_eq!(conf.get("spark.sql.caseSensitive"), None);
    let df = s.table("PEOPLE");
    assert_eq!(block_on(df.schema()).unwrap().fields().len(), 2);
    // A change re-analyzes: the name no longer matches.
    conf.set("spark.sql.caseSensitive", "true").unwrap();
    assert_eq!(conf.get("spark.sql.caseSensitive").as_deref(), Some("true"));
    assert_eq!(
        class(block_on(df.schema())),
        ErrorClass::TableOrViewNotFound
    );
    conf.unset("spark.sql.caseSensitive").unwrap();
    assert_eq!(block_on(df.schema()).unwrap().fields().len(), 2);
    // Entries fixed when the session was built.
    for key in ["krishiv.memory.limit", "krishiv.io.maxConcurrentRequests"] {
        assert!(!conf.is_modifiable(key));
        assert_eq!(
            class(conf.set(key, "1")),
            ErrorClass::CannotModifyStaticConfig
        );
        assert_eq!(class(conf.unset(key)), ErrorClass::CannotModifyStaticConfig);
    }
    assert_eq!(
        class(conf.set("krishiv.nope", "1")),
        ErrorClass::KrishivUnknownConfig
    );
    // Builder values, static ones included.
    let built = Session::builder()
        .config("krishiv.memory.limit", "256m")
        .build()
        .unwrap();
    assert_eq!(
        built.conf().get("krishiv.memory.limit").as_deref(),
        Some("256MB")
    );
    assert!(
        Session::builder()
            .config("krishiv.exec.batchSize", "x")
            .build()
            .is_err()
    );
}

#[test]
fn temporary_tables_are_listed_replaced_and_dropped() {
    let s = session();
    assert_eq!(s.temp_table_names(), ["people"]);
    let df = s.table("people");
    assert_eq!(block_on(df.count()).unwrap(), 3);
    // Replacing the table re-analyzes the DataFrame.
    let (schema, _) = people();
    s.register_batches("people", schema, vec![]).unwrap();
    assert_eq!(block_on(df.count()).unwrap(), 0);
    assert!(s.drop_temp_view("People"));
    assert!(!s.drop_temp_view("people"));
    assert_eq!(class(block_on(df.count())), ErrorClass::TableOrViewNotFound);
    // A batch of another schema is refused.
    let other = Arc::new(Schema::new(vec![Field::new("x", ArrowType::Int32, false)]));
    let (_, batch) = people();
    assert_eq!(
        class(s.register_batches("t", other, vec![batch])),
        ErrorClass::InternalError
    );
    // A temporary view of anything but a read by path is not supported yet.
    let e = block_on(s.table("missing").create_or_replace_temp_view("v"));
    assert_eq!(class(e), ErrorClass::TableOrViewNotFound);
    let e = block_on(session().table("people").create_or_replace_temp_view("v"));
    assert_eq!(class(e), ErrorClass::KrishivUnsupportedFeature);
}

/// Writes a Parquet file with `id INT NOT NULL`.
fn parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2]))],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut w = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

#[test]
fn reads_by_path_and_their_views() {
    let dir = tempfile::tempdir().unwrap();
    parquet_file(&dir.path().join("part-0.parquet"));
    let root = dir.path().to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    // File columns are nullable, as Spark reads them.
    let df = s.read().parquet([root.as_str()]);
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(schema.field(0).name(), "id");
    assert!(schema.field(0).is_nullable());
    let df = block_on(s.sql(&format!("SELECT id FROM parquet.`{root}`"))).unwrap();
    assert_eq!(block_on(df.schema()).unwrap().fields().len(), 1);
    let missing = s
        .read()
        .format("parquet")
        .option("mergeSchema", "false")
        .load([format!("{root}/nope")]);
    assert_eq!(class(block_on(missing.schema())), ErrorClass::PathNotFound);
    // A view over files lists them again at every query.
    block_on(
        s.read()
            .parquet([root.as_str()])
            .create_or_replace_temp_view("ids"),
    )
    .unwrap();
    assert_eq!(s.temp_table_names(), ["ids"]);
    let df = block_on(s.sql("SELECT * FROM ids")).unwrap();
    assert_eq!(block_on(df.schema()).unwrap().fields().len(), 1);
    // Running queries over files waits for the file table provider.
    assert_eq!(
        class(block_on(df.collect())),
        ErrorClass::KrishivUnsupportedFeature
    );
}

#[test]
fn file_views_keep_the_schema_they_were_created_with() {
    // Spark 4.2.0: `spark.read.parquet(p).createOrReplaceTempView("v")`,
    // then overwriting `p` with a wider schema: both the DataFrame built
    // before and a new `SELECT * FROM v` still have the columns `[id]`.
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("part-0.parquet");
    parquet_file(&file);
    let root = dir.path().to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    block_on(
        s.read()
            .parquet([root.as_str()])
            .create_or_replace_temp_view("v"),
    )
    .unwrap();
    let before = block_on(s.sql("SELECT * FROM v")).unwrap();
    let names = |schema: SchemaRef| -> Vec<String> {
        schema.fields().iter().map(|f| f.name().clone()).collect()
    };
    assert_eq!(names(block_on(before.schema()).unwrap()), ["id"]);
    let wider = Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int32, false),
        Field::new("name", ArrowType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&wider),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec!["x"])),
        ],
    )
    .unwrap();
    let mut w =
        parquet::arrow::ArrowWriter::try_new(std::fs::File::create(&file).unwrap(), wider, None)
            .unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    assert_eq!(names(block_on(before.schema()).unwrap()), ["id"]);
    let after = block_on(s.sql("SELECT * FROM v")).unwrap();
    assert_eq!(names(block_on(after.schema()).unwrap()), ["id"]);
    assert_eq!(names(block_on(s.table("v").schema()).unwrap()), ["id"]);
    // A read by path lists the files again: it sees the new column.
    let read = s.read().parquet([root.as_str()]);
    assert_eq!(names(block_on(read.schema()).unwrap()), ["id", "name"]);
}

#[test]
fn table_names_parse_as_multipart_identifiers() {
    // Spark 4.2.0's `spark.table(name)`: backquotes quote a dot, a doubled
    // backquote is one, and a name that does not parse fails at the first
    // action with the parser's error.
    let s = session();
    let (schema, batch) = people();
    s.register_batches("x.y", Arc::clone(&schema), vec![batch.clone()])
        .unwrap();
    s.register_batches("a`b", schema, vec![batch]).unwrap();
    assert_eq!(
        ids(&block_on(s.table("`x.y`").collect()).unwrap()),
        [3, 1, 2]
    );
    assert_eq!(
        ids(&block_on(s.table("`a``b`").collect()).unwrap()),
        [3, 1, 2]
    );
    // A qualified name does not reach a temporary view (Spark 4.2.0).
    assert_eq!(
        class(block_on(s.table(" default.`x.y` ").schema())),
        ErrorClass::TableOrViewNotFound
    );
    assert_eq!(
        class(block_on(s.table("x.y").schema())),
        ErrorClass::TableOrViewNotFound
    );
    for bad in ["`unclosed", "a..b", "a b", "`x.y`."] {
        let df = s.table(bad);
        assert_eq!(
            class(block_on(df.schema())),
            ErrorClass::ParseSyntaxError,
            "{bad}"
        );
        // Derived frames and joins keep the error.
        let derived = df.select([col("id")]).limit(1);
        assert_eq!(
            class(block_on(derived.collect())),
            ErrorClass::ParseSyntaxError
        );
        let joined = s.table("people").cross_join(&df);
        assert_eq!(
            class(block_on(joined.count())),
            ErrorClass::ParseSyntaxError
        );
    }
    assert_eq!(
        class(block_on(s.table("").schema())),
        ErrorClass::ParseEmptyStatement
    );
}

#[test]
fn explain_prints_every_plan() {
    let s = session();
    let df = s.table("people").select([col("id")]).limit(1);
    let simple = block_on(df.explain(ExplainMode::Simple)).unwrap();
    assert!(simple.starts_with("== Physical Plan ==\n"), "{simple}");
    let extended = block_on(df.explain(ExplainMode::Extended)).unwrap();
    let headers: Vec<&str> = extended.lines().filter(|l| l.starts_with("== ")).collect();
    assert_eq!(
        headers,
        [
            "== Parsed Logical Plan ==",
            "== Analyzed Logical Plan ==",
            "== Optimized Logical Plan ==",
            "== Physical Plan ==",
        ]
    );
    // Spark's schema line under the analyzed header.
    assert!(
        extended.contains("== Analyzed Logical Plan ==\nid: int\n"),
        "{extended}"
    );
}

#[test]
fn debug_output_names_the_parts() {
    let s = session();
    let shown = format!("{s:?} {:?} {:?}", Session::builder(), s.table("people"));
    assert!(
        shown.contains("Session")
            && shown.contains("SessionBuilder")
            && shown.contains("DataFrame")
    );
    let reader = format!("{:?}", s.read());
    assert!(reader.contains("parquet"), "{reader}");
    let grouped = format!("{:?}", s.table("people").group_by(["id"]));
    assert!(grouped.contains("GroupedData"), "{grouped}");
}

#[cfg(feature = "unstable-ir")]
#[test]
fn plans_go_in_and_out_with_unstable_ir() {
    use krishiv_plan::{LogicalPlan, Plan};
    let s = session();
    let df = s.from_plan(&Plan::try_new(LogicalPlan::table(["people"])).unwrap());
    assert_eq!(block_on(df.count()).unwrap(), 3);
    assert!(df.plan().to_string().contains("people"));
    let analyzed = block_on(df.analyzed()).unwrap();
    assert_eq!(analyzed.output.len(), 2);
}

#[test]
fn column_builders_and_functions_resolve_with_spark_types() {
    use krishiv_types::DataType;
    let s = session();
    let id = || col("id");
    let df = s.table("people").select([
        id().cast(DataType::Long).alias("long_id"),
        id().not_eq(1).alias("ne"),
        id().eq_null_safe(lit(1)).alias("nse"),
        id().lt(2).and(id().lt_eq(3)).or(id().gt_eq(0)).alias("cmp"),
        col("name").is_null().alias("missing"),
        col("name").like("b%").alias("liked"),
        id().is_in([lit(1), lit(2)]).alias("listed"),
        id().between(1, 2).alias("ranged"),
        (-id()).alias("neg"),
        (!col("name").is_null()).alias("present"),
        coalesce([col("name"), lit("none")]).alias("named"),
    ]);
    let schema = block_on(df.schema()).unwrap();
    let types: Vec<_> = schema
        .fields()
        .iter()
        .map(|f| (f.name().as_str(), f.data_type().clone()))
        .collect();
    assert_eq!(
        types,
        [
            ("long_id", ArrowType::Int64),
            ("ne", ArrowType::Boolean),
            ("nse", ArrowType::Boolean),
            ("cmp", ArrowType::Boolean),
            ("missing", ArrowType::Boolean),
            ("liked", ArrowType::Boolean),
            ("listed", ArrowType::Boolean),
            ("ranged", ArrowType::Boolean),
            ("neg", ArrowType::Int32),
            ("present", ArrowType::Boolean),
            ("named", ArrowType::Utf8View),
        ]
    );
    // Aggregates from the built-in registry, keyed by name.
    let df = s.table("people").group_by(["name"]).agg([
        avg(id()).alias("a"),
        min(id()).alias("lo"),
        max(id()).alias("hi"),
        count(id()).alias("n"),
        count_distinct([id(), col("name")]).alias("d"),
    ]);
    let schema = block_on(df.schema()).unwrap();
    let types: Vec<_> = schema
        .fields()
        .iter()
        .map(|f| f.data_type().clone())
        .collect();
    assert_eq!(
        types,
        [
            ArrowType::Utf8View,
            ArrowType::Float64,
            ArrowType::Int32,
            ArrowType::Int32,
            ArrowType::Int64,
            ArrowType::Int64,
        ]
    );
    // `sort` is `order_by`; both sort ascending by default.
    let sorted = block_on(s.table("people").sort([id().asc()]).collect()).unwrap();
    let ordered = block_on(s.table("people").order_by(["id"]).collect()).unwrap();
    assert_eq!(ids(&sorted), [1, 2, 3]);
    assert_eq!(ids(&sorted), ids(&ordered));
}

#[test]
fn temp_views_resolve_for_unqualified_names_only() {
    // Spark 4.2.0 with a temp view `v`: `v` and `V` resolve; `default.v`,
    // `spark_catalog.default.v` and `global_temp.v` fail with
    // TABLE_OR_VIEW_NOT_FOUND; `spark_catalog.v` fails with
    // REQUIRES_SINGLE_PART_NAMESPACE (not modelled here: it also fails).
    let s = session();
    let (schema, batch) = people();
    s.register_batches("v", schema, vec![batch]).unwrap();
    for ok in ["v", "V"] {
        assert_eq!(
            ids(&block_on(
                block_on(s.sql(&format!("SELECT * FROM {ok}")))
                    .unwrap()
                    .collect()
            )
            .unwrap()),
            [3, 1, 2],
            "{ok}"
        );
    }
    for bad in [
        "default.v",
        "DEFAULT.v",
        "spark_catalog.default.v",
        "global_temp.v",
        "spark_catalog.global_temp.v",
    ] {
        let sql = block_on(s.sql(&format!("SELECT * FROM {bad}")));
        assert_eq!(
            class(match sql {
                Ok(df) => block_on(df.schema()).map(|_| ()),
                Err(e) => Err(e),
            }),
            ErrorClass::TableOrViewNotFound,
            "{bad}"
        );
        assert_eq!(
            class(block_on(s.table(bad).schema())),
            ErrorClass::TableOrViewNotFound,
            "{bad}"
        );
    }
}

/// A session over `events(id INT, tag STRING)`, rows in this order:
/// (3, 'c'), (1, NULL), (2, 'b'), (5, NULL).
fn events_session(builder: krishiv_session::SessionBuilder) -> Session {
    use arrow_array::StringViewArray;
    let s = builder.build().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int32, false),
        Field::new("tag", ArrowType::Utf8View, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![3, 1, 2, 5])),
            Arc::new(StringViewArray::from(vec![
                Some("c"),
                None,
                Some("b"),
                None,
            ])),
        ],
    )
    .unwrap();
    s.register_batches("events", schema, vec![batch]).unwrap();
    s
}

/// One row of strings (NULL as `~`) from a query's result.
fn row(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        1,
        "{sql}"
    );
    batches[0]
        .columns()
        .iter()
        .map(|c| {
            if c.is_null(0) {
                return "~".to_owned();
            }
            match c.data_type() {
                ArrowType::Utf8View => c.as_string_view().value(0).to_owned(),
                ArrowType::Int32 => c.as_primitive::<Int32Type>().value(0).to_string(),
                ArrowType::Decimal128(_, s) => {
                    let v = c
                        .as_primitive::<arrow_array::types::Decimal128Type>()
                        .value(0);
                    format!("{v}e-{s}")
                }
                ArrowType::Timestamp(_, _) => c
                    .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
                    .value(0)
                    .to_string(),
                other => format!("{other}"),
            }
        })
        .collect()
}

fn sql_error(s: &Session, sql: &str) -> KrishivError {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_err()
}

/// Spark 4.2: `first`/`last` pick by input order, a NULL counting unless
/// `ignoreNulls`; `IGNORE NULLS` / `RESPECT NULLS` set the flag.
#[test]
fn first_and_last_aggregates() {
    let s = events_session(Session::builder());
    assert_eq!(
        row(
            &s,
            "SELECT first(tag), last(tag), first(tag, true), last(tag, true), first(id), last(id) FROM events"
        ),
        ["c", "~", "c", "b", "3", "5"]
    );
    assert_eq!(
        row(
            &s,
            "SELECT first_value(tag), last_value(tag), first_value(tag, true), last_value(tag, true) FROM events"
        ),
        ["c", "~", "c", "b"]
    );
    assert_eq!(
        row(
            &s,
            "SELECT first(tag) IGNORE NULLS, last(tag) IGNORE NULLS, first_value(tag) IGNORE NULLS FROM events"
        ),
        ["c", "b", "c"]
    );
    // RESPECT NULLS clears the flag, IGNORE NULLS sets it, whatever was written.
    assert_eq!(
        row(
            &s,
            "SELECT first(tag, false) IGNORE NULLS, last(tag, false) IGNORE NULLS, last(tag, true) RESPECT NULLS FROM events"
        ),
        ["c", "b", "~"]
    );
    // No rows: NULL.
    assert_eq!(
        row(
            &s,
            "SELECT first(tag), last(id, true) FROM events WHERE id > 100"
        ),
        ["~", "~"]
    );
    // Grouped, with the type of the argument.
    let batches = block_on(
        block_on(s.sql("SELECT id > 2 AS big, first(tag), last(tag, true) FROM events GROUP BY id > 2 ORDER BY big")).unwrap().collect(),
    )
    .unwrap();
    let b = &batches[0];
    assert_eq!(b.num_rows(), 2);
    let (first, last) = (b.column(1).as_string_view(), b.column(2).as_string_view());
    assert!(first.is_null(0) && last.value(0) == "b");
    assert_eq!((first.value(1), last.value(1)), ("c", "c"));
    // The generated name leaves the flag out, as Spark's does.
    let df = block_on(
        s.sql("SELECT first(tag, true), last_value(tag) IGNORE NULLS, first(id) FROM events"),
    )
    .unwrap();
    let schema = block_on(df.schema()).unwrap();
    let names: Vec<_> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["first(tag)", "last_value(tag)", "first(id)"]);
    // The flag must be a boolean literal; other functions refuse the clause.
    for sql in [
        "SELECT first(tag, id > 2) FROM events",
        "SELECT first(tag, NULL) FROM events",
        "SELECT last_value(id, 1) FROM events",
    ] {
        let e = sql_error(&s, sql);
        assert_eq!(e.error_class(), ErrorClass::LegacyErrorTemp1210, "{sql}");
    }
    for sql in [
        "SELECT first(id, true, true) FROM events",
        "SELECT first(id, true, 1) IGNORE NULLS FROM events",
    ] {
        let e = sql_error(&s, sql);
        assert_eq!(
            e.error_class(),
            ErrorClass::WrongNumArgsWithoutSuggestion,
            "{sql}"
        );
    }
    let e = sql_error(&s, "SELECT sum(id) IGNORE NULLS FROM events");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

/// The session principal and the query clock.
#[test]
fn session_user_and_local_timestamp() {
    use krishiv_plan::analyzer::{Principal, PrincipalKind};
    let s =
        events_session(Session::builder().principal(Principal::new("etl", PrincipalKind::Service)));
    assert_eq!(
        row(
            &s,
            "SELECT current_user(), user(), session_user(), current_user, user, session_user FROM events LIMIT 1"
        ),
        ["etl"; 6]
    );
    // `localtimestamp` is the session zone's wall clock, read from the same
    // query clock as `current_timestamp`.
    let s = events_session(Session::builder().config("spark.sql.session.timeZone", "+05:30"));
    let r = row(
        &s,
        "SELECT localtimestamp(), current_timestamp() FROM events LIMIT 1",
    );
    let (local, instant): (i64, i64) = (r[0].parse().unwrap(), r[1].parse().unwrap());
    assert_eq!(local - instant, 19_800_000_000);
    let df = block_on(s.sql("SELECT localtimestamp() FROM events")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(
        schema.field(0).data_type(),
        &ArrowType::Timestamp(arrow_schema::TimeUnit::Microsecond, None)
    );
    assert!(!schema.field(0).is_nullable());
    // `current_time` is a TIME, which has no Krishiv type yet.
    for sql in [
        "SELECT current_time() FROM events",
        "SELECT current_time(3) FROM events",
    ] {
        assert_eq!(
            sql_error(&s, sql).error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{sql}"
        );
    }
    // A zone that is not a fixed offset needs the tz database: refused, not
    // read as UTC.
    let s =
        events_session(Session::builder().config("spark.sql.session.timeZone", "America/New_York"));
    for sql in [
        "SELECT localtimestamp() FROM events",
        "SELECT current_date() FROM events",
    ] {
        assert_eq!(
            sql_error(&s, sql).error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{sql}"
        );
    }
}

/// Spark 4.2 folds `CAST(x AS FLOAT/DECIMAL/DOUBLE)` scales of `round`
/// (checked against the oracle).
#[test]
fn round_scales_written_as_casts_fold() {
    let s = events_session(Session::builder());
    let r = |sql: &str| row(&s, &format!("SELECT {sql} FROM events LIMIT 1"));
    assert_eq!(r("round(id, CAST(1.5 AS FLOAT))"), ["3"]);
    assert_eq!(r("round(2.345, CAST(1.5 AS FLOAT))"), ["23e-1"]);
    assert_eq!(r("round(2.345, CAST(2.5 AS DECIMAL(10,1)))"), ["235e-2"]);
    // A decimal target rounds 2.5 to 3 first: scale 3.
    assert_eq!(r("round(2.345, CAST(2.5 AS DECIMAL(10,0)))"), ["2345e-3"]);
    assert_eq!(r("bround(2.345, CAST(2.9 AS FLOAT))"), ["234e-2"]);
    assert_eq!(r("round(2.345, CAST(-0.5 AS FLOAT))"), ["2e-0"]);
    assert_eq!(r("round(1234.5, CAST(-1.5 AS DOUBLE))"), ["1230e-0"]);
    assert_eq!(r("round(2.345, CAST('2.7' AS DOUBLE))"), ["235e-2"]);
    assert_eq!(
        r("round(2.345, CAST(CAST(2.5 AS FLOAT) AS DOUBLE))"),
        ["235e-2"]
    );
    assert_eq!(r("round(2.345, CAST(NULL AS FLOAT))"), ["~"]);
    assert_eq!(r("round(2.345, 1.5)"), ["23e-1"]);
    // Failing casts raise Spark's error where the scale is read.
    for (sql, class) in [
        (
            "round(2.345, CAST(1e30 AS DOUBLE))",
            ErrorClass::CastOverflow,
        ),
        (
            "round(2.345, CAST(3000000000.5 AS FLOAT))",
            ErrorClass::CastOverflow,
        ),
        ("round(2.345, 1e30D)", ErrorClass::CastOverflow),
        (
            "round(2.345, CAST('x' AS FLOAT))",
            ErrorClass::CastInvalidInput,
        ),
        (
            "round(2.345, CAST(300 AS TINYINT))",
            ErrorClass::CastOverflow,
        ),
        (
            "round(2.345, CAST(99.5 AS DECIMAL(2,0)))",
            ErrorClass::NumericValueOutOfRangeWithSuggestion,
        ),
        // A column is not foldable: round checks its scale as an input,
        // ceil/floor as an argument.
        (
            "round(2.345, id)",
            ErrorClass::DatatypeMismatchNonFoldableInput,
        ),
        (
            "bround(2.5D, CAST(id AS FLOAT))",
            ErrorClass::DatatypeMismatchNonFoldableInput,
        ),
        ("floor(2.345, id)", ErrorClass::NonFoldableArgument),
        (
            "floor(2.345, CAST('x' AS INT))",
            ErrorClass::CastInvalidInput,
        ),
        (
            "ceil(2.345, CAST(300 AS TINYINT))",
            ErrorClass::NonFoldableArgument,
        ),
        (
            "ceil(2.345, CAST(1.5 AS FLOAT))",
            ErrorClass::NonFoldableArgument,
        ),
    ] {
        let e = sql_error(&s, &format!("SELECT {sql} FROM events"));
        assert_eq!(e.error_class(), class, "{sql}: {e}");
    }
}

/// A table registered from `Utf8` batches aggregates strings like `Utf8View` ones.
#[test]
fn string_aggregates_over_utf8_tables() {
    let s = session();
    assert_eq!(
        row(
            &s,
            "SELECT min(name), max(name), first(name), last(name, true) FROM people"
        ),
        ["b", "c", "c", "b"]
    );
}
