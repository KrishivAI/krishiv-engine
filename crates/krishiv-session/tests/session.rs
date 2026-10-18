// SPDX-License-Identifier: Apache-2.0
//! Sessions end to end: SQL and DataFrames through analysis, the optimizer,
//! the physical planner and the executor; configuration snapshots and
//! analysis reuse; temporary tables and views; reads by path.
#![cfg(test)]

use std::sync::Arc;

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
    // Relations the physical planner does not have yet fail clearly.
    let df = s.table("people").filter(col("id").gt(lit(1)));
    assert_eq!(
        class(block_on(df.collect())),
        ErrorClass::KrishivUnsupportedFeature
    );
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
    assert_eq!(
        ids(&block_on(s.table(" default.`x.y` ").collect()).unwrap()),
        [3, 1, 2]
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
