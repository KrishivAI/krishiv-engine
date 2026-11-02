// SPDX-License-Identifier: Apache-2.0
//! Intervals in files, as Spark 4.2.0 stores them (design P2-T001b §3). Each
//! expected value is from a Spark 4.2.0 local run.
#![forbid(unsafe_code)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn session() -> R<Session> {
    Ok(Session::new()?)
}

/// `sql`'s first column as strings, `NULL` as `None`.
fn strings(s: &Session, sql: &str) -> R<Vec<Option<String>>> {
    block_on(async {
        let df = s.sql(sql).await?;
        let schema = df.schema().await?;
        let name = schema.fields()[0].name().replace('`', "``");
        let batches = df
            .select_expr([format!("CAST(`{name}` AS STRING) AS v")])?
            .collect()
            .await?;
        let mut out = Vec::new();
        for b in &batches {
            out.extend(
                b.column(0)
                    .as_string_view()
                    .iter()
                    .map(|v| v.map(str::to_owned)),
            );
        }
        Ok(out)
    })
}

fn run(s: &Session, sql: &str) -> Result<(), KrishivError> {
    block_on(async {
        s.sql(sql).await?.collect().await?;
        Ok(())
    })
}

fn class_of(s: &Session, sql: &str) -> String {
    run(s, sql).map_or_else(|e| e.error_class().name().to_owned(), |()| "ok".to_owned())
}

/// The type of each column of `table`.
fn types(s: &Session, table: &str) -> R<Vec<String>> {
    block_on(async {
        let schema = s
            .sql(&format!("SELECT * FROM {table}"))
            .await?
            .schema()
            .await?;
        Ok(schema
            .fields()
            .iter()
            .map(|f| {
                krishiv_types::DataType::from_arrow_field(f)
                    .map_or_else(|e| e.to_string(), |t| t.simple_string())
            })
            .collect())
    })
}

const ALL_KINDS: &str = "SELECT interval '1-2' year to month a, interval '3' month b, \
    interval '1 02' day to hour c, interval '1 02:03:04.5' day to second d, \
    interval '7' second e, array(interval '1' month) f, named_struct('x', interval '1' day) g, \
    map('k', interval '2' hour) h, cast(null as interval day) n";

#[test]
fn a_parquet_table_keeps_every_interval_type() -> R {
    let s = session()?;
    run(&s, &format!("CREATE TABLE t USING parquet AS {ALL_KINDS}"))?;
    assert_eq!(
        types(&s, "t")?,
        [
            "interval year to month",
            "interval month",
            "interval day to hour",
            "interval day to second",
            "interval second",
            "array<interval month>",
            "struct<x:interval day>",
            "map<string,interval hour>",
            "interval day",
        ]
    );
    let text = "SELECT concat_ws('|', cast(a as string), cast(b as string), cast(c as string), \
        cast(d as string), cast(e as string), cast(f as string), cast(g as string), \
        cast(h as string), coalesce(cast(n as string), 'NULL')) FROM t";
    assert_eq!(
        strings(&s, text)?,
        [Some(
            "INTERVAL '1-2' YEAR TO MONTH|INTERVAL '3' MONTH|INTERVAL '1 02' DAY TO HOUR|\
             INTERVAL '1 02:03:04.5' DAY TO SECOND|INTERVAL '07' SECOND|[INTERVAL '1' MONTH]|\
             {INTERVAL '1' DAY}|{k -> INTERVAL '02' HOUR}|NULL"
                .to_owned()
        )]
    );
    Ok(())
}

#[test]
fn filters_on_stored_intervals_keep_their_rows() -> R {
    let s = session()?;
    run(
        &s,
        "CREATE TABLE t USING parquet AS SELECT * FROM VALUES \
         (interval '1' month, interval '1' day), (interval '3' month, interval '3' day), \
         (NULL, NULL) AS v(a, b)",
    )?;
    let count = |p: &str| strings(&s, &format!("SELECT count(*) FROM t WHERE {p}"));
    assert_eq!(count("a > interval '1' month")?, [Some("1".to_owned())]);
    assert_eq!(count("b <= interval '48' hour")?, [Some("1".to_owned())]);
    assert_eq!(count("b = interval '1' day")?, [Some("1".to_owned())]);
    assert_eq!(count("a IS NULL")?, [Some("1".to_owned())]);
    assert_eq!(count("b IS NOT NULL")?, [Some("2".to_owned())]);
    assert_eq!(
        strings(&s, "SELECT cast(max(a) as string) FROM t")?,
        [Some("INTERVAL '3' MONTH".to_owned())]
    );
    assert_eq!(
        strings(&s, "SELECT cast(sum(b) as string) FROM t")?,
        [Some("INTERVAL '4' DAY".to_owned())]
    );
    Ok(())
}

#[test]
fn a_table_declared_with_interval_columns_takes_inserts() -> R {
    let s = session()?;
    run(
        &s,
        "CREATE TABLE t (a interval day, b interval month) USING parquet",
    )?;
    run(
        &s,
        "INSERT INTO t VALUES (interval '2' day, interval '3' month)",
    )?;
    assert_eq!(
        strings(&s, "SELECT cast(a + a as string) FROM t")?,
        [Some("INTERVAL '4' DAY".to_owned())]
    );
    Ok(())
}

#[test]
fn no_table_or_file_holds_a_calendar_interval() -> R {
    let s = session()?;
    for format in ["parquet", "csv", "json", "avro"] {
        assert_eq!(
            class_of(
                &s,
                &format!("CREATE TABLE a USING {format} AS SELECT cast('1 day' as interval) c")
            ),
            "UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE",
            "{format}"
        );
        assert_eq!(
            class_of(
                &s,
                &format!(
                    "CREATE TABLE b USING {format} AS SELECT array(cast('1 day' as interval)) c"
                )
            ),
            "UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE",
            "{format} nested"
        );
        assert_eq!(
            class_of(&s, &format!("CREATE TABLE c (a interval) USING {format}")),
            "UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE",
            "{format} declared"
        );
    }
    Ok(())
}

/// A temporary directory holding `file` with `text`.
fn file_dir(file: &str, text: &str) -> R<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join(file), text)?;
    Ok(dir)
}

/// The rows of the files in `path`, under a table of `format` declaring
/// `schema`, column `a` as text.
fn read_text(
    s: &Session,
    format: &str,
    schema: &str,
    mode: &str,
    path: &std::path::Path,
) -> R<Vec<Option<String>>> {
    run(s, "DROP TABLE IF EXISTS declared")?;
    run(
        s,
        &format!(
            "CREATE TABLE declared ({schema}) USING {format} OPTIONS (mode '{mode}') LOCATION '{}'",
            path.display()
        ),
    )?;
    strings(s, "SELECT CAST(a AS STRING) FROM declared")
}

fn texts(v: &[Option<&str>]) -> Vec<Option<String>> {
    v.iter().map(|x| x.map(str::to_owned)).collect()
}

const TEXT_ROW: &str = "SELECT interval '1-2' year to month a, interval '3' month b, \
    interval '1 02' day to hour c, interval '-1 02:03:04.5' day to second d, \
    cast(null as interval day) n";

#[test]
fn csv_and_json_write_the_ansi_string() -> R {
    let s = session()?;
    for (format, expected) in [
        (
            "csv",
            "INTERVAL '1-2' YEAR TO MONTH,INTERVAL '3' MONTH,INTERVAL '1 02' DAY TO HOUR,\
             INTERVAL '-1 02:03:04.5' DAY TO SECOND,\n",
        ),
        (
            "json",
            "{\"a\":\"INTERVAL '1-2' YEAR TO MONTH\",\"b\":\"INTERVAL '3' MONTH\",\
             \"c\":\"INTERVAL '1 02' DAY TO HOUR\",\
             \"d\":\"INTERVAL '-1 02:03:04.5' DAY TO SECOND\"}\n",
        ),
    ] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("out");
        block_on(async {
            s.sql(TEXT_ROW)
                .await?
                .write()
                .format(format)
                .save(path.to_string_lossy().as_ref())
                .await
        })?;
        let part = std::fs::read_dir(&path)?
            .filter_map(Result::ok)
            .find(|e| e.file_name().to_string_lossy().starts_with("part-"))
            .ok_or("no part file")?;
        assert_eq!(std::fs::read_to_string(part.path())?, expected, "{format}");
    }
    Ok(())
}

#[test]
fn csv_and_json_tables_read_their_intervals_back() -> R {
    let s = session()?;
    for format in ["csv", "json"] {
        run(
            &s,
            &format!("CREATE TABLE t{format} (a interval day, b interval month) USING {format}"),
        )?;
        run(
            &s,
            &format!(
                "INSERT INTO t{format} VALUES (interval '2' day, interval '3' month), (NULL, NULL)"
            ),
        )?;
        assert_eq!(
            strings(
                &s,
                &format!("SELECT cast(a + a as string) FROM t{format} ORDER BY a NULLS LAST")
            )?,
            [Some("INTERVAL '4' DAY".to_owned()), None],
            "{format}"
        );
        assert_eq!(
            types(&s, &format!("t{format}"))?,
            ["interval day", "interval month"]
        );
    }
    Ok(())
}

#[test]
fn bad_csv_interval_text_follows_the_parse_mode() -> R {
    let s = session()?;
    let dir = file_dir("m.csv", "INTERVAL '3' MONTH\nabc\n5\n\n'1-2'\n")?;
    let good = Some("INTERVAL '3' MONTH");
    let five = Some("INTERVAL '5' MONTH");
    assert_eq!(
        read_text(&s, "csv", "a interval month", "PERMISSIVE", dir.path())?,
        texts(&[good, None, five, None])
    );
    assert_eq!(
        read_text(&s, "csv", "a interval month", "DROPMALFORMED", dir.path())?,
        texts(&[good, five])
    );
    assert!(read_text(&s, "csv", "a interval month", "FAILFAST", dir.path()).is_err());
    Ok(())
}

#[test]
fn bad_json_intervals_follow_the_parse_mode() -> R {
    let s = session()?;
    let dir = file_dir(
        "m.json",
        "{\"a\":\"INTERVAL '3' DAY\"}\n{\"a\":5}\n{\"a\":\"abc\"}\n{\"a\":\"3\"}\n{\"a\":null}\n",
    )?;
    let three = Some("INTERVAL '3' DAY");
    assert_eq!(
        read_text(&s, "json", "a interval day", "PERMISSIVE", dir.path())?,
        texts(&[three, None, None, three, None])
    );
    assert_eq!(
        read_text(&s, "json", "a interval day", "DROPMALFORMED", dir.path())?,
        texts(&[three, three, None])
    );
    assert!(read_text(&s, "json", "a interval day", "FAILFAST", dir.path()).is_err());
    assert_eq!(
        read_text(&s, "json", "a interval hour", "PERMISSIVE", dir.path())?,
        texts(&[None, None, None, Some("INTERVAL '03' HOUR"), None])
    );
    Ok(())
}

#[test]
fn csv_refuses_a_nested_interval_and_text_any_interval() -> R {
    let s = session()?;
    for (format, q, message) in [
        (
            "csv",
            "SELECT array(interval '1' day) a",
            "The CSV datasource doesn't support the column `a` of the type \"ARRAY<INTERVAL DAY>\"",
        ),
        (
            "text",
            "SELECT interval '1-2' year to month a",
            "The Text datasource doesn't support the column `a` of the type \"INTERVAL YEAR TO MONTH\"",
        ),
        (
            "text",
            "SELECT 1L id",
            "The Text datasource doesn't support the column `id` of the type \"BIGINT\"",
        ),
        (
            "parquet",
            "SELECT array(cast('1 day' as interval)) c",
            "The Parquet datasource doesn't support the column `c` of the type \"ARRAY<INTERVAL>\"",
        ),
    ] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("out");
        let e = block_on(async {
            s.sql(q)
                .await?
                .write()
                .format(format)
                .save(path.to_string_lossy().as_ref())
                .await
        })
        .expect_err(format);
        assert_eq!(
            e.error_class().name(),
            "UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE"
        );
        assert!(e.to_string().contains(message), "{e}");
    }
    Ok(())
}

/// Spark's reader goes by the physical type: a year-month file reads as numbers,
/// a day-time file as `bigint` or `decimal`, other fields of one family freely.
#[test]
fn a_stored_interval_reads_under_another_declared_type_as_spark_does() -> R {
    let s = session()?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("iv");
    block_on(async {
        s.sql("SELECT interval '3' month a, interval '1' day b")
            .await?
            .write()
            .format("parquet")
            .save(path.to_string_lossy().as_ref())
            .await
    })?;
    let read = |schema: &str, column: &str| -> Result<String, KrishivError> {
        run(&s, "DROP TABLE IF EXISTS declared")?;
        run(
            &s,
            &format!(
                "CREATE TABLE declared ({schema}) USING parquet LOCATION '{}'",
                path.display()
            ),
        )?;
        block_on(async {
            let batches = s
                .sql(&format!("SELECT CAST({column} AS STRING) FROM declared"))
                .await?
                .collect()
                .await?;
            Ok(batches[0].column(0).as_string_view().value(0).to_owned())
        })
    };
    for (schema, column, expected) in [
        ("a int", "a", "3"),
        ("a bigint", "a", "3"),
        ("a smallint", "a", "3"),
        ("a double", "a", "3.0"),
        ("a decimal(10,0)", "a", "3"),
        ("a interval year", "a", "INTERVAL '0' YEAR"),
        ("b bigint", "b", "86400000000"),
        ("b decimal(20,0)", "b", "86400000000"),
        ("b interval hour", "b", "INTERVAL '24' HOUR"),
    ] {
        assert_eq!(read(schema, column)?, expected, "{schema}");
    }
    for (schema, column) in [
        ("a string", "a"),
        ("a interval day", "a"),
        ("b int", "b"),
        ("b double", "b"),
        ("b string", "b"),
        ("b interval month", "b"),
    ] {
        assert_eq!(
            read(schema, column).expect_err(schema).error_class().name(),
            "FAILED_READ_FILE.PARQUET_COLUMN_DATA_TYPE_MISMATCH",
            "{schema}"
        );
    }
    Ok(())
}

#[test]
fn a_file_spark_wrote_reads_with_its_types_and_values() -> R {
    let s = session()?;
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/interval-storage");
    let table = format!("parquet.`{}`", path.display());
    assert_eq!(
        types(&s, &table)?,
        [
            "interval year to month",
            "interval month",
            "interval day to hour",
            "interval day to second",
            "interval second",
            "array<interval month>",
            "struct<x:interval day>",
            "map<string,interval hour>",
            "interval day",
        ]
    );
    assert_eq!(
        strings(
            &s,
            &format!(
                "SELECT concat_ws('|', cast(a as string), cast(b as string), cast(c as string), \
                 cast(d as string), cast(e as string), cast(f as string), cast(g as string), \
                 cast(h as string), coalesce(cast(n as string), 'NULL')) FROM {table}"
            )
        )?,
        [Some(
            "INTERVAL '1-2' YEAR TO MONTH|INTERVAL '3' MONTH|INTERVAL '1 02' DAY TO HOUR|\
             INTERVAL '1 02:03:04.5' DAY TO SECOND|INTERVAL '07' SECOND|[INTERVAL '1' MONTH]|\
             {INTERVAL '1' DAY}|{k -> INTERVAL '02' HOUR}|NULL"
                .to_owned()
        )]
    );
    Ok(())
}

/// Keeps a file for Spark to read back (`KRISHIV_INTERVAL_FILE_DIR`; see
/// `tests/scripts/gen_interval_storage_fixture.py`).
#[test]
fn a_file_krishiv_wrote_has_spark_s_physical_types_and_footer() -> R {
    let s = session()?;
    let dir = tempfile::tempdir()?;
    let kept = std::env::var("KRISHIV_INTERVAL_FILE_DIR").ok();
    let path = kept.map_or_else(|| dir.path().join("t"), std::path::PathBuf::from);
    block_on(async {
        s.sql(ALL_KINDS)
            .await?
            .write()
            .format("parquet")
            .save(path.to_string_lossy().as_ref())
            .await
    })?;
    let part = std::fs::read_dir(&path)?
        .filter_map(Result::ok)
        .find(|e| e.file_name().to_string_lossy().ends_with(".parquet"))
        .ok_or("no part file")?;
    let reader =
        parquet::file::reader::SerializedFileReader::new(std::fs::File::open(part.path())?)?;
    let meta = parquet::file::reader::FileReader::metadata(&reader).file_metadata();
    use parquet::basic::Type;
    let leaves: Vec<(String, Type)> = meta
        .schema_descr()
        .columns()
        .iter()
        .map(|c| (c.name().to_owned(), c.physical_type()))
        .collect();
    let of = |name: &str| leaves.iter().find(|(n, _)| n == name).map(|(_, t)| *t);
    assert_eq!(of("a"), Some(Type::INT32));
    assert_eq!(of("b"), Some(Type::INT32));
    assert_eq!(of("c"), Some(Type::INT64));
    assert_eq!(of("d"), Some(Type::INT64));
    assert_eq!(of("element"), Some(Type::INT32));
    assert_eq!(of("x"), Some(Type::INT64));
    assert_eq!(of("value"), Some(Type::INT64));
    let json = meta
        .key_value_metadata()
        .and_then(|kv| {
            kv.iter()
                .find(|k| k.key == "org.apache.spark.sql.parquet.row.metadata")
        })
        .and_then(|k| k.value.clone())
        .ok_or("no row metadata")?;
    assert!(json.contains("\"type\":\"interval day to hour\""), "{json}");
    assert!(
        json.contains("\"elementType\":\"interval month\""),
        "{json}"
    );
    assert!(json.contains("\"valueType\":\"interval hour\""), "{json}");
    Ok(())
}

const ALL_KINDS_TYPES: [&str; 9] = [
    "interval year to month",
    "interval month",
    "interval day to hour",
    "interval day to second",
    "interval second",
    "array<interval month>",
    "struct<x:interval day>",
    "map<string,interval hour>",
    "interval day",
];

/// `table`, holding [`ALL_KINDS`], has its types and values.
fn assert_all_kinds(s: &Session, table: &str) -> R {
    assert_eq!(types(s, table)?, ALL_KINDS_TYPES);
    assert_eq!(
        strings(
            s,
            &format!(
                "SELECT concat_ws('|', cast(a as string), cast(b as string), cast(c as string), \
                 cast(d as string), cast(e as string), cast(f as string), cast(g as string), \
                 cast(h as string), coalesce(cast(n as string), 'NULL')) FROM {table}"
            )
        )?,
        [Some(
            "INTERVAL '1-2' YEAR TO MONTH|INTERVAL '3' MONTH|INTERVAL '1 02' DAY TO HOUR|\
             INTERVAL '1 02:03:04.5' DAY TO SECOND|INTERVAL '07' SECOND|[INTERVAL '1' MONTH]|\
             {INTERVAL '1' DAY}|{k -> INTERVAL '02' HOUR}|NULL"
                .to_owned()
        )]
    );
    Ok(())
}

#[test]
fn an_avro_table_keeps_every_interval_type() -> R {
    let s = session()?;
    run(&s, &format!("CREATE TABLE t USING avro AS {ALL_KINDS}"))?;
    assert_all_kinds(&s, "t")
}

#[test]
fn an_avro_file_spark_wrote_reads_with_its_types_and_values() -> R {
    let s = session()?;
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/interval-storage-avro");
    assert_all_kinds(&s, &format!("avro.`{}`", path.display()))
}

/// Keeps a file for Spark to read back (`KRISHIV_INTERVAL_AVRO_DIR`).
#[test]
fn an_avro_file_krishiv_wrote_has_spark_s_schema_properties() -> R {
    let s = session()?;
    let dir = tempfile::tempdir()?;
    let kept = std::env::var("KRISHIV_INTERVAL_AVRO_DIR").ok();
    let path = kept.map_or_else(|| dir.path().join("t"), std::path::PathBuf::from);
    block_on(async {
        s.sql(ALL_KINDS)
            .await?
            .write()
            .format("avro")
            .save(path.to_string_lossy().as_ref())
            .await
    })?;
    let part = std::fs::read_dir(&path)?
        .filter_map(Result::ok)
        .find(|e| e.file_name().to_string_lossy().ends_with(".avro"))
        .ok_or("no part file")?;
    let bytes = std::fs::read(part.path())?;
    let text = String::from_utf8_lossy(&bytes);
    for fragment in [
        r#"{"type":"int","spark.sql.catalyst.type":"interval year to month"}"#,
        r#"{"type":"int","spark.sql.catalyst.type":"interval month"}"#,
        r#"{"type":"long","spark.sql.catalyst.type":"interval day to hour"}"#,
        r#"{"type":"long","spark.sql.catalyst.type":"interval day to second"}"#,
        r#"{"type":"long","spark.sql.catalyst.type":"interval second"}"#,
        r#"{"type":"long","spark.sql.catalyst.type":"interval day"}"#,
        r#"{"type":"long","spark.sql.catalyst.type":"interval hour"}"#,
    ] {
        assert!(text.contains(fragment), "{fragment} in {text}");
    }
    Ok(())
}

/// The cases where Spark's Avro reader and the shared schema mapper agree.
/// Spark also refuses `smallint` and `decimal` over these columns, which the
/// mapper allows for every format.
#[test]
fn a_stored_avro_interval_reads_under_another_declared_type_as_spark_does() -> R {
    let s = session()?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("iv");
    block_on(async {
        s.sql("SELECT interval '3' month a, interval '1' day b, 14 i, 90000000000 l")
            .await?
            .write()
            .format("avro")
            .save(path.to_string_lossy().as_ref())
            .await
    })?;
    let read = |schema: &str, column: &str| -> Result<String, KrishivError> {
        run(&s, "DROP TABLE IF EXISTS declared")?;
        run(
            &s,
            &format!(
                "CREATE TABLE declared ({schema}) USING avro LOCATION '{}'",
                path.display()
            ),
        )?;
        block_on(async {
            let batches = s
                .sql(&format!("SELECT CAST({column} AS STRING) FROM declared"))
                .await?
                .collect()
                .await?;
            Ok(batches[0].column(0).as_string_view().value(0).to_owned())
        })
    };
    for (schema, column, expected) in [
        ("a int", "a", "3"),
        ("a bigint", "a", "3"),
        ("a double", "a", "3.0"),
        ("a interval year", "a", "INTERVAL '0' YEAR"),
        ("b bigint", "b", "86400000000"),
        ("b interval hour", "b", "INTERVAL '24' HOUR"),
        ("i interval month", "i", "INTERVAL '14' MONTH"),
        ("l interval day", "l", "INTERVAL '1' DAY"),
    ] {
        assert_eq!(read(schema, column)?, expected, "{schema}");
    }
    for (schema, column) in [
        ("a string", "a"),
        ("a interval day", "a"),
        ("b int", "b"),
        ("b double", "b"),
        ("b string", "b"),
        ("b interval month", "b"),
        ("i interval day", "i"),
        ("l interval month", "l"),
    ] {
        assert!(read(schema, column).is_err(), "{schema}");
    }
    Ok(())
}

/// `avroSchema` on a write and a read, each answer from Spark 4.2.0.
#[test]
fn avro_schema_option_meets_intervals_as_spark_does() -> R {
    let s = session()?;
    let dir = tempfile::tempdir()?;
    let schema =
        |t: &str| format!(r#"{{"type":"record","name":"r","fields":[{{"name":"a","type":{t}}}]}}"#);
    let write = |sql: &str, t: &str, to: &str| -> Result<(), KrishivError> {
        block_on(async {
            s.sql(sql)
                .await?
                .write()
                .format("avro")
                .option("avroSchema", &schema(t))
                .save(dir.path().join(to).to_string_lossy().as_ref())
                .await
        })
    };
    let table = |to: &str| format!("avro.`{}`", dir.path().join(to).display());
    let prop = |avro: &str, spark: &str| {
        format!(r#"{{"type":"{avro}","spark.sql.catalyst.type":"{spark}"}}"#)
    };

    write("SELECT interval '1-2' year to month a", r#""int""#, "ym")?;
    assert_eq!(types(&s, &table("ym"))?, ["int"]);
    assert_eq!(
        strings(&s, &format!("SELECT a FROM {}", table("ym")))?,
        [Some("14".to_owned())]
    );
    write("SELECT interval '1' day a", r#""long""#, "dt")?;
    assert_eq!(types(&s, &table("dt"))?, ["bigint"]);
    assert_eq!(
        strings(&s, &format!("SELECT a FROM {}", table("dt")))?,
        [Some("86400000000".to_owned())]
    );
    write(
        "SELECT interval '1' day a",
        &prop("long", "interval day"),
        "p",
    )?;
    assert_eq!(types(&s, &table("p"))?, ["interval day"]);
    write("SELECT 5 a", &prop("int", "interval month"), "n")?;
    assert_eq!(
        strings(&s, &format!("SELECT a FROM {}", table("n")))?,
        [Some("INTERVAL '5' MONTH".to_owned())]
    );
    assert!(write("SELECT interval '1' month a", r#""long""#, "x1").is_err());
    assert!(write("SELECT interval '1' day a", r#""int""#, "x2").is_err());

    // A map's interval key has no Avro form.
    assert!(
        block_on(async {
            s.sql("SELECT map(interval '1' day, 1) a")
                .await?
                .write()
                .format("avro")
                .save(dir.path().join("k").to_string_lossy().as_ref())
                .await
        })
        .is_err()
    );
    Ok(())
}

/// Spark 4.2.0 over `range`, so its generated code: the days aren't hashed
/// (design P2-T001b Q3).
#[test]
fn a_calendar_interval_hashes_as_spark_s_generated_code() -> R {
    let s = session()?;
    let row = |id: i32, f: &str| -> R<String> {
        let q = format!(
            "SELECT CAST({f} AS STRING) FROM (SELECT {id}L id, \
             make_interval(CAST({id}L AS INT), 2, 0, 3, 0, 0, 4.0) c)"
        );
        Ok(strings(&s, &q)?[0].clone().ok_or("null")?)
    };
    for (id, hash, xx) in [
        (1, "-1178828624", "4189764414194776525"),
        (2, "650588460", "30787519031835256"),
    ] {
        assert_eq!(row(id, "hash(c)")?, hash);
        assert_eq!(row(id, "xxhash64(c)")?, xx);
        assert_eq!(row(id, "hash(array(c))")?, hash);
        assert_eq!(row(id, "hash(named_struct('a', c))")?, hash);
        assert_eq!(row(id, "xxhash64(array(c))")?, xx);
    }
    assert_eq!(row(1, "hash(id, c)")?, "260639151");
    assert_eq!(row(2, "hash(id, c)")?, "-137465629");
    assert_eq!(row(1, "hash(cast(null as interval))")?, "42");
    Ok(())
}

#[test]
fn ansi_intervals_hash_as_spark_does() -> R {
    let s = session()?;
    assert_eq!(
        strings(&s, "SELECT hash(interval '1' month)")?,
        [Some("-559580957".to_owned())]
    );
    assert_eq!(
        strings(&s, "SELECT hash(interval '1' day)")?,
        [Some("892267096".to_owned())]
    );
    assert_eq!(
        strings(&s, "SELECT xxhash64(interval '1' month)")?,
        [Some("-6698625589789238999".to_owned())]
    );
    assert_eq!(
        strings(&s, "SELECT xxhash64(interval '1' day)")?,
        [Some("5650310968337874586".to_owned())]
    );
    Ok(())
}
