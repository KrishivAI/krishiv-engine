// SPDX-License-Identifier: Apache-2.0
//! File reads with a user schema (`spark.read.schema(..)`, KN-1) against Spark 4.2.0: every case of
//! `tests/fixtures/spark-4.2.0/read-user-schema.json` (generator `tests/scripts/gen_read_user_schema.py`, a real
//! Spark run) runs here on the same file tree and must give Spark's reported schema (names, types, nullability) and
//! rows (each column cast to STRING in UTC, sorted), or its error: the condition and message parameters, or, for a
//! failure Spark raises from a task, the condition chain. `KNOWN_GAPS` lists the cases not matched yet, each with
//! its reason; they must still differ, so a fix removes them from the list.
#![expect(clippy::unwrap_used, reason = "test code")]

use std::collections::BTreeMap;
use std::path::Path;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::{DataFrame, Session, functions::col};
use krishiv_types::{DataType, StructField, StructType};
use serde_json::{Value, json};

/// Cases not matched yet, with the reason; each must still differ from Spark.
const KNOWN_GAPS: &[(&str, &str)] = &[
    (
        "json wrong types FAILFAST",
        "a JSON string for an INT field fails without the CANNOT_PARSE_JSON_FIELD cause (krishiv-io json/convert.rs \
         parses the string as a number; its own test expects no cause): class and rows match",
    ),
    (
        "json string for int FAILFAST",
        "as above: the chain lacks CANNOT_PARSE_JSON_FIELD",
    ),
    (
        "json nested struct field absent from data",
        "the JSON reader has flat types only (KRISHIV_UNSUPPORTED_FEATURE for a struct column)",
    ),
    (
        "parquet name case differs, caseSensitive=true",
        "spark.sql.caseSensitive does not reach the file scan's column mapping (krishiv-io FieldMapping::ByName is \
         always case-insensitive)",
    ),
];

fn fixture() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/read-user-schema.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn write_text(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// Writes the rows of `sql` as Parquet under `dir`.
fn parquet(s: &Session, sql: &str, dir: &Path) {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.write().parquet(dir.to_str().unwrap())).unwrap();
}

/// The generator's tree, under `root`.
fn layout(s: &Session, root: &Path) {
    parquet(
        s,
        "SELECT 1 AS a, 'x' AS b, 10L AS c UNION ALL SELECT 2, 'y', 20L",
        &root.join("pq"),
    );
    parquet(s, "SELECT 1 AS a, 'x' AS B", &root.join("pqcase"));
    parquet(s, "SELECT 1 AS a, 'x' AS b", &root.join("pqnum"));
    for k in ["1", "2", "abc"] {
        parquet(
            s,
            &format!("SELECT 'v{k}' AS v"),
            &root.join(format!("pp/p/k={k}")),
        );
        parquet(
            s,
            &format!("SELECT 'v{k}' AS v, 'in{k}' AS k"),
            &root.join(format!("ppk/p/k={k}")),
        );
        write_text(
            &root.join(format!("pc/p/k={k}/f.csv")),
            &format!("v{k},{k}0\n"),
        );
    }
    for k in ["1", "2"] {
        parquet(
            s,
            &format!("SELECT 'v{k}' AS v"),
            &root.join(format!("pp1/p/k={k}")),
        );
        write_text(
            &root.join(format!("pc1/p/k={k}/f.csv")),
            &format!("v{k},{k}0\n"),
        );
    }
    let files = [
        (
            "js/a.json",
            "{\"a\":1,\"b\":\"x\"}\n{\"a\":2}\n{\"b\":\"z\",\"a\":3,\"extra\":9}\n",
        ),
        (
            "jbad/a.json",
            "{\"a\":1,\"b\":\"x\"}\n{\"a\":\"x\",\"b\":\"y\"}\n{\"a\":1.5,\"b\":\"z\"}\n{\"a\":4,\"b\":{\"q\":1}}\n",
        ),
        (
            "jbad1/a.json",
            "{\"a\":1,\"b\":\"x\"}\n{\"a\":\"x\",\"b\":\"y\"}\n",
        ),
        (
            "jbad2/a.json",
            "{\"a\":1,\"b\":\"x\"}\n{\"a\":1.5,\"b\":\"y\"}\n",
        ),
        (
            "jbad3/a.json",
            "{\"a\":1,\"b\":\"x\"}\n{\"a\":4,\"b\":{\"q\":1}}\n",
        ),
        (
            "jnest/a.json",
            "{\"a\":1,\"s\":{\"p\":1}}\n{\"a\":2}\n{\"a\":3,\"s\":{\"p\":2,\"q\":\"t\"}}\n",
        ),
        (
            "jcase/a.json",
            "{\"a\":1,\"b\":\"x\"}\n{\"a\":2,\"b\":\"y\"}\n",
        ),
        (
            "jmalformed/a.json",
            "{\"a\":1,\"b\":\"x\"}\nnot json\n{\"a\":3,\"b\":\"z\"}\n",
        ),
        ("csv/a.csv", "1,x\n2,y\n"),
        ("txt/a.txt", "line1\nline2\n"),
    ];
    for (path, text) in files {
        write_text(&root.join(path), text);
    }
}

/// How a case reads.
enum Read {
    /// `spark.read.schema(..).options(..).format(..).load(path)`; the schema as a DDL string, or as a struct type
    /// built in code when `built` is set; `None` reads without a schema.
    Files {
        format: &'static str,
        path: String,
        schema: Option<String>,
        built: bool,
        options: Vec<(&'static str, &'static str)>,
    },
    /// `CREATE TEMPORARY VIEW v (cols) USING format OPTIONS (path ..)` then `SELECT * FROM v`.
    View {
        name: &'static str,
        columns: &'static str,
        format: &'static str,
        path: &'static str,
    },
}

fn read(format: &'static str, path: &str, schema: &str) -> Read {
    Read::Files {
        format,
        path: path.to_owned(),
        schema: Some(schema.to_owned()),
        built: false,
        options: Vec::new(),
    }
}

fn with(r: Read, opts: &[(&'static str, &'static str)]) -> Read {
    match r {
        Read::Files {
            format,
            path,
            schema,
            built,
            ..
        } => Read::Files {
            format,
            path,
            schema,
            built,
            options: opts.to_vec(),
        },
        view @ Read::View { .. } => view,
    }
}

fn no_schema(format: &'static str, path: &str) -> Read {
    Read::Files {
        format,
        path: path.to_owned(),
        schema: None,
        built: false,
        options: Vec::new(),
    }
}

/// A JSON case: name, directory, schema and options.
type JsonCase = (
    &'static str,
    &'static str,
    &'static str,
    &'static [(&'static str, &'static str)],
);

/// The generator's cases, by name, in its order.
#[expect(clippy::too_many_lines, reason = "one table mirroring the generator")]
fn cases() -> Vec<(String, Read)> {
    let mut out: Vec<(String, Read)> = Vec::new();
    let mut add = |name: String, r: Read| out.push((name, r));
    let nn = "a INT NOT NULL, b STRING NOT NULL";
    for (fmt, path) in [("parquet", "pq"), ("json", "js"), ("csv", "csv")] {
        add(
            format!("nullability {fmt} StructType NOT NULL"),
            Read::Files {
                format: fmt,
                path: path.to_owned(),
                schema: Some(nn.to_owned()),
                built: true,
                options: Vec::new(),
            },
        );
        add(
            format!("nullability {fmt} DDL NOT NULL"),
            read(fmt, path, nn),
        );
    }
    add(
        "nullability json NOT NULL with a null in data".into(),
        with(read("json", "js", nn), &[("mode", "FAILFAST")]),
    );
    add(
        "nullability parquet NOT NULL, FAILFAST irrelevant".into(),
        read("parquet", "pq", "a INT NOT NULL"),
    );
    let ab = "a INT, b STRING";
    let abc = "a INT, b STRING, _corrupt_record STRING";
    let json_cases: [JsonCase; 17] = [
        ("json missing and extra fields", "js", ab, &[]),
        ("json wrong types PERMISSIVE", "jbad", ab, &[]),
        (
            "json wrong types PERMISSIVE with _corrupt_record",
            "jbad",
            abc,
            &[],
        ),
        (
            "json wrong types DROPMALFORMED",
            "jbad",
            ab,
            &[("mode", "DROPMALFORMED")],
        ),
        (
            "json wrong types FAILFAST",
            "jbad",
            ab,
            &[("mode", "FAILFAST")],
        ),
        ("json string for int PERMISSIVE", "jbad1", ab, &[]),
        (
            "json string for int with _corrupt_record",
            "jbad1",
            abc,
            &[],
        ),
        (
            "json string for int FAILFAST",
            "jbad1",
            ab,
            &[("mode", "FAILFAST")],
        ),
        ("json double for int PERMISSIVE", "jbad2", ab, &[]),
        (
            "json double for int with _corrupt_record",
            "jbad2",
            abc,
            &[],
        ),
        (
            "json double for int FAILFAST",
            "jbad2",
            ab,
            &[("mode", "FAILFAST")],
        ),
        ("json object for string PERMISSIVE", "jbad3", ab, &[]),
        (
            "json object for string FAILFAST",
            "jbad3",
            ab,
            &[("mode", "FAILFAST")],
        ),
        ("json malformed line PERMISSIVE", "jmalformed", ab, &[]),
        (
            "json malformed line with _corrupt_record",
            "jmalformed",
            abc,
            &[],
        ),
        (
            "json malformed line FAILFAST",
            "jmalformed",
            ab,
            &[("mode", "FAILFAST")],
        ),
        (
            "json nested struct field absent from data",
            "jnest",
            "a INT, s STRUCT<p: INT, q: STRING, r: INT>",
            &[],
        ),
    ];
    for (name, path, schema, opts) in json_cases {
        add(name.to_owned(), with(read("json", path, schema), opts));
    }
    add(
        "json schema column absent from all data".into(),
        read("json", "js", "a INT, zz STRING"),
    );
    add(
        "json name case differs, default caseSensitive".into(),
        read("json", "jcase", "A INT, B STRING"),
    );
    add(
        "json name case differs, caseSensitive=true".into(),
        read("json", "jcase", "A INT, B STRING"),
    );
    for (name, schema) in [
        ("parquet subset", "c BIGINT"),
        ("parquet reordered", "c BIGINT, a INT"),
        ("parquet absent column", "a INT, zz STRING"),
        (
            "parquet absent column NOT NULL",
            "a INT, zz STRING NOT NULL",
        ),
        ("parquet int read as string", "a STRING"),
        ("parquet int read as bigint", "a BIGINT"),
        ("parquet bigint read as int", "c INT"),
        ("parquet int read as decimal", "a DECIMAL(10,2)"),
        ("parquet string read as int", "b INT"),
        ("parquet int read as double", "a DOUBLE"),
    ] {
        add(name.into(), read("parquet", "pq", schema));
    }
    add(
        "parquet name case differs, default".into(),
        read("parquet", "pqcase", "A INT, b STRING"),
    );
    add(
        "parquet name case differs, caseSensitive=true".into(),
        read("parquet", "pqcase", "A INT, b STRING"),
    );
    for (fmt, tree, base) in [
        ("parquet", "pp", "v STRING"),
        ("csv", "pc", "v STRING, w STRING"),
    ] {
        let p = format!("{tree}/p");
        let w = if fmt == "csv" { ", w STRING" } else { "" };
        add(
            format!("partition {fmt} schema without k"),
            read(fmt, &p, base),
        );
        add(format!("partition {fmt} no schema"), no_schema(fmt, &p));
        add(
            format!("partition {fmt} k last"),
            read(fmt, &p, &format!("{base}, k STRING")),
        );
        add(
            format!("partition {fmt} k first"),
            read(fmt, &p, &format!("k STRING, {base}")),
        );
        add(
            format!("partition {fmt} k in the middle"),
            read(fmt, &p, &format!("v STRING, k STRING{w}")),
        );
        add(
            format!("partition {fmt} k int"),
            read(fmt, &p, &format!("{base}, k INT")),
        );
        add(
            format!("partition {fmt} k int FAILFAST"),
            with(
                read(fmt, &p, &format!("{base}, k INT")),
                &[("mode", "FAILFAST")],
            ),
        );
        add(
            format!("partition {fmt} k bigint NOT NULL"),
            read(fmt, &p, &format!("{base}, k BIGINT NOT NULL")),
        );
        add(format!("partition {fmt} only k"), read(fmt, &p, "k STRING"));
        add(format!("partition {fmt} basePath"), {
            let r = read(fmt, &format!("{tree}/p/k=1"), base);
            base_path(r, &p)
        });
        add(format!("partition {fmt} basePath, k in schema"), {
            let r = read(fmt, &format!("{tree}/p/k=1"), &format!("{base}, k STRING"));
            base_path(r, &p)
        });
        add(
            format!("partition {fmt} sub-directory without basePath"),
            read(fmt, &format!("{tree}/p/k=1"), base),
        );
    }
    for (fmt, tree, base) in [
        ("parquet", "pp1", "v STRING"),
        ("csv", "pc1", "v STRING, w STRING"),
    ] {
        let p = format!("{tree}/p");
        add(
            format!("partition {fmt} values 1,2 no schema"),
            no_schema(fmt, &p),
        );
        add(
            format!("partition {fmt} values 1,2 schema without k"),
            read(fmt, &p, base),
        );
        for (label, schema) in [
            ("k int last", format!("{base}, k INT")),
            ("k int first", format!("k INT, {base}")),
            ("k bigint NOT NULL", format!("{base}, k BIGINT NOT NULL")),
            ("k string NOT NULL", format!("{base}, k STRING NOT NULL")),
            ("k double", format!("{base}, k DOUBLE")),
        ] {
            add(
                format!("partition {fmt} values 1,2 {label}"),
                read(fmt, &p, &schema),
            );
        }
    }
    for (label, schema) in [
        ("schema without k", "v STRING"),
        ("k STRING", "v STRING, k STRING"),
        ("k INT", "v STRING, k INT"),
    ] {
        add(
            format!("partition parquet k also in data files, {label}"),
            read("parquet", "ppk/p", schema),
        );
    }
    add(
        "partition parquet k also in data files, no schema".into(),
        no_schema("parquet", "ppk/p"),
    );
    add(
        "text one column renamed".into(),
        read("text", "txt", "line STRING"),
    );
    add("text default schema".into(), no_schema("text", "txt"));
    add(
        "text two columns".into(),
        read("text", "txt", "a STRING, b STRING"),
    );
    add("text int column".into(), read("text", "txt", "line INT"));
    add(
        "text one column NOT NULL".into(),
        read("text", "txt", "line STRING NOT NULL"),
    );
    let views: [(&str, &str, &str, &str, &str); 6] = [
        ("sql view json", "v1", ab, "json", "jbad"),
        ("sql view json NOT NULL", "v1b", nn, "json", "js"),
        (
            "sql view parquet",
            "v2",
            "c BIGINT, zz STRING",
            "parquet",
            "pq",
        ),
        (
            "sql view parquet partitioned without k",
            "v3",
            "v STRING",
            "parquet",
            "pp/p",
        ),
        (
            "sql view parquet partitioned with k",
            "v4",
            "k STRING, v STRING",
            "parquet",
            "pp/p",
        ),
        (
            "sql view csv partitioned with k int",
            "v5",
            "v STRING, w STRING, k INT",
            "csv",
            "pc/p",
        ),
    ];
    for (case, name, columns, format, path) in views {
        add(
            case.to_owned(),
            Read::View {
                name,
                columns,
                format,
                path,
            },
        );
    }
    out
}

/// `r` with `basePath` set to `base` (made absolute when the case runs).
fn base_path(r: Read, base: &str) -> Read {
    match r {
        Read::Files {
            format,
            path,
            schema,
            built,
            mut options,
        } => {
            options.push(("basePath", leak(base)));
            Read::Files {
                format,
                path,
                schema,
                built,
                options,
            }
        }
        view @ Read::View { .. } => view,
    }
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

/// A struct type built in code with the same columns as `ddl` (simple `name TYPE [NOT NULL]` lists only).
fn built_schema(ddl: &str) -> StructType {
    StructType::new(ddl.split(", ").map(|c| {
        let mut parts = c.split(' ');
        let name = parts.next().unwrap();
        let ty = if parts.next().unwrap() == "INT" {
            DataType::Integer
        } else {
            DataType::String
        };
        StructField::new(name, ty, !c.ends_with("NOT NULL"))
    }))
}

/// The frame a case reads.
fn frame(s: &Session, root: &Path, r: &Read) -> Result<DataFrame, KrishivError> {
    let abs = |p: &str| root.join(p).to_str().unwrap().to_owned();
    match r {
        Read::Files {
            format,
            path,
            schema,
            built,
            options,
        } => {
            let mut reader = s.read();
            match schema {
                Some(ddl) if *built => reader = reader.schema(built_schema(ddl)),
                Some(ddl) => reader = reader.schema_ddl(ddl)?,
                None => {}
            }
            for (k, v) in options {
                let v = if *k == "basePath" {
                    abs(v)
                } else {
                    (*v).to_owned()
                };
                reader = reader.option(k, &v);
            }
            Ok(reader.format(format).load([abs(path)]))
        }
        Read::View {
            name,
            columns,
            format,
            path,
        } => {
            block_on(s.sql(&format!("DROP VIEW IF EXISTS {name}")))?;
            block_on(s.sql(&format!(
                "CREATE TEMPORARY VIEW {name} ({columns}) USING {format} OPTIONS (path '{}')",
                abs(path)
            )))?;
            block_on(s.sql(&format!("SELECT * FROM {name}")))
        }
    }
}

/// The outcome of a case in the fixture's shape.
fn outcome(s: &Session, root: &Path, r: &Read) -> Value {
    let result = (|| -> Result<Value, KrishivError> {
        let df = frame(s, root, r)?;
        let arrow = block_on(df.schema())?;
        let mut schema = Vec::new();
        for f in arrow.fields() {
            let t = DataType::from_arrow_field(f)?;
            schema.push(
                json!({"name": f.name(), "type": t.simple_string(), "nullable": f.is_nullable()}),
            );
        }
        let names: Vec<String> = (0..arrow.fields().len()).map(|i| format!("c{i}")).collect();
        let cast = df
            .to_df(&names)
            .select(names.iter().map(|n| col(n).cast(DataType::String)));
        let mut rows: Vec<Vec<Option<String>>> = Vec::new();
        for b in block_on(cast.collect())? {
            for i in 0..b.num_rows() {
                rows.push(
                    b.columns()
                        .iter()
                        .map(|c| {
                            (!c.is_null(i)).then(|| match c.data_type() {
                                arrow_schema::DataType::Utf8View => {
                                    c.as_string_view().value(i).to_owned()
                                }
                                arrow_schema::DataType::LargeUtf8 => {
                                    c.as_string::<i64>().value(i).to_owned()
                                }
                                _ => c.as_string::<i32>().value(i).to_owned(),
                            })
                        })
                        .collect(),
                );
            }
        }
        rows.sort_by_key(|r| {
            r.iter()
                .map(|v| v.as_ref().map_or((0, String::new()), |v| (1, v.clone())))
                .collect::<Vec<_>>()
        });
        let root = root.to_str().unwrap();
        let rows: Vec<Value> = rows
            .into_iter()
            .map(|r| {
                Value::from(
                    r.into_iter()
                        .map(|v| v.map_or(Value::Null, |v| Value::from(v.replace(root, "<root>"))))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        Ok(json!({"schema": schema, "rows": rows}))
    })();
    match result {
        Ok(v) => v,
        Err(e) => {
            let params: BTreeMap<&str, String> = e
                .message_parameters()
                .iter()
                .map(|(k, v)| (*k, v.replace(root.to_str().unwrap(), "<root>")))
                .collect();
            let chain: Vec<String> = std::iter::once(&e)
                .chain(e.causes())
                .map(|c| c.error_class().name().to_owned())
                .collect();
            json!({"error_class": e.error_class().name(), "message_parameters": params, "error_chain": chain, "message": e.to_string()})
        }
    }
}

/// Whether `got` matches the fixture case `want`: schema and rows, or the error at the level the fixture recorded.
fn matches(want: &Value, got: &Value) -> bool {
    if let Some(class) = want.get("error_class") {
        if got.get("error_class") != Some(class) {
            return false;
        }
        if let Some(chain) = want.get("error_chain") {
            return got.get("error_chain") == Some(chain);
        }
        return got.get("message_parameters") == want.get("message_parameters");
    }
    got.get("schema") == want.get("schema") && got.get("rows") == want.get("rows")
}

#[test]
fn reads_with_a_user_schema_match_spark() {
    let fixture = fixture();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let s = Session::new().unwrap();
    s.conf().set("spark.sql.session.timeZone", "UTC").unwrap();
    // The tree is laid out as another engine would: a commit log would hide nothing here, but keeps directories
    // the generator does not have.
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "NONE")
        .unwrap();
    layout(&s, root);
    let cases = cases();
    let want = fixture["cases"].as_array().unwrap();
    assert_eq!(
        cases.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        want.iter()
            .map(|c| c["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        "the cases follow the generator"
    );
    let mut failures = Vec::new();
    let mut gaps_matched = Vec::new();
    for ((name, r), want) in cases.iter().zip(want) {
        let conf = want.get("conf").and_then(Value::as_object);
        for (k, v) in conf.into_iter().flatten() {
            s.conf().set(k, v.as_str().unwrap()).unwrap();
        }
        let got = outcome(&s, root, r);
        for (k, _) in conf.into_iter().flatten() {
            s.conf().unset(k).unwrap();
        }
        let ok = matches(want, &got);
        let gap = KNOWN_GAPS.iter().any(|(n, _)| n == name);
        match (ok, gap) {
            (false, false) => failures.push(format!("{name}\n  want {want}\n  got  {got}")),
            (true, true) => gaps_matched.push(name.clone()),
            _ => {}
        }
    }
    assert!(
        gaps_matched.is_empty(),
        "now matching Spark, remove from KNOWN_GAPS: {gaps_matched:?}"
    );
    assert!(
        failures.is_empty(),
        "{} of {} cases differ from Spark:\n{}",
        failures.len(),
        want.len(),
        failures.join("\n")
    );
}
