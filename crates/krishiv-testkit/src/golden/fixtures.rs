// SPDX-License-Identifier: Apache-2.0
//! The shared test tables `SQLQueryTestSuite` creates before it runs (`testdata`, `arraydata`,
//! `mapdata`, `aggtest`, `onek`, `tenk1`) and the TPC-DS table schemas it creates for
//! `pipe-operators.sql`, loaded from `tests/fixtures/spark-4.2.0/golden-tables/`.
//!
//! The files are generated from a Spark 4.2.0 session by `tests/scripts/gen_golden_fixtures.py`,
//! so the schemas are the ones Spark reports (`catalogString` types, every column nullable) and
//! the rows are in the order Spark stores them, map entries included. Values decode to
//! [`Cell`]s; an engine turns them into its own batches. [`SessionSetup::load_fixtures`] resolves
//! the fixtures a run declares.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::cell::Cell;
use super::setup::{Fixture, SessionSetup};
use super::value::{SparkType, ValueError};

/// A column of a fixture table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    /// Column name.
    pub name: String,
    /// Spark's `catalogString` of the type, e.g. `array<int>` or `char(16)`.
    pub spark_type: String,
    /// Whether the column accepts NULL.
    pub nullable: bool,
}

/// A fixture table: its schema and rows.
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    /// Table name.
    pub name: String,
    /// Columns, in order.
    pub columns: Vec<Column>,
    /// Rows, one cell per column; empty for the TPC-DS schemas.
    pub rows: Vec<Vec<Cell>>,
}

/// A fixture that cannot be loaded.
#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    /// A file cannot be read.
    #[error("reading {path}: {source}")]
    Io {
        /// File path.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },
    /// A file is not JSON.
    #[error("{path}: {source}")]
    Json {
        /// File path.
        path: PathBuf,
        /// Underlying error.
        source: serde_json::Error,
    },
    /// A table is malformed.
    #[error("table {table}: {source}")]
    Table {
        /// Table name (or the file's, when it has none).
        table: String,
        /// The problem.
        source: ValueError,
    },
}

/// Directory of the table fixtures.
#[must_use]
pub fn fixtures_dir() -> PathBuf {
    crate::tests_dir().join("fixtures/spark-4.2.0/golden-tables")
}

fn bad(msg: impl Into<String>) -> ValueError {
    ValueError(msg.into())
}

impl Table {
    /// Decode one table object: `{"name", "columns": [{"name", "type", "nullable"}], "rows"}`.
    pub fn from_json(v: &Value) -> Result<Self, ValueError> {
        let name = v["name"]
            .as_str()
            .ok_or_else(|| bad("table without a name"))?;
        let in_table = |e: ValueError| bad(format!("{name}: {e}"));
        let columns = v["columns"]
            .as_array()
            .ok_or_else(|| bad(format!("{name}: no columns")))?
            .iter()
            .map(|c| {
                match (
                    c["name"].as_str(),
                    c["type"].as_str(),
                    c["nullable"].as_bool(),
                ) {
                    (Some(name), Some(ty), Some(nullable)) => Ok(Column {
                        name: name.to_owned(),
                        spark_type: ty.to_owned(),
                        nullable,
                    }),
                    _ => Err(bad(format!("{name}: malformed column {c}"))),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let types = columns
            .iter()
            .map(|c| SparkType::parse(&c.spark_type))
            .collect::<Result<Vec<_>, _>>()
            .map_err(in_table)?;
        let rows = v["rows"].as_array().map_or(&[][..], Vec::as_slice);
        let rows = rows
            .iter()
            .map(|row| {
                let cells = row
                    .as_array()
                    .ok_or_else(|| bad(format!("row {row} is not a list")))?;
                if cells.len() != types.len() {
                    return Err(bad(format!(
                        "row {row} has {} cells, not {}",
                        cells.len(),
                        types.len()
                    )));
                }
                cells
                    .iter()
                    .zip(&types)
                    .map(|(c, t)| Cell::from_json(c, t))
                    .collect()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(in_table)?;
        Ok(Self {
            name: name.to_owned(),
            columns,
            rows,
        })
    }
}

fn read_json(path: &Path) -> Result<Value, FixtureError> {
    let text = std::fs::read_to_string(path).map_err(|source| FixtureError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| FixtureError::Json {
        path: path.to_path_buf(),
        source,
    })
}

impl Fixture {
    /// The fixture file name under [`fixtures_dir`].
    #[must_use]
    pub const fn file(self) -> &'static str {
        match self {
            Self::TestData => "testdata.json",
            Self::ArrayData => "arraydata.json",
            Self::MapData => "mapdata.json",
            Self::AggTest => "aggtest.json",
            Self::Onek => "onek.json",
            Self::Tenk1 => "tenk1.json",
            Self::TpcdsSchemas => "tpcds-schemas.json",
        }
    }

    /// Load the tables of this fixture from `dir` (see [`fixtures_dir`]): one table, or the
    /// TPC-DS schemas in name order.
    pub fn load(self, dir: &Path) -> Result<Vec<Table>, FixtureError> {
        let doc = read_json(&dir.join(self.file()))?;
        let table = |v: &Value| {
            Table::from_json(v).map_err(|source| FixtureError::Table {
                table: self.table().to_owned(),
                source,
            })
        };
        match (self, doc["tables"].as_array()) {
            (Self::TpcdsSchemas, Some(tables)) => tables.iter().map(table).collect(),
            _ => table(&doc).map(|t| vec![t]),
        }
    }
}

impl SessionSetup {
    /// Load the tables of every fixture this run declares from `dir`, in declaration order.
    pub fn load_fixtures(&self, dir: &Path) -> Result<Vec<Table>, FixtureError> {
        let mut tables = Vec::new();
        for fixture in &self.fixtures {
            tables.extend(fixture.load(dir)?);
        }
        Ok(tables)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::golden::cell::{CellFormat, render_row};
    use crate::golden::parse::Expected;
    use crate::golden::run::load_statements;
    use crate::golden::setup::FileSetup;
    use crate::golden::{DEFAULT_TAG, GoldenSet};

    type R = Result<(), Box<dyn std::error::Error>>;

    fn load(f: Fixture) -> Result<Table, FixtureError> {
        let mut tables = f.load(&fixtures_dir())?;
        assert_eq!(tables.len(), 1);
        Ok(tables.remove(0))
    }

    fn ints(t: &Table, col: usize) -> Vec<i64> {
        t.rows
            .iter()
            .filter_map(|r| match r[col] {
                Cell::Int(i) => Some(i),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn every_declared_fixture_loads_with_spark_shapes() -> R {
        let testdata = load(Fixture::TestData)?;
        assert_eq!(testdata.name, "testdata");
        assert_eq!(testdata.rows.len(), 100);
        assert_eq!(
            testdata.columns,
            [
                Column {
                    name: "key".into(),
                    spark_type: "int".into(),
                    nullable: true
                },
                Column {
                    name: "value".into(),
                    spark_type: "string".into(),
                    nullable: true
                },
            ]
        );
        assert_eq!(testdata.rows[99], [Cell::Int(100), Cell::Str("100".into())]);

        let arrays = load(Fixture::ArrayData)?;
        assert_eq!(arrays.columns[1].spark_type, "array<array<int>>");
        assert_eq!(arrays.rows.len(), 2);

        let maps = load(Fixture::MapData)?;
        assert_eq!(maps.columns[0].spark_type, "map<int,string>");
        assert_eq!(maps.rows.len(), 5);
        // Spark stores the five-entry map with key 5 first; the loader keeps storage order.
        assert!(matches!(&maps.rows[0][0], Cell::Map(m) if m.len() == 5 && m[0].0 == Cell::Int(5)));

        let agg = load(Fixture::AggTest)?;
        assert_eq!(agg.rows[0], [Cell::Int(56), Cell::Float(7.8)]);
        assert_eq!(agg.rows.len(), 4);

        let onek = load(Fixture::Onek)?;
        let tenk = load(Fixture::Tenk1)?;
        assert_eq!((onek.rows.len(), tenk.rows.len()), (1000, 10000));
        assert_eq!(onek.columns.len(), 16);
        assert_eq!(tenk.columns[13].name, "stringu1");
        assert_eq!(tenk.columns[13].spark_type, "string");
        let mut unique1 = ints(&tenk, 0);
        unique1.sort_unstable();
        assert!(
            unique1.iter().copied().eq(0..10000),
            "unique1 is a permutation"
        );

        let tpcds = Fixture::TpcdsSchemas.load(&fixtures_dir())?;
        assert_eq!(tpcds.len(), 24);
        assert!(tpcds.iter().all(|t| t.rows.is_empty()));
        assert!(tpcds.windows(2).all(|w| w[0].name < w[1].name));
        let sales = tpcds
            .iter()
            .find(|t| t.name == "store_sales")
            .ok_or("store_sales")?;
        let cost = sales
            .columns
            .iter()
            .find(|c| c.name == "ss_wholesale_cost")
            .ok_or("cost")?;
        assert_eq!(cost.spark_type, "decimal(7,2)");
        let customer = tpcds
            .iter()
            .find(|t| t.name == "customer")
            .ok_or("customer")?;
        assert!(customer.columns.iter().any(|c| c.spark_type == "char(16)"));
        Ok(())
    }

    /// Every fixture that any vendored golden declares loads, and `pipe-operators` brings the
    /// TPC-DS schemas.
    #[test]
    fn fixtures_declared_by_the_vendored_goldens_all_load() -> R {
        let set = GoldenSet::scan(&GoldenSet::tag_dir(DEFAULT_TAG))?;
        let mut declared = std::collections::BTreeSet::new();
        for pair in &set.pairs {
            let sql = std::fs::read_to_string(&pair.sql)?;
            for session in FileSetup::parse(&pair.name, &sql).sessions() {
                declared.insert(session.fixtures);
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for fixtures in declared {
            let mut setup = SessionSetup::plain(crate::golden::setup::TestKind::Regular);
            setup.fixtures = fixtures;
            let tables = setup.load_fixtures(&fixtures_dir())?;
            assert!(tables.len() >= setup.fixtures.len());
            seen.extend(setup.fixtures);
        }
        let all: Vec<Fixture> = Fixture::ALL
            .into_iter()
            .chain([Fixture::TpcdsSchemas])
            .collect();
        assert_eq!(seen.into_iter().collect::<Vec<_>>(), all);
        Ok(())
    }

    fn golden_output(file: &str, query: &str) -> Result<String, Box<dyn std::error::Error>> {
        let set = GoldenSet::scan(&GoldenSet::tag_dir(DEFAULT_TAG))?;
        let pair = set
            .pairs
            .iter()
            .find(|p| p.name == file)
            .ok_or("golden file")?;
        let statement = load_statements(pair)?
            .into_iter()
            .find(|s| s.query == query)
            .ok_or("query")?;
        match statement.expected {
            Expected::Rows(out) => Ok(out),
            Expected::Error(_) => Err("expected an error".into()),
        }
    }

    /// The loaded rows, printed as `HiveResult` does, are what Spark recorded for
    /// `SELECT * FROM <table> LIMIT 2`.
    #[test]
    fn first_rows_print_as_the_recorded_goldens() -> R {
        let format = CellFormat::default();
        for (fixture, query) in [
            (Fixture::TestData, "SELECT * FROM testdata LIMIT 2"),
            (Fixture::ArrayData, "SELECT * FROM arraydata LIMIT 2"),
            (Fixture::MapData, "SELECT * FROM mapdata LIMIT 2"),
        ] {
            let table = load(fixture)?;
            let printed: Vec<String> = table.rows[..2]
                .iter()
                .map(|r| render_row(r, &format).join("\t"))
                .collect();
            assert_eq!(
                printed.join("\n"),
                golden_output("limit", query)?
                    .trim_end_matches(crate::golden::normalise::JAVA_SPACE),
                "{query}"
            );
        }
        Ok(())
    }

    /// The aggregates Spark recorded over the PostgreSQL tables hold for the loaded rows.
    #[test]
    fn aggregates_match_the_recorded_goldens() -> R {
        let onek = load(Fixture::Onek)?;
        let sum_four: i64 = ints(&onek, 3).iter().sum();
        let out = golden_output(
            "postgreSQL/aggregates_part1",
            "SELECT sum(four) AS sum_1500 FROM onek",
        )?;
        assert_eq!(out.trim(), sum_four.to_string());
        let out = golden_output(
            "postgreSQL/aggregates_part1",
            "SELECT count(four) AS cnt_1000 FROM onek",
        )?;
        assert_eq!(out.trim(), ints(&onek, 3).len().to_string());
        let agg = load(Fixture::AggTest)?;
        let out = golden_output(
            "udf/postgreSQL/udf-aggregates_part1",
            "SELECT udf(sum(a)) AS sum_198 FROM aggtest",
        )?;
        assert_eq!(out.trim(), ints(&agg, 0).iter().sum::<i64>().to_string());
        Ok(())
    }

    #[test]
    fn session_resolves_its_declared_fixtures_in_order() -> R {
        let mut setup = SessionSetup::plain(crate::golden::setup::TestKind::Regular);
        assert!(setup.load_fixtures(&fixtures_dir())?.is_empty());
        setup.fixtures = vec![Fixture::MapData, Fixture::TestData];
        let names: Vec<String> = setup
            .load_fixtures(&fixtures_dir())?
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(names, ["mapdata", "testdata"]);
        Ok(())
    }

    #[test]
    fn load_errors_name_the_problem() -> R {
        let dir = std::env::temp_dir().join(format!("krishiv-fixtures-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let write = |text: &str| std::fs::write(dir.join(Fixture::TestData.file()), text);
        let err = Fixture::TestData.load(&dir.join("missing"));
        assert!(matches!(err, Err(FixtureError::Io { .. })));
        write("{")?;
        assert!(matches!(
            Fixture::TestData.load(&dir),
            Err(FixtureError::Json { .. })
        ));
        write(
            r#"{"name":"t","columns":[{"name":"a","type":"int","nullable":true}],"rows":[["x"]]}"#,
        )?;
        let err = Fixture::TestData
            .load(&dir)
            .err()
            .ok_or("expected an error")?;
        assert!(err.to_string().contains("table testdata: t:"), "{err}");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn table_json_errors() {
        let bad_tables = [
            json!({"columns": []}),
            json!({"name": "t"}),
            json!({"name": "t", "columns": [{"name": "a"}]}),
            json!({"name": "t", "columns": [{"name": "a", "type": "blob", "nullable": true}]}),
            json!({"name": "t", "columns": [{"name": "a", "type": "int", "nullable": true}], "rows": [1]}),
            json!({"name": "t", "columns": [{"name": "a", "type": "int", "nullable": true}], "rows": [[1, 2]]}),
        ];
        for t in bad_tables {
            assert!(Table::from_json(&t).is_err(), "{t}");
        }
        let empty = json!({"name": "t", "columns": []});
        assert_eq!(Table::from_json(&empty).map(|t| t.rows.len()), Ok(0));
    }
}
