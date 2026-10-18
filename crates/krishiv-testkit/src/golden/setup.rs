// SPDX-License-Identifier: Apache-2.0
//! Per-file session setup, read from the `.sql` input the way `SQLQueryTestSuite` does.
//!
//! * Folder defaults ([`TestKind`]): ANSI mode is on unless the file is under `nonansi/`;
//!   `postgreSQL/` also enables legacy intervals; `timestampNTZ/` selects `TIMESTAMP_NTZ`;
//!   `udf/`, `udaf/` and `udtf/` are markers for the engine's test UDF support.
//! * `--SET k=v[,k=v]` lines apply to every run of the file.
//! * `--CONFIG_DIM<n> k=v[,k=v]` lines: each line is one value of dimension `n`; the file runs
//!   once per element of the cartesian product of the dimensions. A statement counts as passed
//!   only if it passes in every run (all runs share one golden file), and counts once in the
//!   denominators.
//! * Fixture tables the file references are declared in [`SessionSetup::fixtures`]. The table
//!   data is not vendored yet; the declaration lets an engine load it (or skip the file).

use std::collections::BTreeMap;

use super::cell::SUITE_TIME_ZONE;

/// Folder-derived kind of a golden file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TestKind {
    /// Any other file: ANSI on.
    Regular,
    /// `nonansi/`: ANSI off.
    NonAnsi,
    /// `postgreSQL/`: ANSI on, legacy intervals, PostgreSQL test UDFs.
    PgSql,
    /// `timestampNTZ/`: `TIMESTAMP_NTZ` default timestamp type.
    TimestampNtz,
    /// `udf/postgreSQL/`: test UDF wrapper plus PostgreSQL setup.
    UdfPgSql,
    /// `udf/`: test UDF wrapper.
    Udf,
    /// `udaf/`.
    Udaf,
    /// `udtf/`.
    Udtf,
}

impl TestKind {
    /// Kind of the golden file `name` (relative to `inputs/`, `/`-separated). `udf/postgreSQL`
    /// is checked before `udf`, as in Spark.
    #[must_use]
    pub fn of(name: &str) -> Self {
        if name.starts_with("udf/postgreSQL") {
            Self::UdfPgSql
        } else if name.starts_with("udf/") {
            Self::Udf
        } else if name.starts_with("udaf/") {
            Self::Udaf
        } else if name.starts_with("udtf/") {
            Self::Udtf
        } else if name.starts_with("postgreSQL/") {
            Self::PgSql
        } else if name.starts_with("nonansi/") {
            Self::NonAnsi
        } else if name.starts_with("timestampNTZ/") {
            Self::TimestampNtz
        } else {
            Self::Regular
        }
    }

    /// Session configs the kind implies: the suite's session time zone
    /// ([`SUITE_TIME_ZONE`]), which the formatter of [`super::cell`] applies to timestamps, and
    /// the kind's own.
    #[must_use]
    pub fn configs(self) -> Vec<(String, String)> {
        let zone = (
            "spark.sql.session.timeZone".to_owned(),
            SUITE_TIME_ZONE.to_owned(),
        );
        let ansi = |v: &str| ("spark.sql.ansi.enabled".to_owned(), v.to_owned());
        let own = match self {
            Self::NonAnsi => vec![ansi("false")],
            Self::PgSql | Self::UdfPgSql => vec![
                ansi("true"),
                (
                    "spark.sql.legacy.interval.enabled".to_owned(),
                    "true".to_owned(),
                ),
            ],
            // Spark sets only the timestamp type here and leaves ANSI at its default.
            Self::TimestampNtz => vec![(
                "spark.sql.timestampType".to_owned(),
                "TIMESTAMP_NTZ".to_owned(),
            )],
            Self::Regular | Self::Udf | Self::Udaf | Self::Udtf => vec![ansi("true")],
        };
        std::iter::once(zone).chain(own).collect()
    }

    /// Test functions Spark registers in the session: the identity-cast `udf` (`udaf` for
    /// grouped-aggregate UDFs) for UDF folders, and `booleq`, `boolne` and `vol` for PostgreSQL
    /// ones. `udtf/` registers Python UDTFs, which have no entry here.
    #[must_use]
    pub fn udfs(self) -> Vec<String> {
        let names: &[&str] = match self {
            Self::Udf => &["udf"],
            Self::UdfPgSql => &["udf", "booleq", "boolne", "vol"],
            Self::PgSql => &["booleq", "boolne", "vol"],
            Self::Udaf => &["udaf"],
            _ => &[],
        };
        names.iter().map(|n| (*n).to_owned()).collect()
    }
}

/// A shared test table Spark creates before the run (from its `test-data/` files).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Fixture {
    /// `testdata`.
    TestData,
    /// `arraydata`.
    ArrayData,
    /// `mapdata`.
    MapData,
    /// `aggtest`.
    AggTest,
    /// `onek`.
    Onek,
    /// `tenk1`.
    Tenk1,
    /// The TPC-DS table schemas Spark creates for `pipe-operators` (never detected by name).
    TpcdsSchemas,
}

impl Fixture {
    /// Every fixture.
    pub const ALL: [Self; 6] = [
        Self::TestData,
        Self::ArrayData,
        Self::MapData,
        Self::AggTest,
        Self::Onek,
        Self::Tenk1,
    ];

    /// The table name.
    #[must_use]
    pub const fn table(self) -> &'static str {
        match self {
            Self::TestData => "testdata",
            Self::ArrayData => "arraydata",
            Self::MapData => "mapdata",
            Self::AggTest => "aggtest",
            Self::Onek => "onek",
            Self::Tenk1 => "tenk1",
            Self::TpcdsSchemas => "tpcds",
        }
    }
}

/// What an engine session must look like for one run of a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSetup {
    /// Folder-derived kind.
    pub kind: TestKind,
    /// Configs to apply in order (folder defaults, then `--SET`, then the `--CONFIG_DIM`
    /// combination); a later entry for a key overrides an earlier one.
    pub configs: Vec<(String, String)>,
    /// Test functions the session must register (see [`TestKind::udfs`]).
    pub udfs: Vec<String>,
    /// Fixture tables the file references (a textual over-approximation: a file that creates its
    /// own table of the same name is listed too).
    pub fixtures: Vec<Fixture>,
}

impl SessionSetup {
    /// The setup of a plain session of `kind` with no extra configs or fixtures.
    #[must_use]
    pub fn plain(kind: TestKind) -> Self {
        Self {
            kind,
            configs: kind.configs(),
            udfs: kind.udfs(),
            fixtures: Vec::new(),
        }
    }
}

/// The directives of one `.sql` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSetup {
    kind: TestKind,
    set: Vec<(String, String)>,
    /// One entry per dimension; each a list of alternatives.
    dims: Vec<Vec<Vec<(String, String)>>>,
    fixtures: Vec<Fixture>,
}

/// `k=v,k=v` as Spark splits it: on `,`, then on the first `=`, trimming both sides.
fn parse_pairs(s: &str) -> Vec<(String, String)> {
    s.split(',')
        .filter_map(|kv| kv.split_once('='))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        .collect()
}

impl FileSetup {
    /// Read the directives of the `.sql` text of golden file `name`.
    #[must_use]
    pub fn parse(name: &str, sql: &str) -> Self {
        let comments = sql.lines().map(str::trim).filter(|l| l.starts_with("--"));
        let mut set = Vec::new();
        let mut dims: BTreeMap<&str, Vec<Vec<(String, String)>>> = BTreeMap::new();
        for c in comments {
            if let Some(rest) = c.strip_prefix("--SET ") {
                set.extend(parse_pairs(rest));
            } else if let Some(rest) = c.strip_prefix("--CONFIG_DIM") {
                let (dim, value) = rest.split_once(' ').unwrap_or((rest, ""));
                dims.entry(dim).or_default().push(parse_pairs(value));
            }
        }
        let lower = sql.to_ascii_lowercase();
        let words: Vec<&str> = lower
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .collect();
        Self {
            kind: TestKind::of(name),
            set,
            dims: dims.into_values().collect(),
            fixtures: Fixture::ALL
                .into_iter()
                .filter(|f| words.contains(&f.table()))
                .chain((name == "pipe-operators").then_some(Fixture::TpcdsSchemas))
                .collect(),
        }
    }

    /// Every run of the file: one per `--CONFIG_DIM` combination, or exactly one.
    #[must_use]
    pub fn sessions(&self) -> Vec<SessionSetup> {
        let mut combos: Vec<Vec<(String, String)>> = vec![Vec::new()];
        for dim in &self.dims {
            combos = combos
                .iter()
                .flat_map(|c| {
                    dim.iter().map(move |alt| {
                        let mut next = c.clone();
                        next.extend(alt.iter().cloned());
                        next
                    })
                })
                .collect();
        }
        combos
            .into_iter()
            .map(|combo| {
                let mut configs = self.kind.configs();
                configs.extend(self.set.iter().cloned());
                configs.extend(combo);
                SessionSetup {
                    kind: self.kind,
                    configs,
                    udfs: self.kind.udfs(),
                    fixtures: self.fixtures.clone(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folder_kinds_and_defaults() {
        assert_eq!(TestKind::of("udf/postgreSQL/udf-case"), TestKind::UdfPgSql);
        assert_eq!(TestKind::of("udf/udf-union"), TestKind::Udf);
        assert_eq!(TestKind::of("nonansi/cast"), TestKind::NonAnsi);
        assert_eq!(TestKind::of("cast"), TestKind::Regular);
        let zone = (
            "spark.sql.session.timeZone".to_owned(),
            "America/Los_Angeles".to_owned(),
        );
        let off = SessionSetup::plain(TestKind::NonAnsi);
        assert_eq!(
            off.configs,
            vec![
                zone.clone(),
                ("spark.sql.ansi.enabled".into(), "false".into())
            ]
        );
        assert_eq!(SessionSetup::plain(TestKind::Regular).configs[1].1, "true");
        let ntz = SessionSetup::plain(TestKind::TimestampNtz).configs;
        assert_eq!(
            ntz,
            vec![
                zone,
                ("spark.sql.timestampType".into(), "TIMESTAMP_NTZ".into())
            ]
        );
        let pg = SessionSetup::plain(TestKind::PgSql);
        assert_eq!(pg.udfs, ["booleq", "boolne", "vol"]);
        assert!(
            pg.configs
                .contains(&("spark.sql.legacy.interval.enabled".into(), "true".into()))
        );
        assert_eq!(TestKind::UdfPgSql.udfs()[0], "udf");
    }

    #[test]
    fn set_dims_and_fixtures() {
        let sql = "--SET spark.a = 1, spark.b=2\n--CONFIG_DIM1 spark.x=true\n\
                   --CONFIG_DIM1 spark.x=false,spark.y=z\n--CONFIG_DIM2 spark.w=1\n\
                   --CONFIG_DIM2 spark.w=2\nSELECT * FROM TestData JOIN onek;\n";
        let setup = FileSetup::parse("group-by", sql);
        let sessions = setup.sessions();
        assert_eq!(sessions.len(), 4);
        let last = &sessions[3].configs;
        assert_eq!(last[2], ("spark.a".into(), "1".into()));
        assert_eq!(last[3], ("spark.b".into(), "2".into()));
        assert_eq!(
            &last[4..],
            &[
                ("spark.x".into(), "false".into()),
                ("spark.y".into(), "z".into()),
                ("spark.w".into(), "2".into())
            ]
        );
        assert_eq!(sessions[0].fixtures, vec![Fixture::TestData, Fixture::Onek]);
    }

    #[test]
    fn pipe_operators_needs_tpcds_schemas() {
        let setup = FileSetup::parse("pipe-operators", "select 1;");
        assert_eq!(setup.sessions()[0].fixtures, vec![Fixture::TpcdsSchemas]);
    }

    #[test]
    fn no_directives_is_one_session() {
        assert_eq!(FileSetup::parse("x", "select 1;\n").sessions().len(), 1);
    }
}
