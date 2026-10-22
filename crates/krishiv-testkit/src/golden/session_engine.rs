// SPDX-License-Identifier: Apache-2.0
//! The [`Engine`] over the real query pipeline: a `krishiv-session` [`Session`] per run.
//!
//! * [`Engine::reset`] builds a fresh session with the run's configs and registers the fixture
//!   tables the file declares ([`super::fixtures`]). A fixture whose column types the session
//!   cannot hold yet (arrays, maps) is not registered; the statements that read it then fail with
//!   the analyzer's `TABLE_OR_VIEW_NOT_FOUND`, which is what the engine would answer.
//! * [`Engine::parse`] runs `krishiv-sql`'s statement parser under the session's parser options.
//! * [`Engine::execute`] is `Session::sql` followed by `collect`. `SET key=value` and `RESET`
//!   update the session configuration first (so later statements run under the configs Spark
//!   used), then go to `Session::sql` like any statement: the session answers commands with its
//!   own error, and the statement is scored on that answer.
//!
//! The harness adds no SQL support of its own: everything the score measures is the session's
//! answer. Rows are rendered by [`super::cell`] from the batches and the result's Spark types;
//! row order counts when [`super::sorted::is_semantically_sorted`] says so on the analyzed plan.
//! Each statement runs under a wall-clock limit ([`SessionEngine::with_timeout`]): a statement
//! that exceeds it is cancelled (its future is dropped, which cancels the query) and fails with
//! [`TIMEOUT_CLASS`].

use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray as _;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType as ArrowType, Schema};
use krishiv_common::error::KrishivError;
use krishiv_exec::{Executor, ExecutorConfig, TokioExecutor};
use krishiv_session::Session;
use krishiv_sql::ParserOptions;
use krishiv_types::DataType;

use super::cell::{Cell, CellFormat};
use super::fixtures::{Table, fixtures_dir};
use super::run::{Engine, EngineError, QueryOutput};
use super::setup::SessionSetup;
use super::sorted::is_semantically_sorted;

/// The error class a statement that exceeds the time limit fails with (not a Spark class, so it
/// never matches an expected error).
pub const TIMEOUT_CLASS: &str = "KRISHIV_GOLDEN_TIMEOUT";

/// The default per-statement time limit.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// The memory limit of each session: far above what any golden statement needs, and below the
/// cap the harness runs under, so a runaway query fails inside the engine.
const SESSION_MEMORY_LIMIT: &str = "4g";

/// Runs golden statements through `krishiv-session`.
pub struct SessionEngine {
    executor: Arc<dyn Executor>,
    runtime: tokio::runtime::Runtime,
    timeout: Duration,
    session: Option<Session>,
    /// The run's configs, and the configs as `SET`/`RESET` statements left them (in order; a
    /// later entry for a key overrides an earlier one).
    base: Vec<(String, String)>,
    conf: Vec<(String, String)>,
}

impl std::fmt::Debug for SessionEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionEngine")
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

fn engine_error(e: &KrishivError) -> EngineError {
    EngineError {
        class: Some(e.error_class().name().to_owned()),
        sql_state: e.sql_state().map(str::to_owned),
        message: e.message().to_owned(),
    }
}

fn harness_error(message: impl Into<String>) -> EngineError {
    EngineError {
        class: None,
        sql_state: None,
        message: message.into(),
    }
}

impl SessionEngine {
    /// An engine whose sessions share one executor.
    ///
    /// # Errors
    /// When the executor or the harness's timer runtime cannot start.
    pub fn new() -> Result<Self, EngineError> {
        let executor: Arc<dyn Executor> =
            TokioExecutor::new(&ExecutorConfig::default()).map_err(|e| engine_error(&e))?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .map_err(|e| harness_error(format!("timer runtime: {e}")))?;
        Ok(Self {
            executor,
            runtime,
            timeout: DEFAULT_TIMEOUT,
            session: None,
            base: Vec::new(),
            conf: Vec::new(),
        })
    }

    /// The same engine with a per-statement time limit of `timeout`.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    fn conf_value(&self, key: &str) -> Option<&str> {
        self.conf
            .iter()
            .rev()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }

    fn options(&self) -> ParserOptions {
        ParserOptions::from_conf(|k| self.conf_value(k))
    }

    /// Mirrors a `SET key=value` or `RESET [key]` statement into the session configuration.
    fn apply_set(&mut self, sql: &str) {
        let sql = sql.trim().trim_end_matches(';').trim();
        let mut words = sql.split_whitespace();
        let first = words.next().unwrap_or_default();
        if first.eq_ignore_ascii_case("RESET") {
            let key = sql.get(5..).map(str::trim).filter(|k| !k.is_empty());
            match key {
                None => {
                    let old = std::mem::replace(&mut self.conf, self.base.clone());
                    if let Some(s) = &self.session {
                        let conf = s.conf();
                        // Every key the run set, including ones `base` does
                        // not hold: unsetting only `base` keys would leave a
                        // `SET` key behind in the session.
                        for (k, _) in &old {
                            drop(conf.unset(k));
                        }
                        for (k, v) in &self.base {
                            drop(conf.set(k, v));
                        }
                    }
                }
                Some(key) => {
                    let mut removed: Vec<String> = Vec::new();
                    self.conf.retain(|(k, _)| {
                        let hit = k.eq_ignore_ascii_case(key);
                        if hit {
                            removed.push(k.clone());
                        }
                        !hit
                    });
                    let base = self
                        .base
                        .iter()
                        .rev()
                        .find(|(k, _)| k.eq_ignore_ascii_case(key))
                        .cloned();
                    if let Some(kv) = &base {
                        self.conf.push(kv.clone());
                    }
                    if let Some(s) = &self.session {
                        let conf = s.conf();
                        // Unset under the stored spellings: the session keeps
                        // an unregistered key under the case it was set with.
                        for k in &removed {
                            drop(conf.unset(k));
                        }
                        match base {
                            Some((k, v)) => drop(conf.set(&k, &v)),
                            None => drop(conf.unset(key)),
                        }
                    }
                }
            }
            return;
        }
        if !first.eq_ignore_ascii_case("SET") {
            return;
        }
        if let Some((k, v)) = sql.get(3..).and_then(|rest| rest.split_once('=')) {
            let (k, v) = (k.trim().to_owned(), v.trim().to_owned());
            // A key Krishiv does not know is ignored by the session, as Spark keys are.
            if let Some(s) = &self.session {
                drop(s.conf().set(&k, &v));
            }
            self.conf.push((k, v));
        }
    }

    fn block_on<T>(
        &self,
        fut: impl Future<Output = Result<T, KrishivError>>,
    ) -> Result<T, EngineError> {
        match self
            .runtime
            .block_on(async { tokio::time::timeout(self.timeout, fut).await })
        {
            Ok(r) => r.map_err(|e| engine_error(&e)),
            Err(_) => Err(EngineError {
                class: Some(TIMEOUT_CLASS.to_owned()),
                sql_state: None,
                message: format!("no answer within {:?}", self.timeout),
            }),
        }
    }
}

impl Engine for SessionEngine {
    fn name(&self) -> &str {
        "krishiv-session"
    }

    fn reset(&mut self, setup: &SessionSetup) -> Result<(), EngineError> {
        self.session = None;
        let mut builder = Session::builder()
            .executor(Arc::clone(&self.executor))
            .config("krishiv.memory.limit", SESSION_MEMORY_LIMIT);
        for (k, v) in &setup.configs {
            builder = builder.config(k.clone(), v.clone());
        }
        // A config Krishiv rejects fails the run's reset: the session's answer.
        let session = builder.build().map_err(|e| engine_error(&e))?;
        let tables = setup
            .load_fixtures(&fixtures_dir())
            .map_err(|e| harness_error(e.to_string()))?;
        for table in &tables {
            // Not registering a table of types the session cannot hold is the engine's answer
            // (see the module docs); any other failure is a harness failure.
            if let Some((schema, batch)) = fixture_batch(table)? {
                session
                    .register_batches(&table.name, schema, vec![batch])
                    .map_err(|e| engine_error(&e))?;
            }
        }
        self.base.clone_from(&setup.configs);
        self.conf.clone_from(&setup.configs);
        self.session = Some(session);
        Ok(())
    }

    fn parse(&mut self, sql: &str) -> Result<(), EngineError> {
        krishiv_sql::parse_statement(sql, &self.options())
            .map(drop)
            .map_err(|e| engine_error(&e.error))
    }

    fn execute(&mut self, sql: &str) -> Result<QueryOutput, EngineError> {
        self.apply_set(sql);
        let session = self
            .session
            .clone()
            .ok_or_else(|| harness_error("no session"))?;
        let format = CellFormat::for_session(&self.conf).map_err(|e| harness_error(e.to_string()));
        let (analyzed, batches) = self.block_on(async {
            let df = session.sql(sql).await?;
            let analyzed = df.analyzed().await?;
            let batches = df.collect().await?;
            Ok((analyzed, batches))
        })?;
        let format = format?;
        let output = &analyzed.output;
        let schema = format!(
            "struct<{}>",
            output
                .iter()
                .map(|c| format!("{}:{}", c.name, c.data_type.catalog_string()))
                .collect::<Vec<_>>()
                .join(",")
        );
        let mut rows = Vec::new();
        for batch in &batches {
            // Each column's Spark type, from the batch's own field.
            let types = batch
                .schema()
                .fields()
                .iter()
                .map(|f| DataType::from_arrow_field(f).map_err(|e| engine_error(&e)))
                .collect::<Result<Vec<_>, _>>()?;
            for row in 0..batch.num_rows() {
                let cells = batch
                    .columns()
                    .iter()
                    .zip(&types)
                    .map(|(col, ty)| arrow_cell(col, row, ty))
                    .collect::<Result<Vec<_>, _>>()?;
                rows.push(format.render_row(&cells));
            }
        }
        Ok(QueryOutput {
            schema,
            rows,
            sorted: Some(is_semantically_sorted(analyzed.plan.root())),
        })
    }
}

/// The value at `row` of `col`, a column of Spark type `ty`.
fn arrow_cell(col: &ArrayRef, row: usize, ty: &DataType) -> Result<Cell, EngineError> {
    if col.is_null(row) {
        return Ok(Cell::Null);
    }
    let unexpected = || {
        harness_error(format!(
            "Arrow type {} for {}",
            col.data_type(),
            ty.catalog_string()
        ))
    };
    Ok(match (ty, col.data_type()) {
        (DataType::Null, _) => Cell::Null,
        (DataType::Boolean, ArrowType::Boolean) => Cell::Bool(col.as_boolean().value(row)),
        (DataType::Byte, ArrowType::Int8) => {
            Cell::Int(i64::from(col.as_primitive::<Int8Type>().value(row)))
        }
        (DataType::Short, ArrowType::Int16) => {
            Cell::Int(i64::from(col.as_primitive::<Int16Type>().value(row)))
        }
        (DataType::Integer, ArrowType::Int32) => {
            Cell::Int(i64::from(col.as_primitive::<Int32Type>().value(row)))
        }
        (DataType::Long, ArrowType::Int64) => Cell::Int(col.as_primitive::<Int64Type>().value(row)),
        (DataType::Float, ArrowType::Float32) => {
            Cell::Float(col.as_primitive::<Float32Type>().value(row))
        }
        (DataType::Double, ArrowType::Float64) => {
            Cell::Double(col.as_primitive::<Float64Type>().value(row))
        }
        (DataType::Decimal(d), ArrowType::Decimal128(..)) => Cell::Decimal(
            col.as_primitive::<Decimal128Type>().value(row),
            u32::from(d.scale()),
        ),
        (DataType::String, ArrowType::Utf8View) => {
            Cell::Str(col.as_string_view().value(row).to_owned())
        }
        (DataType::String, ArrowType::Utf8) => {
            Cell::Str(col.as_string::<i32>().value(row).to_owned())
        }
        (DataType::String, ArrowType::LargeUtf8) => {
            Cell::Str(col.as_string::<i64>().value(row).to_owned())
        }
        // A binary-safe STRING holding bytes that are not UTF-8: Spark's `UTF8String.toString`
        // replaces the malformed sequences.
        (DataType::String, _) => Cell::Str(
            String::from_utf8_lossy(&binary_value(col, row).ok_or_else(unexpected)?).into_owned(),
        ),
        (DataType::Binary, _) => Cell::Binary(binary_value(col, row).ok_or_else(unexpected)?),
        (DataType::Date, ArrowType::Date32) => {
            Cell::Date(col.as_primitive::<Date32Type>().value(row))
        }
        (DataType::Timestamp, ArrowType::Timestamp(..)) => {
            Cell::Timestamp(col.as_primitive::<TimestampMicrosecondType>().value(row))
        }
        (DataType::TimestampNtz, ArrowType::Timestamp(..)) => {
            Cell::TimestampNtz(col.as_primitive::<TimestampMicrosecondType>().value(row))
        }
        _ => return Err(unexpected()),
    })
}

fn binary_value(col: &ArrayRef, row: usize) -> Option<Vec<u8>> {
    Some(match col.data_type() {
        ArrowType::BinaryView => col.as_binary_view().value(row).to_vec(),
        ArrowType::Binary => col.as_binary::<i32>().value(row).to_vec(),
        ArrowType::LargeBinary => col.as_binary::<i64>().value(row).to_vec(),
        _ => return None,
    })
}

/// A fixture table as one batch, or `None` when a column type is one the session cannot hold.
fn fixture_batch(table: &Table) -> Result<Option<(Arc<Schema>, RecordBatch)>, EngineError> {
    let mut fields = Vec::with_capacity(table.columns.len());
    let mut types = Vec::with_capacity(table.columns.len());
    for c in &table.columns {
        let Ok(ty) = krishiv_types::parse_data_type(&c.spark_type) else {
            return Ok(None);
        };
        fields.push(ty.to_arrow_field(&c.name, c.nullable));
        types.push(ty);
    }
    let schema = Arc::new(Schema::new(fields));
    let columns = types
        .iter()
        .enumerate()
        .map(|(i, ty)| fixture_column(table, i, ty))
        .collect::<Result<Vec<_>, _>>()?;
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)
        .map_err(|e| harness_error(format!("fixture `{}`: {e}", table.name)))?;
    Ok(Some((schema, batch)))
}

fn fixture_column(table: &Table, i: usize, ty: &DataType) -> Result<ArrayRef, EngineError> {
    use arrow_array::builder::{
        BinaryViewBuilder, BooleanBuilder, Date32Builder, Decimal128Builder, Float32Builder,
        Float64Builder, Int8Builder, Int16Builder, Int32Builder, Int64Builder, StringViewBuilder,
        TimestampMicrosecondBuilder,
    };
    let cells = table.rows.iter().map(|r| r.get(i).unwrap_or(&Cell::Null));
    let bad = |c: &Cell| {
        harness_error(format!(
            "fixture `{}` column {i}: {c:?} for {}",
            table.name,
            ty.catalog_string()
        ))
    };
    macro_rules! ints {
        ($builder:ty, $t:ty) => {{
            let mut b = <$builder>::new();
            for c in cells {
                match c {
                    Cell::Null => b.append_null(),
                    Cell::Int(v) => b.append_value(<$t>::try_from(*v).map_err(|_| bad(c))?),
                    other => return Err(bad(other)),
                }
            }
            Arc::new(b.finish()) as ArrayRef
        }};
    }
    Ok(match ty {
        DataType::Byte => ints!(Int8Builder, i8),
        DataType::Short => ints!(Int16Builder, i16),
        DataType::Integer => ints!(Int32Builder, i32),
        DataType::Long => ints!(Int64Builder, i64),
        DataType::Boolean => {
            let mut b = BooleanBuilder::new();
            for c in cells {
                match c {
                    Cell::Null => b.append_null(),
                    Cell::Bool(v) => b.append_value(*v),
                    other => return Err(bad(other)),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Float => {
            let mut b = Float32Builder::new();
            for c in cells {
                match c {
                    Cell::Null => b.append_null(),
                    Cell::Float(v) => b.append_value(*v),
                    other => return Err(bad(other)),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Double => {
            let mut b = Float64Builder::new();
            for c in cells {
                match c {
                    Cell::Null => b.append_null(),
                    Cell::Double(v) => b.append_value(*v),
                    other => return Err(bad(other)),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Decimal(d) => {
            let mut b = Decimal128Builder::new();
            for c in cells {
                match c {
                    Cell::Null => b.append_null(),
                    Cell::Decimal(v, _) => b.append_value(*v),
                    other => return Err(bad(other)),
                }
            }
            let ArrowType::Decimal128(p, s) = ty.to_arrow() else {
                return Err(bad(&Cell::Null));
            };
            let _ = d;
            Arc::new(
                b.finish()
                    .with_precision_and_scale(p, s)
                    .map_err(|e| harness_error(e.to_string()))?,
            )
        }
        DataType::String => {
            let mut b = StringViewBuilder::new();
            for c in cells {
                match c {
                    Cell::Null => b.append_null(),
                    Cell::Str(v) => b.append_value(v),
                    other => return Err(bad(other)),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Binary => {
            let mut b = BinaryViewBuilder::new();
            for c in cells {
                match c {
                    Cell::Null => b.append_null(),
                    Cell::Binary(v) => b.append_value(v),
                    other => return Err(bad(other)),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Date => {
            let mut b = Date32Builder::new();
            for c in cells {
                match c {
                    Cell::Null => b.append_null(),
                    Cell::Date(v) => b.append_value(*v),
                    other => return Err(bad(other)),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Timestamp | DataType::TimestampNtz => {
            let mut b = TimestampMicrosecondBuilder::new();
            for c in cells {
                match c {
                    Cell::Null => b.append_null(),
                    Cell::Timestamp(v) | Cell::TimestampNtz(v) => b.append_value(*v),
                    other => return Err(bad(other)),
                }
            }
            let arr = b.finish();
            match ty.to_arrow() {
                ArrowType::Timestamp(_, Some(tz)) => Arc::new(arr.with_timezone(tz)),
                _ => Arc::new(arr),
            }
        }
        DataType::Null => Arc::new(arrow_array::NullArray::new(table.rows.len())),
        other => {
            return Err(harness_error(format!(
                "fixture `{}`: type {}",
                table.name,
                other.catalog_string()
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden::setup::{Fixture, TestKind};

    type R = Result<(), Box<dyn std::error::Error>>;

    fn engine(fixtures: Vec<Fixture>) -> Result<SessionEngine, EngineError> {
        let mut e = SessionEngine::new()?;
        let mut setup = SessionSetup::plain(TestKind::of("t"));
        setup.fixtures = fixtures;
        e.reset(&setup)?;
        Ok(e)
    }

    #[expect(
        clippy::needless_pass_by_value,
        reason = "used as `map_err(err)`, which passes the error by value"
    )]
    fn err(e: EngineError) -> Box<dyn std::error::Error> {
        format!("{e:?}").into()
    }

    #[test]
    fn renders_rows_schema_and_sortedness() -> R {
        let mut e = engine(vec![]).map_err(err)?;
        let out = e
            .execute(
                "SELECT 1 AS i, CAST(2 AS TINYINT) AS b, CAST(3 AS SMALLINT) AS s, 4L AS l, \
                 CAST(1.5 AS FLOAT) AS f, 2.5D AS d, 1.50BD AS m, 'x' AS t, X'4142' AS y, \
                 DATE'2020-01-02' AS dt, CAST('2020-01-02 03:04:05' AS TIMESTAMP_NTZ) AS n, true AS z, \
                 CAST(NULL AS INT) AS u",
            )
            .map_err(err)?;
        assert_eq!(
            out.schema,
            "struct<i:int,b:tinyint,s:smallint,l:bigint,f:float,d:double,m:decimal(3,2),t:string,\
             y:binary,dt:date,n:timestamp_ntz,z:boolean,u:int>"
        );
        assert_eq!(
            out.rows,
            [[
                "1",
                "2",
                "3",
                "4",
                "1.5",
                "2.5",
                "1.50",
                "x",
                "AB",
                "2020-01-02",
                "2020-01-02 03:04:05",
                "true",
                "NULL"
            ]]
        );
        assert_eq!(out.sorted, Some(false));
        let sorted = e.execute("SELECT 2 AS a ORDER BY a").map_err(err)?;
        assert_eq!(sorted.sorted, Some(true));
        Ok(())
    }

    #[test]
    fn errors_carry_the_class_and_parse_uses_the_session_options() -> R {
        let mut e = engine(vec![]).map_err(err)?;
        let got = e.execute("SELECT * FROM missing").map(drop).unwrap_err();
        assert_eq!(got.class.as_deref(), Some("TABLE_OR_VIEW_NOT_FOUND"));
        assert_eq!(got.sql_state.as_deref(), Some("42P01"));
        assert!(e.parse("SELECT 1").is_ok());
        let got = e.parse("XYZZY 1").unwrap_err();
        assert_eq!(got.class.as_deref(), Some("PARSE_SYNTAX_ERROR"));
        // `"a"` is a string, until double-quoted identifiers are on.
        assert!(e.parse("SELECT \"a\" FROM t").is_ok());
        e.apply_set("SET spark.sql.ansi.doubleQuotedIdentifiers = true;");
        assert_eq!(
            e.conf_value("SPARK.SQL.ANSI.DOUBLEQUOTEDIDENTIFIERS"),
            Some("true")
        );
        assert!(e.options().double_quoted_identifiers);
        e.apply_set("RESET spark.sql.ansi.doubleQuotedIdentifiers");
        assert!(!e.options().double_quoted_identifiers);
        Ok(())
    }

    #[test]
    fn set_and_reset_reach_the_session() -> R {
        let mut e = engine(vec![]).map_err(err)?;
        let ts = |e: &mut SessionEngine| -> Result<String, Box<dyn std::error::Error>> {
            let out = e
                .execute("SELECT CAST('2020-01-01 00:00:00' AS TIMESTAMP) AS t")
                .map_err(err)?;
            Ok(out.rows[0][0].clone())
        };
        e.apply_set("SET spark.sql.session.timeZone=UTC");
        // The command itself is the session's answer: it does not parse yet.
        let set = e
            .execute("SET spark.sql.session.timeZone=+01:00")
            .map(drop)
            .unwrap_err();
        assert_eq!(set.class.as_deref(), Some("PARSE_SYNTAX_ERROR"));
        assert_eq!(ts(&mut e)?, "2020-01-01 00:00:00");
        e.apply_set("RESET spark.sql.session.timeZone");
        e.apply_set("SET spark.sql.session.timeZone=UTC");
        e.apply_set("RESET");
        assert_eq!(e.conf, e.base);
        e.apply_set("SET spark.sql.session.timeZone");
        assert_eq!(e.conf, e.base);
        Ok(())
    }

    #[test]
    fn reset_clears_set_keys_and_ignores_case() -> R {
        let mut e = engine(vec![]).map_err(err)?;
        let session_get =
            |e: &SessionEngine, key: &str| e.session.as_ref().and_then(|s| s.conf().get(key));
        // A bare `RESET` unsets a `SET` key the run never had, in the
        // session as well as in `conf`.
        e.apply_set("SET spark.sql.ansi.enabled = false");
        assert_eq!(
            session_get(&e, "spark.sql.ansi.enabled").as_deref(),
            Some("false")
        );
        e.apply_set("RESET");
        assert_eq!(e.conf, e.base);
        assert_eq!(
            session_get(&e, "spark.sql.ansi.enabled").as_deref(),
            Some("true")
        );
        // `RESET key` matches case-insensitively, in both directions.
        e.apply_set("SET My.Custom.Key = 1");
        e.apply_set("RESET my.custom.key");
        assert!(
            e.conf
                .iter()
                .all(|(k, _)| !k.eq_ignore_ascii_case("My.Custom.Key"))
        );
        assert_eq!(session_get(&e, "My.Custom.Key"), None);
        assert_eq!(session_get(&e, "my.custom.key"), None);
        // ... and restores a base value it finds case-insensitively.
        e.base
            .push(("my.base.key".to_owned(), "base-val".to_owned()));
        e.conf
            .push(("my.base.key".to_owned(), "base-val".to_owned()));
        if let Some(s) = &e.session {
            drop(s.conf().set("my.base.key", "base-val"));
        }
        e.apply_set("SET my.base.key = changed");
        e.apply_set("RESET MY.BASE.KEY");
        assert_eq!(e.conf_value("my.base.key"), Some("base-val"));
        assert_eq!(session_get(&e, "my.base.key").as_deref(), Some("base-val"));
        Ok(())
    }

    #[test]
    fn fixtures_with_session_types_are_registered() -> R {
        let mut e = engine(vec![
            Fixture::TestData,
            Fixture::ArrayData,
            Fixture::AggTest,
        ])
        .map_err(err)?;
        let out = e
            .execute("SELECT count(*), max(key), min(value) FROM testdata")
            .map_err(err)?;
        assert_eq!(out.rows, [["100", "100", "1"]]);
        let out = e
            .execute("SELECT sum(a), max(b) FROM aggtest")
            .map_err(err)?;
        assert_eq!(out.rows, [["198", "324.78"]]);
        // `arraydata` has array columns, which the session cannot hold yet.
        let got = e.execute("SELECT * FROM arraydata").map(drop).unwrap_err();
        assert_eq!(got.class.as_deref(), Some("TABLE_OR_VIEW_NOT_FOUND"));
        Ok(())
    }

    #[test]
    fn a_statement_past_the_time_limit_fails_with_the_timeout_class() -> R {
        let e = SessionEngine::new()
            .map_err(err)?
            .with_timeout(Duration::ZERO);
        let got = e
            .block_on(std::future::pending::<Result<(), KrishivError>>())
            .unwrap_err();
        assert_eq!(got.class.as_deref(), Some(TIMEOUT_CLASS));
        assert!(format!("{e:?}").contains("timeout"));
        let mut idle = SessionEngine::new().map_err(err)?;
        assert_eq!(
            idle.execute("SELECT 1").map(drop).unwrap_err().message,
            "no session"
        );
        assert_eq!(idle.name(), "krishiv-session");
        Ok(())
    }

    #[test]
    fn cells_follow_the_spark_type_not_only_the_arrow_type() -> R {
        use arrow_array::{BinaryArray, Int32Array, LargeStringArray, StringArray};
        let s: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
        assert_eq!(
            arrow_cell(&s, 0, &DataType::String).map_err(err)?,
            Cell::Str("a".into())
        );
        let s: ArrayRef = Arc::new(LargeStringArray::from(vec!["b"]));
        assert_eq!(
            arrow_cell(&s, 0, &DataType::String).map_err(err)?,
            Cell::Str("b".into())
        );
        // A binary-safe STRING holding bytes that are not UTF-8.
        let b: ArrayRef = Arc::new(BinaryArray::from(vec![&b"a\xff"[..]]));
        assert_eq!(
            arrow_cell(&b, 0, &DataType::String).map_err(err)?,
            Cell::Str("a\u{fffd}".into())
        );
        assert_eq!(
            arrow_cell(&b, 0, &DataType::Binary).map_err(err)?,
            Cell::Binary(b"a\xff".to_vec())
        );
        let i: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        assert_eq!(arrow_cell(&i, 0, &DataType::Null).map_err(err)?, Cell::Null);
        let bad = arrow_cell(&i, 0, &DataType::String).unwrap_err();
        assert!(
            bad.message.contains("Arrow type Int32 for string"),
            "{}",
            bad.message
        );
        assert!(arrow_cell(&i, 0, &DataType::Binary).is_err());
        Ok(())
    }

    #[test]
    fn fixture_tables_of_every_session_type_round_trip() -> R {
        use crate::golden::fixtures::{Column, Table};
        let col = |name: &str, ty: &str| Column {
            name: name.into(),
            spark_type: ty.into(),
            nullable: true,
        };
        let table = Table {
            name: "every".into(),
            columns: vec![
                col("b", "tinyint"),
                col("s", "smallint"),
                col("i", "int"),
                col("l", "bigint"),
                col("z", "boolean"),
                col("f", "float"),
                col("d", "double"),
                col("m", "decimal(5,2)"),
                col("t", "string"),
                col("y", "binary"),
                col("dt", "date"),
                col("ts", "timestamp"),
                col("n", "timestamp_ntz"),
                col("v", "void"),
            ],
            rows: vec![
                vec![
                    Cell::Int(1),
                    Cell::Int(2),
                    Cell::Int(3),
                    Cell::Int(4),
                    Cell::Bool(true),
                    Cell::Float(1.5),
                    Cell::Double(2.5),
                    Cell::Decimal(150, 2),
                    Cell::Str("x".into()),
                    Cell::Binary(b"ab".to_vec()),
                    Cell::Date(1),
                    Cell::Timestamp(1_000_000),
                    Cell::TimestampNtz(2_000_000),
                    Cell::Null,
                ],
                vec![Cell::Null; 14],
            ],
        };
        let (schema, batch) = fixture_batch(&table)
            .map_err(err)?
            .ok_or("not registered")?;
        assert_eq!(batch.num_rows(), 2);
        let types = schema
            .fields()
            .iter()
            .map(|f| DataType::from_arrow_field(f))
            .collect::<Result<Vec<_>, _>>()?;
        let back = |row: usize| -> Result<Vec<Cell>, EngineError> {
            batch
                .columns()
                .iter()
                .zip(&types)
                .map(|(c, t)| arrow_cell(c, row, t))
                .collect()
        };
        assert_eq!(back(0).map_err(err)?, table.rows[0]);
        assert_eq!(back(1).map_err(err)?, table.rows[1]);
        // A cell of the wrong kind, or out of the column type's range, is a harness error.
        for (ty, cell) in [
            ("tinyint", Cell::Int(300)),
            ("int", Cell::Str("x".into())),
            ("boolean", Cell::Int(1)),
            ("float", Cell::Int(1)),
            ("double", Cell::Int(1)),
            ("decimal(5,2)", Cell::Int(1)),
            ("string", Cell::Int(1)),
            ("binary", Cell::Int(1)),
            ("date", Cell::Int(1)),
            ("timestamp", Cell::Int(1)),
        ] {
            let bad = Table {
                name: "bad".into(),
                columns: vec![col("c", ty)],
                rows: vec![vec![cell]],
            };
            let e = fixture_batch(&bad).map(drop).unwrap_err();
            assert!(
                e.message.contains("fixture `bad` column 0"),
                "{ty}: {}",
                e.message
            );
        }
        // Nested columns: the table is not registered.
        let nested = Table {
            name: "n".into(),
            columns: vec![col("a", "array<int>")],
            rows: vec![],
        };
        assert!(fixture_batch(&nested).map_err(err)?.is_none());
        Ok(())
    }
}
