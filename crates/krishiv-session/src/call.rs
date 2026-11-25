// SPDX-License-Identifier: Apache-2.0
//! `CALL cat.system.p(args)` (design P3-T027 PR-36e §3): resolves the catalog, loads the
//! procedure, rearranges the arguments by name as Spark's `NamedParametersSupport` does, casts
//! each to its parameter's type (ANSI whatever `spark.sql.ansi.enabled` says, through the
//! statement-internal query path with an ANSI query environment, so a bad value fails instead of
//! becoming `NULL` and no extra query is recorded), runs the procedure and returns its rows.
//! Arguments Spark's binding cannot take are refused (`refuse_unbindable`), and so are `NULL`
//! elements of arrays and `NULL` map values. Not planned through the optimizer and never cached.

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use krishiv_common::error::{
    DuplicateRoutineParameterAssignmentBothPositionalAndNamed,
    DuplicateRoutineParameterAssignmentDoubleNamedArgumentReference, FailedToLoadRoutine,
    KrishivError, KrishivIcebergProcedureArgument, MissingCatalogAbilityProcedures,
    RequiredParameterNotFound, UnexpectedPositionalArgument, UnrecognizedParameterName,
    WrongNumArgsWithoutSuggestion,
};
use krishiv_plan::procedure::{
    ProcedureContext, ProcedureParameter, ProcedureType, ProcedureValue,
};
use krishiv_sql::ast::{BinaryOp, Call, DdlName, Expr, ExprKind};

use crate::dataframe::DataFrame;
use crate::session::Session;

/// Spark's documentation root, as the other arity errors name it.
const DOC_ROOT: &str = "https://spark.apache.org/docs/latest";

fn quote(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

fn quote_all(parts: &[String]) -> String {
    parts.iter().map(|p| quote(p)).collect::<Vec<_>>().join(".")
}

impl Session {
    /// Runs a `CALL`.
    pub(crate) async fn exec_call(
        &self,
        call: &Call,
        query: &str,
    ) -> Result<DataFrame, KrishivError> {
        let DdlName::Parts(name) = &call.name else {
            return Err(crate::session::unsupported_statement(
                "CALL with a dynamic procedure name",
            ));
        };
        let parts: Vec<String> = name.parts.iter().map(|p| p.value.clone()).collect();
        // `cat.system.p`, or `system.p` in the current catalog.
        let (catalog, namespace, procedure) = match parts.as_slice() {
            [c, n, p] => (c.clone(), n.clone(), p.clone()),
            [n, p] => (self.state.catalog.current_catalog(), n.clone(), p.clone()),
            _ => {
                return Err(FailedToLoadRoutine {
                    routine_name: quote_all(&parts),
                }
                .into());
            }
        };
        let view = self.declared_catalogs();
        let confines = view.confines();
        let Some(built) = view.built(&catalog) else {
            return Err(MissingCatalogAbilityProcedures {
                plugin: catalog.clone(),
            }
            .into());
        };
        let (built, caller) = built?;
        let shown = quote_all(&[catalog.clone(), namespace.clone(), procedure.clone()]);
        // One configuration snapshot for the whole call: the procedure's inputs and the name
        // binding below read the same one, whatever `conf().set` does meanwhile.
        let inputs = self.procedure_inputs();
        let key = "spark.sql.caseSensitive";
        let case_sensitive = krishiv_common::config::lookup(key)
            .and_then(|e| inputs.config.text(e))
            .or_else(|| inputs.config.get_unregistered(key).map(str::to_owned))
            .is_some_and(|v| v.eq_ignore_ascii_case("true"));
        // In server mode, every location the procedure reads or deletes is confined as a declared
        // table's scan is, before it is read.
        let inputs = if confines {
            crate::catalog::confined_inputs(
                inputs,
                &[catalog.clone(), namespace.clone(), procedure.clone()],
            )?
        } else {
            inputs
        };
        let loaded = if namespace.eq_ignore_ascii_case("system") {
            built
                .load_procedure(&caller, inputs, &procedure.to_lowercase())
                .await
                .map_err(|e| names_plugin(e, &catalog))?
        } else {
            None
        };
        let Some(loaded) = loaded else {
            return Err(FailedToLoadRoutine {
                routine_name: shown,
            }
            .into());
        };
        // Spark names the procedure alone in the binding errors (`procedure.name()`), and
        // compares names under the session resolver (`case_sensitive`, from the snapshot above).
        let slots = rearrange(
            &quote(loaded.name()),
            loaded.parameters(),
            &loaded.name_order(),
            call,
            case_sensitive,
        )?;
        self.refuse_unbindable(call, query).await?;
        let mut args = Vec::with_capacity(slots.len());
        for (param, slot) in loaded.parameters().iter().zip(slots) {
            args.push(match slot {
                None => ProcedureValue::Null,
                Some(i) => {
                    let arg = &call.args[i];
                    let text = query
                        .get(arg.expr.span.start..arg.expr.span.end)
                        .unwrap_or_default();
                    self.cast_argument(text, param.data_type).await?
                }
            });
        }
        let ctx = ProcedureContext {
            catalog: catalog.clone(),
        };
        let target = loaded
            .table(&args)
            .filter(|_| loaded.modifies_table())
            // An empty name is the procedure's own argument error (it opens nothing).
            .filter(|t| !t.is_empty())
            .map(|t| table_ident(&catalog, &t))
            .transpose()?;
        // `CALL` itself needs `MANAGE` on the cluster (checked before dispatch); a procedure that
        // changes a table also needs the write privilege on it, asked before it opens the table.
        if let Some(id) = &target {
            self.authorize_privilege(
                &krishiv_plan::analyzer::Securable::Table(id.clone()),
                krishiv_plan::analyzer::Privilege::Modify,
                krishiv_sql::StatementClass::Admin,
            )
            .await?;
        }
        let result = if loaded.name() == "rewrite_data_files"
            && let Some(id) = &target
        {
            let mut quoted = vec![id.catalog.clone()];
            quoted.extend(id.namespace.iter().cloned());
            quoted.push(id.name.clone());
            self.run_rewrite_data_files(&quote_all(&quoted), &rewrite_args(&args))
                .await
        } else {
            loaded.call(&ctx, args).await
        };
        if let Some(id) = target {
            // Frames and plans that read the table analyze again (`REFRESH TABLE`), whether the
            // procedure succeeded or failed after a commit. Only the caches are dropped: the next
            // read loads the table, so a failed load never turns a done expire into an error.
            built.invalidate(&caller, &id);
        }
        let batch = result?;
        self.script_result(batch.schema(), vec![batch]).await
    }

    /// The one row of `sql`, a `SELECT` this statement makes for itself: analyzed and run with
    /// ANSI on whatever `spark.sql.ansi.enabled` says (a non-ANSI cast turns a bad value into
    /// `NULL`, which a procedure reads as an absent argument: an expire would fall back to its
    /// default retention), and through the statement-internal path, so it is no query of its own
    /// (no `system.queries` row, no history entry).
    async fn internal_row(&self, sql: &str) -> Result<RecordBatch, KrishivError> {
        let mut env = self.query_env();
        let mut config = (*env.config).clone();
        config.set("spark.sql.ansi.enabled", "true")?;
        env.config = std::sync::Arc::new(config);
        env.analysis.conf.ansi = true;
        let analysis = self.analyze_variable_query_in(sql, env).await?;
        let batches = self.run_analyzed_variable_query(&analysis).await?;
        batches
            .into_iter()
            .find(|b| b.num_rows() > 0)
            .ok_or_else(|| crate::session::internal(format!("`{sql}` gave no row")))
    }

    /// `SELECT CAST((text) AS type)`: the value, as `CAST` gives it in ANSI mode.
    async fn cast_argument(
        &self,
        text: &str,
        ty: ProcedureType,
    ) -> Result<ProcedureValue, KrishivError> {
        let expr = match ty {
            ProcedureType::Timestamp => {
                format!("unix_micros(CAST(({text}) AS TIMESTAMP))")
            }
            other => format!("CAST(({text}) AS {})", other.sql()),
        };
        let batch = self.internal_row(&format!("SELECT {expr}")).await?;
        value_of(&batch, ty)
    }
}

/// A procedure's `table` argument (`ns.t`, or `cat.ns.t` naming the catalog the call runs in,
/// as the catalog reads it) as the table in `catalog`, parsed as a multipart identifier
/// (`` db.`t.v1` `` is two parts).
///
/// # Errors
/// `KRISHIV_ICEBERG_PROCEDURE_ARGUMENT` for a name that does not parse, as Iceberg's.
pub(crate) fn table_ident(
    catalog: &str,
    table: &str,
) -> Result<krishiv_plan::analyzer::TableIdent, KrishivError> {
    let mut parts: Vec<String> =
        krishiv_sql::parse_multipart_identifier(table, &krishiv_sql::ParserOptions::default())
            .map_err(|_| unparsable_table(table))?
            .parts
            .into_iter()
            .map(|p| p.value)
            .collect();
    if parts.len() > 2 && parts[0].eq_ignore_ascii_case(catalog) {
        parts.remove(0);
    }
    let name = parts.pop().unwrap_or_default();
    Ok(krishiv_plan::analyzer::TableIdent {
        catalog: catalog.to_owned(),
        namespace: parts,
        name,
    })
}

/// The arguments of `rewrite_data_files`, in its signature's order (`table`, `strategy`,
/// `sort_order`, `options`, `where`, `branch`).
fn rewrite_args(args: &[ProcedureValue]) -> crate::table_commands::RewriteArgs {
    let text = |i: usize| match args.get(i) {
        Some(ProcedureValue::String(s)) => Some(s.clone()),
        _ => None,
    };
    crate::table_commands::RewriteArgs {
        strategy: text(1),
        sort_order: text(2),
        options: match args.get(3) {
            Some(ProcedureValue::StringMap(m)) => m.iter().cloned().collect(),
            _ => std::collections::BTreeMap::new(),
        },
        filter: text(4),
        branch: text(5),
    }
}

/// A `NULL` inside an array argument: refused, never dropped (the procedure would act on
/// another list than the one written).
fn null_element() -> KrishivError {
    KrishivIcebergProcedureArgument {
        message: "An array argument must not contain NULL elements".to_owned(),
    }
    .into()
}

/// A `table` argument that is no multipart identifier: Iceberg 1.12.0's argument error
/// (`Spark3Util.catalogAndIdentifier`: `Cannot parse identifier for arg table: <name>`).
pub(crate) fn unparsable_table(table: &str) -> KrishivError {
    KrishivIcebergProcedureArgument {
        message: format!("Cannot parse identifier for arg table: {table}"),
    }
    .into()
}

/// A `NULL` value inside a map argument: refused, as an array's `NULL` element is.
fn null_map_value() -> KrishivError {
    KrishivIcebergProcedureArgument {
        message: "A map argument must not contain NULL values".to_owned(),
    }
    .into()
}

/// Row 0 of column 0 as a procedure value.
fn value_of(batch: &RecordBatch, ty: ProcedureType) -> Result<ProcedureValue, KrishivError> {
    use arrow_array::types::{Int32Type, Int64Type};
    let col = batch.column(0);
    if col.is_null(0) {
        return Ok(ProcedureValue::Null);
    }
    let bad = || crate::session::internal(format!("a procedure argument of type {ty:?}"));
    Ok(match ty {
        ProcedureType::String => ProcedureValue::String(
            col.as_string_opt::<i32>()
                .map(|a| a.value(0).to_owned())
                .or_else(|| col.as_string_view_opt().map(|a| a.value(0).to_owned()))
                .ok_or_else(bad)?,
        ),
        ProcedureType::Int => ProcedureValue::Long(i64::from(
            col.as_primitive_opt::<Int32Type>()
                .ok_or_else(bad)?
                .value(0),
        )),
        ProcedureType::Long => ProcedureValue::Long(
            col.as_primitive_opt::<Int64Type>()
                .ok_or_else(bad)?
                .value(0),
        ),
        ProcedureType::Timestamp => ProcedureValue::TimestampMicros(
            col.as_primitive_opt::<Int64Type>()
                .ok_or_else(bad)?
                .value(0),
        ),
        ProcedureType::Boolean => {
            ProcedureValue::Boolean(col.as_boolean_opt().ok_or_else(bad)?.value(0))
        }
        ProcedureType::StringMap => {
            let map = col.as_map_opt().ok_or_else(bad)?;
            let entries = map.value(0);
            let keys = entries.column(0);
            let values = entries.column(1);
            let text = |a: &arrow_array::ArrayRef, i: usize| -> Option<String> {
                a.as_string_opt::<i32>()
                    .map(|s| s.value(i).to_owned())
                    .or_else(|| a.as_string_view_opt().map(|s| s.value(i).to_owned()))
            };
            let mut out = Vec::with_capacity(entries.len());
            for i in 0..entries.len() {
                out.push((
                    text(keys, i).ok_or_else(bad)?,
                    if values.is_null(i) {
                        return Err(null_map_value());
                    } else {
                        text(values, i).ok_or_else(bad)?
                    },
                ));
            }
            ProcedureValue::StringMap(out)
        }
        ProcedureType::LongArray => {
            let list = col.as_list_opt::<i32>().ok_or_else(bad)?;
            let items = list.value(0);
            let longs = items.as_primitive_opt::<Int64Type>().ok_or_else(bad)?;
            ProcedureValue::LongArray(
                longs
                    .iter()
                    .collect::<Option<Vec<i64>>>()
                    .ok_or_else(null_element)?,
            )
        }
        ProcedureType::StringArray => {
            let list = col.as_list_opt::<i32>().ok_or_else(bad)?;
            let items = list.value(0);
            // The engine's strings are `Utf8` or `Utf8View`, whichever the array was built of.
            let strings: Option<Vec<String>> = if let Some(s) = items.as_string_opt::<i32>() {
                s.iter().map(|v| v.map(str::to_owned)).collect()
            } else if let Some(s) = items.as_string_view_opt() {
                s.iter().map(|v| v.map(str::to_owned)).collect()
            } else {
                return Err(bad());
            };
            ProcedureValue::StringArray(strings.ok_or_else(null_element)?)
        }
        _ => return Err(bad()),
    })
}

/// Spark's random-seed family (`ExpressionWithRandomSeed` and its aliases), whose calls
/// `ResolveRandomSeed` rewrites. Not every non-deterministic function: `spark_partition_id()` and
/// the like are not rewritten and are not refused.
const RANDOM_SEED_FUNCTIONS: [&str; 7] = [
    "rand", "randn", "random", "uuid", "shuffle", "uniform", "randstr",
];

/// A node of an argument whose operand types decide whether Spark's binding takes it.
enum Typed<'a> {
    /// `+` or `-` (`ResolveBinaryArithmetic`, `BinaryArithmeticWithDatetimeResolver`).
    AddOrSubtract(&'a Expr, &'a Expr),
    /// `/` (an implicit cast to `double` unless both sides are `double` or both `decimal`).
    Divide(&'a Expr, &'a Expr),
    /// `CAST(x AS <target>)`, the target as Spark's type name in lower case (the time zone
    /// `ResolveTimeZone` adds when `Cast.needsTimeZone(from, to)`).
    Cast(&'a Expr, String),
}

fn refuse_argument(what: &str) -> KrishivError {
    krishiv_common::error::KrishivUnsupportedFeature {
        feature: format!("{what} in a CALL argument"),
    }
    .into()
}

/// The kind of a type name for [`needs_time_zone`]: a `typeof` result or a cast's target as
/// written, in lower case.
#[derive(PartialEq, Eq, Clone, Copy)]
enum TzKind {
    String,
    Date,
    Timestamp,
    TimestampNtz,
    Variant,
    Other,
}

fn tz_kind(t: &str) -> TzKind {
    let t = t.trim();
    // `string collate UTF8_LCASE` is a string too (Spark's rows match `_: StringType`).
    if t.starts_with("string") || t.starts_with("varchar") || t.starts_with("char") {
        TzKind::String
    } else if t == "date" {
        TzKind::Date
    } else if t == "timestamp_ntz" {
        TzKind::TimestampNtz
    } else if t == "timestamp" || t == "timestamp_ltz" {
        TzKind::Timestamp
    } else if t == "variant" {
        TzKind::Variant
    } else {
        TzKind::Other
    }
}

/// Spark's `Cast.needsTimeZone` over the top-level types (nested arrays, maps and structs are not
/// looked into): such a cast stays unresolved until `ResolveTimeZone` gives it a zone, so Spark's
/// binding fails on it. Every pair was checked in the Spark 4.1.0 + Iceberg 1.12.0 run (string
/// → timestamp, date → timestamp, timestamp → date, timestamp → string, timestamp ↔
/// timestamp_ntz fail; string → date, string → timestamp_ntz, string → int bind).
fn needs_time_zone(from: &str, to: &str) -> bool {
    use TzKind as K;
    matches!(
        (tz_kind(from), tz_kind(to)),
        (K::Variant, _)
            | (K::String | K::Date | K::TimestampNtz, K::Timestamp)
            | (K::Timestamp, K::String | K::Date | K::TimestampNtz)
    )
}

/// Whether `typeof` names a date, timestamp or interval type.
fn is_datetime(t: &str) -> bool {
    t == "date" || t.starts_with("timestamp") || t.starts_with("interval")
}

/// The interval family of a `typeof` name: calendar, year-month or day-time; `None` for any other
/// type.
fn interval_family(t: &str) -> Option<u8> {
    let rest = t.strip_prefix("interval")?;
    Some(if rest.is_empty() {
        0
    } else if rest.contains("year") || rest.contains("month") {
        1
    } else {
        2
    })
}

impl Session {
    /// The arguments Spark's binding cannot take are refused (`KRISHIV_UNSUPPORTED_FEATURE`).
    /// Spark binds a procedure's arguments once they are resolved, and an argument that a rule
    /// of the resolution batch then rewrites fails `Call.checkArgTypes` with an internal
    /// `IllegalArgumentException` ("number of args and params must match after binding"). The
    /// shapes refused here, each checked in a Spark 4.1.0 + Iceberg 1.12.0 run (2026-11-23):
    /// - a subquery (it would also read another table to decide what the procedure changes);
    /// - a call of the random-seed family (`ResolveRandomSeed`);
    /// - `+`/`-` with a date, timestamp or interval operand (`current_date() - 7`, `1 + current_date()`,
    ///   `ts - make_interval(..)`, `ts + INTERVAL 1 DAY`, `date - date`), except an interval
    ///   with an interval of the same family (`INTERVAL 1 DAY + INTERVAL 2 DAYS` binds);
    /// - `/` unless both operands are `double` or both `decimal` (`7 / 7` fails);
    /// - a cast `Cast.needsTimeZone` holds for ([`needs_time_zone`]): `CAST('..' AS TIMESTAMP)`,
    ///   `CAST(current_date() AS TIMESTAMP)`, `CAST(current_timestamp() AS DATE)`, timestamp to
    ///   string, timestamp to and from `TIMESTAMP_NTZ`.
    ///
    /// These bind in the run and are taken: `1 + 1`, `2 * 3`, `length('abc')`,
    /// `current_timestamp()`, `date_sub(current_date(), 5)`, `datediff(..)`, `interval * 2`,
    /// `CAST('..' AS DATE)`, `CAST('..' AS TIMESTAMP_NTZ)`, `CAST('1' AS INT)`, a plain string or
    /// `DATE` literal for a timestamp parameter (Spark casts those itself). Divergence toward
    /// refusal: `CAST(date_sub(current_date(), 7) AS TIMESTAMP)` binds in the run although
    /// `CAST(current_date() AS TIMESTAMP)` does not; both sources are `date`, so the type rule
    /// refuses both (`older_than => date_sub(current_date(), 7)` without the cast is taken). Other
    /// rewrites of Spark's resolution batch (coercions of other operators, casts inside nested
    /// types) are not probed; such an argument is evaluated here, where Spark may fail.
    async fn refuse_unbindable(&self, call: &Call, query: &str) -> Result<(), KrishivError> {
        let mut typed: Vec<Typed<'_>> = Vec::new();
        let mut stack: Vec<&Expr> = call.args.iter().map(|a| &a.expr).collect();
        while let Some(e) = stack.pop() {
            match &e.kind {
                ExprKind::Subquery(_) | ExprKind::Exists(_) | ExprKind::InSubquery { .. } => {
                    return Err(refuse_argument("a subquery"));
                }
                // Refused by name: the type probes below analyze an operand alone, where a
                // lambda's parameter has no binding (Spark's result for this form is not pinned
                // by a run yet, so no lambda argument is accepted).
                ExprKind::Lambda { .. } => {
                    return Err(refuse_argument("a lambda function"));
                }
                ExprKind::Function(f)
                    if f.name.parts.last().is_some_and(|n| {
                        RANDOM_SEED_FUNCTIONS
                            .iter()
                            .any(|d| n.value.eq_ignore_ascii_case(d))
                    }) =>
                {
                    return Err(refuse_argument("a random-seed function (rand, uuid, ..)"));
                }
                ExprKind::Binary(l, BinaryOp::Plus | BinaryOp::Minus, r) => {
                    typed.push(Typed::AddOrSubtract(l, r));
                }
                ExprKind::Binary(l, BinaryOp::Divide, r) => typed.push(Typed::Divide(l, r)),
                ExprKind::Cast {
                    expr, data_type, ..
                } => {
                    let target = query
                        .get(data_type.span.start..data_type.span.end)
                        .unwrap_or_default()
                        .trim()
                        .to_ascii_lowercase();
                    typed.push(Typed::Cast(expr, target));
                }
                _ => {}
            }
            e.for_each_child(&mut |c| stack.push(c));
        }
        for t in typed {
            match t {
                Typed::AddOrSubtract(l, r) => {
                    let [a, b] = self.types_of(query, [l, r]).await?;
                    let same_intervals = matches!(
                        (interval_family(&a), interval_family(&b)),
                        (Some(x), Some(y)) if x == y
                    );
                    if (is_datetime(&a) || is_datetime(&b)) && !same_intervals {
                        return Err(refuse_argument(
                            "+ or - with a date, timestamp or interval operand",
                        ));
                    }
                }
                Typed::Divide(l, r) => {
                    let [a, b] = self.types_of(query, [l, r]).await?;
                    let fractional = (a == "double" && b == "double")
                        || (a.starts_with("decimal") && b.starts_with("decimal"));
                    if !fractional {
                        return Err(refuse_argument(
                            "/ of operands other than two doubles or two decimals",
                        ));
                    }
                }
                Typed::Cast(x, target) => {
                    let [from] = self.types_of(query, [x]).await?;
                    if needs_time_zone(&from, &target) {
                        return Err(refuse_argument(&format!(
                            "CAST of a {from} AS {}",
                            target.to_ascii_uppercase()
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// `typeof` of each expression, as Spark spells types (`date`, `interval day`).
    async fn types_of<const N: usize>(
        &self,
        query: &str,
        exprs: [&Expr; N],
    ) -> Result<[String; N], KrishivError> {
        let items: Vec<String> = exprs
            .iter()
            .map(|e| {
                let text = query.get(e.span.start..e.span.end).unwrap_or_default();
                format!("typeof(({text}))")
            })
            .collect();
        let batch = self
            .internal_row(&format!("SELECT {}", items.join(", ")))
            .await?;
        let mut out: [String; N] = std::array::from_fn(|_| String::new());
        for (i, slot) in out.iter_mut().enumerate() {
            let col = batch.column(i);
            // Fail closed: a `typeof` column of another shape must not read as a type no rule
            // refuses.
            *slot = col
                .as_string_opt::<i32>()
                .map(|a| a.value(0).to_owned())
                .or_else(|| col.as_string_view_opt().map(|a| a.value(0).to_owned()))
                .ok_or_else(|| {
                    crate::session::internal(format!(
                        "typeof of a CALL argument gave a {:?} column",
                        col.data_type()
                    ))
                })?;
        }
        Ok(out)
    }
}

/// The names `UNRECOGNIZED_PARAMETER_NAME` proposes: Spark's `orderSuggestedIdentifiersBySimilarity`
/// over `name_order` (Spark's candidate order), stably sorted by the edit distance between the
/// quoted parameter and the name as written, the first three (matches the Spark 4.1.0 + Iceberg
/// 1.12.0 run).
fn proposal(name: &str, name_order: &[&str]) -> String {
    let mut ranked: Vec<(usize, String)> = name_order
        .iter()
        .map(|p| {
            let q = quote(p);
            (levenshtein(&q, name), q)
        })
        .collect();
    ranked.sort_by_key(|(d, _)| *d);
    ranked
        .into_iter()
        .take(3)
        .map(|(_, q)| q)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Character edit distance (insertions, deletions, substitutions).
fn levenshtein(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let cur = row[j + 1];
            row[j + 1] = (prev + usize::from(x != *y)).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

/// For each parameter of `signature`, the index of the argument that gives it (`None`: the
/// default, `NULL`). Spark's `NamedParametersSupport.defaultRearrange`: positional arguments
/// first, then named ones; the errors are its.
fn rearrange(
    routine: &str,
    signature: &[ProcedureParameter],
    name_order: &[&str],
    call: &Call,
    case_sensitive: bool,
) -> Result<Vec<Option<usize>>, KrishivError> {
    let param = |name: &str| quote(name);
    let same = |a: &str, b: &str| {
        if case_sensitive {
            a == b
        } else {
            a.eq_ignore_ascii_case(b)
        }
    };
    let names: Vec<Option<&str>> = call
        .args
        .iter()
        .map(|a| a.name.as_ref().map(|n| n.value.as_str()))
        .collect();
    let positional = names
        .iter()
        .position(Option::is_some)
        .unwrap_or(names.len());
    let mut seen: Vec<&str> = Vec::new();
    for (i, name) in names.iter().enumerate().skip(positional) {
        match name {
            None => {
                let prev = i.checked_sub(1).and_then(|j| names[j]).unwrap_or_default();
                return Err(UnexpectedPositionalArgument {
                    routine_name: routine.to_owned(),
                    parameter_name: param(prev),
                }
                .into());
            }
            Some(name) => {
                if seen.iter().any(|s| same(s, name)) {
                    return Err(
                        DuplicateRoutineParameterAssignmentDoubleNamedArgumentReference {
                            routine_name: routine.to_owned(),
                            parameter_name: param(name),
                        }
                        .into(),
                    );
                }
                seen.push(name);
            }
        }
    }
    for name in &seen {
        if !signature.iter().any(|p| same(p.name, name)) {
            return Err(UnrecognizedParameterName {
                routine_name: routine.to_owned(),
                argument_name: param(name),
                proposal: proposal(name, name_order),
            }
            .into());
        }
        if signature[..positional.min(signature.len())]
            .iter()
            .any(|p| same(p.name, name))
        {
            return Err(DuplicateRoutineParameterAssignmentBothPositionalAndNamed {
                routine_name: routine.to_owned(),
                parameter_name: param(name),
            }
            .into());
        }
    }
    if signature.len() < call.args.len() {
        let required = signature.iter().take_while(|p| p.required).count();
        let valid: Vec<String> = (required..=signature.len())
            .map(|v| v.to_string())
            .collect();
        let expected = match valid.as_slice() {
            [only] => only.clone(),
            _ => format!("[{}]", valid.join(", ")),
        };
        return Err(WrongNumArgsWithoutSuggestion {
            function_name: routine.to_owned(),
            expected_num: expected,
            actual_num: call.args.len().to_string(),
            docroot: DOC_ROOT.to_owned(),
        }
        .into());
    }
    let mut slots: Vec<Option<usize>> = (0..positional.min(call.args.len())).map(Some).collect();
    for (i, p) in signature.iter().enumerate().skip(positional) {
        if let Some(at) = names
            .iter()
            .position(|n| n.is_some_and(|n| same(n, p.name)))
        {
            slots.push(Some(at));
        } else if p.required {
            return Err(RequiredParameterNotFound {
                routine_name: routine.to_owned(),
                parameter_name: param(p.name),
                index: (i - positional).to_string(),
            }
            .into());
        } else {
            slots.push(None);
        }
    }
    Ok(slots)
}

/// `e` naming the catalog `catalog` when it is a catalog's refusal of procedures: Spark's
/// `missingCatalogProceduresAbilityError(plugin)` names `plugin.name` (v4.3.0
/// `QueryCompilationErrors.scala`), which a catalog's own default refusal cannot know.
fn names_plugin(e: KrishivError, catalog: &str) -> KrishivError {
    if e.error_class() == krishiv_common::error::ErrorClass::MissingCatalogAbilityProcedures {
        return MissingCatalogAbilityProcedures {
            plugin: catalog.to_owned(),
        }
        .into();
    }
    e
}
