// SPDX-License-Identifier: Apache-2.0
//! `EXECUTE IMMEDIATE query [INTO v, ...] [USING arg [AS name], ...]`.
//!
//! The statement text and every argument are evaluated first, in the
//! session, as a variable default is. The arguments are literals or
//! variables (a subquery is refused); each parameter marker in the text is
//! replaced by the literal bound to it, and the result runs through the
//! ordinary pipeline, so it is parsed, authorized and executed exactly like
//! the same statement typed by the user.

use std::collections::{BTreeSet, HashMap};

use arrow_array::RecordBatch;
use krishiv_common::error::{
    AllParametersMustBeNamed, AssignmentArityMismatch, DuplicateAssignments,
    ExecImmediateDuplicateArgumentAliases, InvalidExprTypeForQueryExecuteImmediate,
    InvalidQueryMixedQueryParameters, InvalidStatementForExecuteInto, KrishivError,
    KrishivUnsupportedFeature, NestedExecuteImmediate, NullQueryStringExecuteImmediate,
    RowSubqueryTooManyRows, UnboundSqlParameter, UnsupportedExprForParameter,
};
use krishiv_common::redact::RedactedSql;
use krishiv_plan::Literal;
use krishiv_sql::ast::{ExecuteImmediate, Expr, ExprKind, Statement};
use krishiv_types::DataType;

use crate::command::{arrow_value, padded_select};
use crate::dataframe::DataFrame;
use crate::session::{Session, sql_context};
use crate::variables::{StoredVariable, quote_normalized, variable_key};

/// How many `EXECUTE IMMEDIATE` statements may run inside one another: the
/// text of a statement may itself be an `EXECUTE IMMEDIATE`, once.
const MAX_NESTING: usize = 1;

/// One evaluated `USING` argument.
struct Argument {
    /// The name it binds: its alias, or the variable it reads.
    name: Option<String>,
    /// Whether `name` is an alias written with `AS`.
    explicit: bool,
    /// Its value as SQL text.
    sql: String,
    /// How an error names it.
    shown: String,
}

impl Session {
    /// Runs `EXECUTE IMMEDIATE`; `depth` counts the enclosing ones.
    pub(crate) async fn exec_execute_immediate(
        &self,
        e: &ExecuteImmediate,
        original: &str,
        text: &RedactedSql,
        depth: usize,
    ) -> Result<DataFrame, KrishivError> {
        let slice = |x: &Expr| original.get(x.span.start..x.span.end).unwrap_or_default();
        let env0 = self.query_env();
        let case_sensitive = env0.analysis.conf.case_sensitive;
        let escaped = env0.parser.escaped_string_literals;

        refuse_subquery(&e.query)?;
        let query_sql = padded_select("SELECT ", slice(&e.query), e.query.span.start, "");
        let (data_type, value) = self
            .eval_variable_value(&query_sql, text, e.span, e.query.span)
            .await?;
        let statement_text = match value {
            Literal::String(s) => s,
            Literal::Null(_) if matches!(data_type, DataType::String | DataType::Null) => {
                let written = slice(&e.query).trim();
                return Err(NullQueryStringExecuteImmediate {
                    var_name: if written.starts_with('`') {
                        written.to_owned()
                    } else {
                        format!("`{written}`")
                    },
                }
                .into());
            }
            _ => {
                return Err(InvalidExprTypeForQueryExecuteImmediate {
                    expr_type: format!("\"{}\"", data_type.sql()),
                }
                .into());
            }
        };

        let mut args = Vec::with_capacity(e.using.len());
        for a in &e.using {
            refuse_subquery(&a.value)?;
            let sql = padded_select("SELECT ", slice(&a.value), a.value.span.start, "");
            let (_, value) = self
                .eval_variable_value(&sql, text, e.span, a.value.span)
                .await?;
            let name = match (&a.alias, &a.value.kind) {
                (Some(alias), _) => Some(alias.value.clone()),
                (None, ExprKind::Column(parts)) if parts.len() == 1 => {
                    parts.first().map(|p| p.value.clone())
                }
                (None, _) => None,
            };
            args.push(Argument {
                explicit: a.alias.is_some(),
                name,
                sql: render_literal(&value, escaped)?,
                shown: shown_expression(slice(&a.value)),
            });
        }
        duplicate_aliases(&args, case_sensitive)?;

        let bound = bind(&statement_text, &args, case_sensitive)?;
        let inner_text = krishiv_sql::redact_sql(&bound);
        let env = self.query_env();
        let statement = krishiv_sql::parse_statement(&bound, &env.parser)
            .map_err(|err| sql_context(err.error, &inner_text, err.span.start, err.span.end))?;
        if !e.into.is_empty() && !matches!(statement, Statement::Query(_)) {
            return Err(InvalidStatementForExecuteInto {
                sql_string: statement_text.clone(),
            }
            .into());
        }
        let frame = match statement {
            Statement::ExecuteImmediate(inner) => {
                if depth >= MAX_NESTING {
                    return Err(NestedExecuteImmediate {
                        sql_string: inner_text.fragment(0..bound.len()).to_owned(),
                    }
                    .into());
                }
                Box::pin(self.exec_execute_immediate(&inner, &bound, &inner_text, depth + 1))
                    .await?
            }
            statement => {
                let class = krishiv_sql::classify(&statement);
                self.authorize_statement(&statement, class, &env).await?;
                Box::pin(self.run_statement(statement, env, &bound, &inner_text)).await?
            }
        };
        if e.into.is_empty() {
            return Ok(frame);
        }
        self.assign_into(e, &frame, text).await
    }

    /// `INTO v, ...`: the one row of `frame` goes to the variables, cast to
    /// their types; no row sets them to NULL.
    async fn assign_into(
        &self,
        e: &ExecuteImmediate,
        frame: &DataFrame,
        text: &RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        let mut keys = Vec::with_capacity(e.into.len());
        for name in &e.into {
            let Some(key) = variable_key(name) else {
                return Err(crate::command::unresolved_variable(
                    name,
                    crate::variables::RESOLVE_SEARCH_PATH,
                    text,
                ));
            };
            keys.push(key);
        }
        let mut seen = BTreeSet::new();
        if let Some(dup) = keys.iter().find(|k| !seen.insert(k.as_str())) {
            return Err(DuplicateAssignments {
                name_list: format!("`{}`", dup.replace('`', "``")),
            }
            .into());
        }
        let mut stored = Vec::with_capacity(keys.len());
        for key in &keys {
            let Some(var) = self.state.variables.get(key) else {
                return Err(krishiv_common::error::VariableNotFound {
                    variable_name: quote_normalized(key),
                }
                .into());
            };
            stored.push(var);
        }
        let width = frame.schema().await?.fields().len();
        if width != keys.len() {
            return Err(AssignmentArityMismatch {
                num_target: keys.len().to_string(),
                num_expr: width.to_string(),
            }
            .into());
        }
        let batches = frame.collect().await?;
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        if rows > 1 {
            return Err(RowSubqueryTooManyRows {}.into());
        }
        let batch = batches.iter().find(|b| b.num_rows() > 0);
        let mut pending = Vec::with_capacity(keys.len());
        for (i, (key, var)) in keys.iter().zip(&stored).enumerate() {
            let value = match batch {
                None => Literal::Null(Some(var.data_type.clone())),
                Some(b) => {
                    let (_, raw) = arrow_value(b.column(i), 0)?;
                    let cast = format!(
                        "SELECT CAST({} AS {})",
                        render_literal(&raw, self.query_env().parser.escaped_string_literals)?,
                        var.data_type.simple_string()
                    );
                    let span = e.span;
                    self.eval_variable_value(&cast, text, span, span).await?.1
                }
            };
            pending.push((key.clone(), var.clone(), value));
        }
        for (key, var, value) in pending {
            self.state.variables.put(
                key,
                StoredVariable {
                    data_type: var.data_type,
                    value,
                    default_query: var.default_query,
                },
            );
        }
        self.empty_frame().await
    }
}

/// A subquery is not a literal or a variable.
fn refuse_subquery(expr: &Expr) -> Result<(), KrishivError> {
    if contains_subquery(expr) {
        return Err(UnsupportedExprForParameter {
            invalid_expr_sql: "\"scalarsubquery()\"".to_owned(),
        }
        .into());
    }
    Ok(())
}

fn contains_subquery(expr: &Expr) -> bool {
    if matches!(
        expr.kind,
        ExprKind::Subquery(_) | ExprKind::Exists(_) | ExprKind::InSubquery { .. }
    ) {
        return true;
    }
    let mut found = false;
    expr.for_each_child(&mut |c| found = found || contains_subquery(c));
    found
}

/// How an unnamed argument is shown in `ALL_PARAMETERS_MUST_BE_NAMED`: its
/// text in double quotes, a string literal without its single quotes.
fn shown_expression(source: &str) -> String {
    let trimmed = source.trim();
    let inner = trimmed
        .strip_prefix('\'')
        .and_then(|s| s.strip_suffix('\''))
        .unwrap_or(trimmed);
    format!("\"{inner}\"")
}

/// `EXEC_IMMEDIATE_DUPLICATE_ARGUMENT_ALIASES` for repeated names.
fn duplicate_aliases(args: &[Argument], case_sensitive: bool) -> Result<(), KrishivError> {
    let mut seen = BTreeSet::new();
    let mut duplicates = Vec::new();
    for name in args
        .iter()
        .filter(|a| a.explicit)
        .filter_map(|a| a.name.as_deref())
    {
        let key = if case_sensitive {
            name.to_owned()
        } else {
            name.to_lowercase()
        };
        if !seen.insert(key) && !duplicates.contains(&name) {
            duplicates.push(name);
        }
    }
    if duplicates.is_empty() {
        return Ok(());
    }
    Err(ExecImmediateDuplicateArgumentAliases {
        aliases: duplicates
            .iter()
            .map(|d| format!("`{d}`"))
            .collect::<Vec<_>>()
            .join(", "),
    }
    .into())
}

/// `sql` with each parameter marker replaced by its argument.
///
/// Positional markers take the arguments in order (names are ignored);
/// named markers take the argument of that name, and then every argument
/// must be named. Mixing the two is an error, as is a marker without an
/// argument.
fn bind(sql: &str, args: &[Argument], case_sensitive: bool) -> Result<String, KrishivError> {
    let markers = krishiv_sql::parameter_markers(sql);
    let named = markers.iter().filter(|m| m.name.is_some()).count();
    if named > 0 && named < markers.len() {
        return Err(InvalidQueryMixedQueryParameters {}.into());
    }
    let key = |s: &str| {
        if case_sensitive {
            s.to_owned()
        } else {
            s.to_lowercase()
        }
    };
    let by_name: HashMap<String, &Argument> = args
        .iter()
        .filter_map(|a| a.name.as_deref().map(|n| (key(n), a)))
        .collect();
    if named > 0 {
        let unnamed: Vec<&str> = args
            .iter()
            .filter(|a| a.name.is_none())
            .map(|a| a.shown.as_str())
            .collect();
        if !unnamed.is_empty() {
            return Err(AllParametersMustBeNamed {
                exprs: unnamed.join(", "),
            }
            .into());
        }
    }
    let mut out = String::with_capacity(sql.len());
    let mut at = 0;
    for (ordinal, m) in markers.iter().enumerate() {
        out.push_str(sql.get(at..m.start).unwrap_or_default());
        let arg = match &m.name {
            Some(name) => by_name.get(&key(name)).copied(),
            None => args.get(ordinal),
        };
        let Some(arg) = arg else {
            return Err(UnboundSqlParameter {
                name: m.name.clone().unwrap_or_else(|| {
                    format!("_{}", sql.get(..m.start).map_or(0, |p| p.chars().count()))
                }),
            }
            .into());
        };
        pad_value(&mut out, &arg.sql, sql.get(m.end..).unwrap_or_default());
        at = m.end;
    }
    out.push_str(sql.get(at..).unwrap_or_default());
    Ok(out)
}

/// Appends `value` to `out` so that it stays a token of its own: a space where the text around it
/// would otherwise join it (`5-?` with `-3`, `?FROM`).
fn pad_value(out: &mut String, value: &str, rest: &str) {
    let word = |c: char| c.is_alphanumeric() || matches!(c, '_' | '.' | '$' | '\'' | '"' | '`');
    if out
        .chars()
        .next_back()
        .is_some_and(|c| word(c) || matches!(c, '-' | '+' | '/' | '*'))
    {
        out.push(' ');
    }
    out.push_str(value);
    if rest.chars().next().is_some_and(|c| word(c) || c == '-') {
        out.push(' ');
    }
}

/// `value` as SQL text that reads back as the same value and type, or an unsupported-feature
/// error for a type the text cannot carry exactly.
fn render_literal(value: &Literal, escaped_strings: bool) -> Result<String, KrishivError> {
    let unsupported = |what: &str| -> KrishivError {
        KrishivUnsupportedFeature {
            feature: format!("a parameter value of {what}"),
        }
        .into()
    };
    match value {
        Literal::Null(None) => Ok("NULL".to_owned()),
        Literal::Null(Some(t)) => Ok(format!("CAST(NULL AS {})", t.simple_string())),
        Literal::String(s) => {
            if escaped_strings && s.contains(['\\', '\'']) {
                return Err(unsupported(
                    "a string holding a quote or backslash under spark.sql.parser.escapedStringLiterals",
                ));
            }
            Ok(format!(
                "'{}'",
                s.replace('\\', "\\\\").replace('\'', "\\'")
            ))
        }
        Literal::Binary(b) => Ok(format!(
            "X'{}'",
            b.iter().map(|x| format!("{x:02X}")).collect::<String>()
        )),
        Literal::Decimal {
            unscaled,
            data_type,
        } => {
            let scale = usize::from(data_type.scale());
            let digits = unscaled.unsigned_abs().to_string();
            let padded = format!("{digits:0>width$}", width = scale + 1);
            let (whole, frac) = padded.split_at(padded.len() - scale);
            let sign = if *unscaled < 0 { "-" } else { "" };
            let number = if scale == 0 {
                format!("{sign}{whole}")
            } else {
                format!("{sign}{whole}.{frac}")
            };
            Ok(format!(
                "CAST({number} AS DECIMAL({},{}))",
                data_type.precision(),
                data_type.scale()
            ))
        }
        Literal::Timestamp(micros) => Ok(format!("TIMESTAMP '{}+00:00'", civil(*micros))),
        Literal::TimestampNtz(micros) => Ok(format!("TIMESTAMP_NTZ '{}'", civil(*micros))),
        Literal::Boolean(_)
        | Literal::Byte(_)
        | Literal::Short(_)
        | Literal::Integer(_)
        | Literal::Long(_)
        | Literal::Float(_)
        | Literal::Double(_)
        | Literal::Date(_) => Ok(value.to_string()),
        _ => Err(unsupported("this type")),
    }
}

/// `micros` since the epoch as `yyyy-MM-dd HH:mm:ss.ffffff`.
fn civil(micros: i64) -> String {
    let days = micros.div_euclid(86_400_000_000);
    let of_day = micros.rem_euclid(86_400_000_000);
    // Days since 1970-01-01 to a proleptic Gregorian date (Hinnant's civil_from_days).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let (secs, frac) = (of_day / 1_000_000, of_day % 1_000_000);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02}.{frac:06}",
        secs / 3600,
        secs / 60 % 60,
        secs % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arg(name: Option<&str>, sql: &str, shown: &str) -> Argument {
        Argument {
            explicit: name.is_some(),
            name: name.map(str::to_owned),
            sql: sql.to_owned(),
            shown: shown.to_owned(),
        }
    }

    fn class(r: Result<String, KrishivError>) -> String {
        r.unwrap_err().error_class().name().to_owned()
    }

    #[test]
    fn markers_bind_by_position_or_name() {
        let args = [arg(Some("b"), "2", "\"2\""), arg(Some("a"), "'x'", "\"x\"")];
        assert_eq!(
            bind("SELECT ?, ? FROM t", &args, false).unwrap(),
            "SELECT 2, 'x' FROM t"
        );
        assert_eq!(
            bind("SELECT :A, :b, ':a'", &args, false).unwrap(),
            "SELECT 'x', 2, ':a'"
        );
        assert_eq!(
            class(bind("SELECT :A", &args, true)),
            "UNBOUND_SQL_PARAMETER"
        );
    }

    #[test]
    fn binding_errors_follow_spark() {
        let named = [arg(Some("a"), "1", "\"1\"")];
        let unnamed = [arg(None, "1", "\"1\"")];
        assert_eq!(
            class(bind("SELECT ? + :a", &named, false)),
            "INVALID_QUERY_MIXED_QUERY_PARAMETERS"
        );
        assert_eq!(
            class(bind("SELECT :a", &unnamed, false)),
            "ALL_PARAMETERS_MUST_BE_NAMED"
        );
        let missing = bind("SELECT ?", &[], false).unwrap_err();
        assert_eq!(missing.error_class().name(), "UNBOUND_SQL_PARAMETER");
        assert!(missing.to_string().contains("_7"), "{missing}");
        // Extra arguments are ignored.
        assert_eq!(bind("SELECT 1", &named, false).unwrap(), "SELECT 1");
    }

    #[test]
    fn duplicate_aliases_ignore_case_unless_asked() {
        let args = [arg(Some("a"), "1", ""), arg(Some("A"), "2", "")];
        assert!(duplicate_aliases(&args, false).is_err());
        assert!(duplicate_aliases(&args, true).is_ok());
    }

    #[test]
    fn literals_render_as_sql_that_reads_back() {
        use krishiv_types::DecimalType;
        let r = |l| render_literal(&l, false).unwrap();
        assert_eq!(r(Literal::String("it's \\ ok".into())), "'it\\'s \\\\ ok'");
        assert_eq!(r(Literal::Binary(vec![1, 255])), "X'01FF'");
        assert_eq!(r(Literal::Null(None)), "NULL");
        assert_eq!(r(Literal::Integer(-3)), "-3");
        let dec = |unscaled, precision, scale| Literal::Decimal {
            unscaled,
            data_type: DecimalType::try_new(precision, scale).unwrap(),
        };
        assert_eq!(r(dec(12345, 10, 2)), "CAST(123.45 AS DECIMAL(10,2))");
        assert_eq!(r(dec(-5, 10, 2)), "CAST(-0.05 AS DECIMAL(10,2))");
        assert_eq!(r(dec(7, 3, 0)), "CAST(7 AS DECIMAL(3,0))");
        assert_eq!(
            r(Literal::Timestamp(1_703_500_245_123_456)),
            "TIMESTAMP '2023-12-25 10:30:45.123456+00:00'"
        );
        assert_eq!(
            r(Literal::TimestampNtz(-1)),
            "TIMESTAMP_NTZ '1969-12-31 23:59:59.999999'"
        );
        // Types the text cannot carry exactly are refused, never guessed.
        let interval = Literal::YearMonthInterval {
            months: 14,
            start: krishiv_plan::YearMonthField::Year,
            end: krishiv_plan::YearMonthField::Month,
        };
        assert!(render_literal(&interval, false).is_err());
        // With escapedStringLiterals a quote or backslash cannot be written back.
        assert!(render_literal(&Literal::String("a'b".into()), true).is_err());
        assert!(render_literal(&Literal::String("plain".into()), true).is_ok());
    }

    #[test]
    fn values_stay_tokens_of_their_own() {
        let pad = |before: &str, value: &str, after: &str| {
            let mut out = before.to_owned();
            pad_value(&mut out, value, after);
            format!("{out}{after}")
        };
        assert_eq!(pad("SELECT 5-", "-3", ", 1"), "SELECT 5- -3, 1");
        assert_eq!(pad("SELECT ", "5", "FROM t"), "SELECT 5 FROM t");
        assert_eq!(pad("SELECT f(", "'x'", ")"), "SELECT f('x')");
        assert_eq!(pad("SELECT ", "1", ""), "SELECT 1");
        assert_eq!(pad("a-", "1", "-b"), "a- 1 -b");
    }

    #[test]
    fn unnamed_arguments_are_shown_in_double_quotes() {
        assert_eq!(shown_expression("'p'"), "\"p\"");
        assert_eq!(shown_expression(" 2 "), "\"2\"");
        assert_eq!(shown_expression("a + 1"), "\"a + 1\"");
    }
}
