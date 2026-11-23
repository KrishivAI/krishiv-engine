// SPDX-License-Identifier: Apache-2.0
//! `CALL cat.system.p(args)` (design P3-T027 PR-36e §3): resolves the catalog, loads the
//! procedure, rearranges the arguments by name as Spark's `NamedParametersSupport` does, casts
//! each to its parameter's type (ANSI whatever `spark.sql.ansi.enabled` says, through a clone of
//! the session, so a bad value fails instead of becoming `NULL`), runs the procedure and returns
//! its rows. A `NULL` element of an array argument is refused. Not planned through the optimizer
//! and never cached.

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
use krishiv_sql::ast::{BinaryOp, Call, DdlName, Expr, ExprKind, Literal};

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
        let Some(built) = self.declared_catalogs().built(&catalog) else {
            return Err(MissingCatalogAbilityProcedures {
                plugin: catalog.clone(),
            }
            .into());
        };
        let (built, caller) = built?;
        let shown = quote_all(&[catalog.clone(), namespace.clone(), procedure.clone()]);
        let loaded = if namespace.eq_ignore_ascii_case("system") {
            let inputs = self.procedure_inputs();
            built
                .load_procedure(&caller, inputs, &procedure.to_lowercase())
                .await?
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
        // compares names under the session resolver.
        let case_sensitive = self
            .conf()
            .get("spark.sql.caseSensitive")
            .is_some_and(|v| v.eq_ignore_ascii_case("true"));
        let slots = rearrange(
            &quote(loaded.name()),
            loaded.parameters(),
            &loaded.name_order(),
            call,
            case_sensitive,
        )?;
        refuse_unbindable(call)?;
        // The casts run in a clone with ANSI on: a non-ANSI cast turns a bad value into `NULL`,
        // which a procedure reads as an absent argument (an expire would fall back to its
        // default retention).
        let caster = if slots.iter().any(Option::is_some) {
            let c = self.clone_session()?;
            c.conf().set("spark.sql.ansi.enabled", "true")?;
            Some(c)
        } else {
            None
        };
        let mut args = Vec::with_capacity(slots.len());
        for (param, slot) in loaded.parameters().iter().zip(slots) {
            args.push(match slot {
                None => ProcedureValue::Null,
                Some(i) => {
                    let arg = &call.args[i];
                    let text = query
                        .get(arg.expr.span.start..arg.expr.span.end)
                        .unwrap_or_default();
                    let caster = caster.as_ref().unwrap_or(self);
                    caster.cast_argument(text, param.data_type).await?
                }
            });
        }
        if let Some(c) = caster {
            c.close();
        }
        let ctx = ProcedureContext {
            catalog: catalog.clone(),
        };
        let target = loaded
            .table(&args)
            .filter(|_| loaded.modifies_table())
            .map(|t| table_ident(&catalog, &t));
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
        let batch = loaded.call(&ctx, args).await?;
        if let Some(id) = target {
            // Frames and plans that read the table analyze again (`REFRESH TABLE`).
            built.refresh(&caller, &id).await?;
        }
        self.script_result(batch.schema(), vec![batch]).await
    }

    /// `SELECT CAST((text) AS type)`: the value, as the session's `CAST` gives it.
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
        let frame = self.sql_boxed(&format!("SELECT {expr}")).await?;
        let batches = frame.collect_boxed().await?;
        let batch = batches.iter().find(|b| b.num_rows() > 0).ok_or_else(|| {
            crate::session::internal("a procedure argument gave no row".to_owned())
        })?;
        value_of(batch, ty)
    }
}

/// A procedure's `table` argument (`ns.t`, or `cat.ns.t` naming the catalog the call runs in,
/// as the catalog reads it) as the table in `catalog`.
fn table_ident(catalog: &str, table: &str) -> krishiv_plan::analyzer::TableIdent {
    let mut parts: Vec<String> = table.split('.').map(str::to_owned).collect();
    if parts.len() > 2 && parts[0].eq_ignore_ascii_case(catalog) {
        parts.remove(0);
    }
    let name = parts.pop().unwrap_or_default();
    krishiv_plan::analyzer::TableIdent {
        catalog: catalog.to_owned(),
        namespace: parts,
        name,
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
                        return Err(null_element());
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
            let strings = items.as_string_opt::<i32>().ok_or_else(bad)?;
            ProcedureValue::StringArray(
                strings
                    .iter()
                    .map(|v| v.map(str::to_owned))
                    .collect::<Option<Vec<String>>>()
                    .ok_or_else(null_element)?,
            )
        }
        _ => return Err(bad()),
    })
}

/// The arguments Spark's binding cannot take are refused: a subquery, a non-deterministic call
/// and `+`/`-` with an `INTERVAL` operand. In a Spark 4.1.0 + Iceberg 1.12.0 run each fails
/// with an internal `IllegalArgumentException` ("number of args and params must match after
/// binding"); `1 + 1`, `length('abc')`, `current_timestamp()` and `date_sub(current_date(), 5)`
/// bind. A subquery would also read another table to decide what the procedure changes.
fn refuse_unbindable(call: &Call) -> Result<(), KrishivError> {
    const NON_DETERMINISTIC: [&str; 7] = [
        "rand", "randn", "random", "uuid", "shuffle", "uniform", "randstr",
    ];
    let refuse = |what: &str| -> KrishivError {
        krishiv_common::error::KrishivUnsupportedFeature {
            feature: format!("{what} in a CALL argument"),
        }
        .into()
    };
    let is_interval = |e: &Expr| {
        let mut e = e;
        while let ExprKind::Nested(inner) = &e.kind {
            e = inner;
        }
        matches!(e.kind, ExprKind::Literal(Literal::Interval(_)))
    };
    let mut stack: Vec<&Expr> = call.args.iter().map(|a| &a.expr).collect();
    while let Some(e) = stack.pop() {
        match &e.kind {
            ExprKind::Subquery(_) | ExprKind::Exists(_) | ExprKind::InSubquery { .. } => {
                return Err(refuse("a subquery"));
            }
            ExprKind::Function(f)
                if f.name.parts.last().is_some_and(|n| {
                    NON_DETERMINISTIC
                        .iter()
                        .any(|d| n.value.eq_ignore_ascii_case(d))
                }) =>
            {
                return Err(refuse("a non-deterministic expression"));
            }
            ExprKind::Binary(l, BinaryOp::Plus | BinaryOp::Minus, r)
                if is_interval(l) || is_interval(r) =>
            {
                return Err(refuse("datetime arithmetic with an INTERVAL"));
            }
            _ => {}
        }
        e.for_each_child(&mut |c| stack.push(c));
    }
    Ok(())
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
