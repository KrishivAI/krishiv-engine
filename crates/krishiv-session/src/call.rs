// SPDX-License-Identifier: Apache-2.0
//! `CALL cat.system.p(args)` (design P3-T027 PR-36e §3): resolves the catalog, loads the
//! procedure, rearranges the arguments by name as Spark's `NamedParametersSupport` does, casts
//! each to its parameter's type (ANSI, through the session's own `CAST`), runs the procedure and
//! returns its rows. Not planned through the optimizer and never cached.

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use krishiv_common::error::{
    DuplicateRoutineParameterAssignmentBothPositionalAndNamed,
    DuplicateRoutineParameterAssignmentDoubleNamedArgumentReference, FailedToLoadRoutine,
    KrishivError, MissingCatalogAbilityProcedures, RequiredParameterNotFound,
    UnexpectedPositionalArgument, UnrecognizedParameterName, WrongNumArgsWithoutSuggestion,
};
use krishiv_plan::procedure::{
    ProcedureContext, ProcedureParameter, ProcedureType, ProcedureValue,
};
use krishiv_sql::ast::{Call, DdlName};

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
            built
                .load_procedure(&caller, &procedure.to_lowercase())
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
        let slots = rearrange(&shown, loaded.parameters(), call)?;
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
        let table = loaded.table(&args);
        let batch = loaded.call(&ctx, args).await?;
        if loaded.modifies_table()
            && let Some(table) = table
        {
            // Frames and plans that read the table analyze again (`REFRESH TABLE`).
            let (namespace, name) = split_table(&table);
            let id = krishiv_plan::analyzer::TableIdent {
                catalog,
                namespace,
                name,
            };
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

/// The table's `ns.t` name as a namespace and a name.
fn split_table(table: &str) -> (Vec<String>, String) {
    let mut parts: Vec<String> = table.split('.').map(str::to_owned).collect();
    let name = parts.pop().unwrap_or_default();
    (parts, name)
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
                    text(values, i).unwrap_or_default(),
                ));
            }
            ProcedureValue::StringMap(out)
        }
        ProcedureType::LongArray => {
            let list = col.as_list_opt::<i32>().ok_or_else(bad)?;
            let items = list.value(0);
            let longs = items.as_primitive_opt::<Int64Type>().ok_or_else(bad)?;
            ProcedureValue::LongArray(longs.iter().flatten().collect())
        }
        _ => return Err(bad()),
    })
}

/// For each parameter of `signature`, the index of the argument that gives it (`None`: the
/// default, `NULL`). Spark's `NamedParametersSupport.defaultRearrange`: positional arguments
/// first, then named ones; the errors are its.
fn rearrange(
    routine: &str,
    signature: &[ProcedureParameter],
    call: &Call,
) -> Result<Vec<Option<usize>>, KrishivError> {
    let param = |name: &str| quote(name);
    let same = |a: &str, b: &str| a.eq_ignore_ascii_case(b);
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
                proposal: String::new(),
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
