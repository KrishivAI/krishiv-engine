// SPDX-License-Identifier: Apache-2.0
//! Instantiation of a `PhysNode::PythonEval` (design P2-T059): each call's
//! arguments compiled over the input and its UDF prepared by the worker's
//! [`UdfRuntime`].

use super::*;

impl Instantiation<'_> {
    /// The sink of a `PythonEval` over input of `schema` (attributes
    /// `layout`): each call's arguments compiled over the input and its UDF
    /// prepared by the worker's runtime.
    pub(super) fn python_eval_sink(
        &self,
        schema: &SchemaRef,
        layout: &[AttrId],
        calls: &[PythonCallPlan],
    ) -> Result<Arc<PythonEvalSink>> {
        let values = Values {
            env: &self.env,
            subqueries: &[],
        };
        let mut compiled = Vec::with_capacity(calls.len());
        for c in calls {
            let ExprNode::Resolved(ResolvedExpr::Function(f)) = &c.call.node else {
                return Err(internal("a PythonEval call is not a function".to_owned()));
            };
            let FunctionId::PythonUdf {
                name,
                version,
                eval_type,
                thread_safe,
            } = &f.id
            else {
                return Err(internal("a PythonEval call is not a Python UDF".to_owned()));
            };
            let unavailable = || {
                KrishivError::from(KrishivUdfNotAvailable {
                    name: name.clone(),
                    version: version.to_string(),
                })
            };
            let runtime = self.worker.python.as_ref().ok_or_else(unavailable)?;
            let args: Vec<&Expr> = f.args.iter().map(AsRef::as_ref).collect();
            let args = compile_all(&args, layout, schema, &values)?;
            let arg_types = f
                .args
                .iter()
                .map(|a| {
                    a.resolved_type().ok_or_else(|| {
                        internal("a Python UDF argument has no resolved type".to_owned())
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let canonical = f.data_type.to_arrow();
            let client = client_arrow_type(&canonical);
            let udf = runtime.prepare(&UdfCallSpec {
                name: name.clone(),
                version: *version,
                eval_type: *eval_type,
                thread_safe: *thread_safe,
                arg_types,
                return_type: f.data_type.clone(),
                return_arrow: client.clone(),
            })?;
            compiled.push(EvalCall {
                args: args
                    .into_iter()
                    .map(|e| Arc::new(Compiled(e)) as Arc<dyn ScalarExpr>)
                    .collect(),
                udf,
                client,
                canonical,
            });
        }
        let max_rows = self
            .worker
            .config
            .as_deref()
            .and_then(|c| c.get(&ARROW_MAX_RECORDS_PER_BATCH))
            .and_then(krishiv_common::config::ConfigValue::as_int)
            .unwrap_or(10_000);
        let max_rows = usize::try_from(max_rows).unwrap_or(0);
        Ok(PythonEvalSink::new(
            schema,
            compiled,
            max_rows,
            DEFAULT_WINDOW,
            self.spill.clone(),
        ))
    }
}
