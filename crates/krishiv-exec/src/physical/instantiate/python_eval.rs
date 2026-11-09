// SPDX-License-Identifier: Apache-2.0
//! Instantiation of a `PhysNode::PythonEval` (design P2-T059): each call's
//! arguments compiled over the input and its UDF prepared by the worker's
//! [`UdfRuntime`].

use super::*;
use krishiv_common::config::ARROW_MAX_RECORDS_PER_BATCH;
use krishiv_common::error::{KrishivError, KrishivUdfNotAvailable};
use krishiv_plan::{ExprNode, FunctionId, ResolvedExpr};
use krishiv_types::arrow::client_arrow_type;

use crate::operators::ScalarExpr;
use crate::physical::PythonCallPlan;
use crate::python_eval::{DEFAULT_WINDOW, EvalCall, PythonEvalSink, UdfCallSpec, UdfRuntime};

impl WorkerContext {
    /// The same context calling Python UDFs through `runtime`. A plan
    /// calling a Python UDF without one fails with
    /// `KRISHIV_UDF_NOT_AVAILABLE`.
    #[must_use]
    pub fn with_python_udfs(mut self, runtime: Arc<dyn UdfRuntime>) -> Self {
        self.python = Some(runtime);
        self
    }
}

impl Instantiation<'_> {
    /// A `PythonEval` over `input`: a breaker whose sink submits the input's
    /// batches to the calls and whose source yields them with the results.
    pub(super) fn python_eval(&mut self, node: &PhysNode) -> Result<Built> {
        let PhysNode::PythonEval {
            input,
            layout,
            calls,
        } = node
        else {
            return Err(internal("not a PythonEval node".to_owned()));
        };
        // A range's order is kept only where a `LIMIT` or the result needs it
        // (`order.rs`); then one driver submits the input, so the list holds
        // the slices in range order. Elsewhere the results are read in
        // parallel (design P2-T059 §6.3).
        let range_ordered = self
            .range_ordered
            .contains(&std::ptr::from_ref(node).addr());
        let ordered = ends_in_sort(input) || range_ordered;
        let mut input = self.node(input)?;
        if range_ordered {
            input.node = Arc::new(OrderedNode { inner: input.node });
        }
        let sink = self.python_eval_sink(&input.schema, layout, calls)?;
        let source = SharedSource::new(sink.schema());
        Ok(Built {
            schema: sink.schema(),
            node: Arc::new(BreakerNode {
                input: input.node,
                sink,
                source,
                ordered,
            }),
        })
    }

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

/// Whether `n` ends in a sort (through row-wise nodes): its rows are in
/// order, so a breaker over it keeps that order (one driver above it).
fn ends_in_sort(n: &PhysNode) -> bool {
    match n {
        PhysNode::Sort { .. } | PhysNode::TopN { .. } => true,
        PhysNode::Project { input, .. }
        | PhysNode::Filter { input, .. }
        | PhysNode::LateFetch { input, .. }
        | PhysNode::PythonEval { input, .. }
        | PhysNode::Limit { input, .. } => ends_in_sort(input),
        _ => false,
    }
}
