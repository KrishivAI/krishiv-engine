// SPDX-License-Identifier: Apache-2.0
//! Instantiation of the Python relation nodes (design P3-T085 §3.1-§3.2):
//! each node's UDFs prepared by the worker's [`UdfRuntime`] as relation call
//! sites, and a [`RelationSink`] breaker over the input.
//!
//! [`UdfRuntime`]: crate::python_eval::UdfRuntime

use super::*;
use arrow_schema::{DataType as ArrowType, Field};
use krishiv_common::config::ARROW_MAX_RECORDS_PER_BATCH;
use krishiv_common::error::{KrishivError, KrishivUdfNotAvailable};
use krishiv_plan::{ExprNode, PythonUdfRef, ResolvedExpr};
use krishiv_types::arrow::client_arrow_type;

use crate::operators::ScalarExpr;
use crate::physical::{PythonCallPlan, PythonRelationPlan};
use crate::python_relation::{
    DEFAULT_WINDOW, InputColumn, PreparedRelationUdf, RelationCallSpec, RelationLane,
    RelationLimits, RelationShape, RelationSink, RunnerConf,
};

/// Spark's `spark.sql.execution.arrow.maxBytesPerBatch`, read as the
/// session wrote it (Krishiv registers no key for it yet): no limit unless
/// set.
const MAX_BYTES_PER_BATCH: &str = "spark.sql.execution.arrow.maxBytesPerBatch";

impl Instantiation<'_> {
    /// A Python relation node: a breaker whose sink feeds the node's tasks
    /// and whose source yields what they returned.
    pub(super) fn python_relation(&mut self, node: &PhysNode) -> Result<Built> {
        let (input, shape) = match node {
            PhysNode::PythonMap { input, .. } => (input, RelationShape::Map),
            PhysNode::PythonGroupMap { input, keys, .. } => {
                (input, RelationShape::Grouped { keys: keys.clone() })
            }
            PhysNode::PythonCoGroupMap {
                input, keys, side, ..
            } => (
                input,
                RelationShape::CoGrouped {
                    keys: keys.clone(),
                    side: *side,
                },
            ),
            PhysNode::PythonAggregate { input, keys, .. } => {
                (input, RelationShape::Aggregate { keys: keys.clone() })
            }
            _ => return Err(internal("not a Python relation node".to_owned())),
        };
        let input = self.node(input)?;
        let conf = self.runner_conf();
        let (lanes, schema) = match node {
            PhysNode::PythonMap { udf, .. }
            | PhysNode::PythonGroupMap { udf, .. }
            | PhysNode::PythonCoGroupMap { udf, .. } => {
                let keys = match &shape {
                    RelationShape::Grouped { keys } | RelationShape::CoGrouped { keys, .. } => {
                        keys.len()
                    }
                    RelationShape::Map | RelationShape::Aggregate { .. } => 0,
                };
                self.relation_lane(&input.schema, udf, keys, &conf)?
            }
            PhysNode::PythonAggregate {
                layout,
                keys,
                calls,
                ..
            } => self.aggregate_lanes(&input.schema, layout, keys, calls, &conf)?,
            _ => return Err(internal("not a Python relation node".to_owned())),
        };
        let sink = RelationSink::try_new(
            Arc::clone(&input.schema),
            shape,
            lanes,
            schema,
            RelationLimits {
                max_rows: conf.max_records_per_batch,
                max_bytes: conf.max_bytes_per_batch,
                window: DEFAULT_WINDOW,
            },
            self.spill.clone(),
        )?;
        let source = SharedSource::new(sink.schema());
        Ok(Built {
            schema: sink.schema(),
            node: Arc::new(BreakerNode {
                input: input.node,
                sink,
                source,
                ordered: false,
            }),
        })
    }

    /// The runner configuration of this worker's session.
    fn runner_conf(&self) -> RunnerConf {
        let config = self.worker.config.as_deref();
        let max_rows = config
            .and_then(|c| c.get(&ARROW_MAX_RECORDS_PER_BATCH))
            .and_then(krishiv_common::config::ConfigValue::as_int)
            .unwrap_or(10_000);
        let max_bytes = config
            .and_then(|c| c.get_unregistered(MAX_BYTES_PER_BATCH))
            .and_then(|v| v.trim().parse::<i64>().ok())
            .unwrap_or(0);
        let mut settings = std::collections::BTreeMap::new();
        if let Some(c) = config {
            for (k, v) in c.explicit_registered() {
                settings.insert(k.to_owned(), v);
            }
            for (k, v) in c.explicit_unregistered() {
                settings.insert(k.to_owned(), v.to_owned());
            }
        }
        RunnerConf {
            max_records_per_batch: usize::try_from(max_rows).unwrap_or(0),
            max_bytes_per_batch: usize::try_from(max_bytes).unwrap_or(0),
            settings,
        }
    }

    /// The prepared relation call site of `spec`.
    fn prepare_relation(
        &self,
        udf: &PythonUdfRef,
        spec: &RelationCallSpec,
    ) -> Result<Arc<dyn PreparedRelationUdf>> {
        let runtime = self.worker.python.as_ref().ok_or_else(|| {
            KrishivError::from(KrishivUdfNotAvailable {
                name: udf.name.clone(),
                version: udf.version.to_string(),
            })
        })?;
        runtime.prepare_relation(spec)
    }

    /// The lane of a map, grouped or cogrouped node, and the node's schema.
    fn relation_lane(
        &self,
        input: &SchemaRef,
        plan: &PythonRelationPlan,
        key_count: usize,
        conf: &RunnerConf,
    ) -> Result<(Vec<RelationLane>, SchemaRef)> {
        let canonical: Vec<Field> = plan
            .output
            .iter()
            .map(|c| Field::new(&c.name, c.data_type.to_arrow(), c.nullable))
            .collect();
        let client: Vec<ArrowType> = canonical
            .iter()
            .map(|f| client_arrow_type(f.data_type()))
            .collect();
        let client_fields: Vec<Field> = canonical
            .iter()
            .zip(&client)
            .map(|(f, t)| Field::new(f.name(), t.clone(), f.is_nullable()))
            .collect();
        let mut sides = Vec::with_capacity(plan.sides.len());
        for side in &plan.sides {
            let columns = side
                .columns
                .iter()
                .map(|&c| {
                    let field = input
                        .fields()
                        .get(c)
                        .ok_or_else(|| internal(format!("frame column {c} out of range")))?;
                    Ok(Arc::new(InputColumn {
                        index: c,
                        data_type: field.data_type().clone(),
                        nullable: field.is_nullable(),
                    }) as Arc<dyn ScalarExpr>)
                })
                .collect::<Result<Vec<_>>>()?;
            sides.push(columns);
        }
        let spec = RelationCallSpec {
            name: plan.udf.name.clone(),
            version: plan.udf.version,
            eval_type: plan.udf.eval_type,
            thread_safe: plan.udf.thread_safe,
            input_types: plan.sides.iter().map(|s| s.types.clone()).collect(),
            input_names: plan.sides.iter().map(|s| s.names.clone()).collect(),
            key_count,
            arity: plan
                .sides
                .first()
                .map_or(0, |s| s.columns.len().saturating_sub(key_count)),
            output_names: plan.output.iter().map(|c| c.name.clone()).collect(),
            output_types: plan.output.iter().map(|c| c.data_type.clone()).collect(),
            output_arrow: Arc::new(Schema::new(client_fields)),
            conf: conf.clone(),
        };
        let udf = self.prepare_relation(&plan.udf, &spec)?;
        let lane = RelationLane {
            udf,
            sides,
            names: plan.sides.iter().map(|s| s.names.clone()).collect(),
            client,
        };
        Ok((vec![lane], Arc::new(Schema::new(canonical))))
    }

    /// One lane per call of a Python aggregate, and the node's schema: the
    /// key columns, then one column per call.
    fn aggregate_lanes(
        &self,
        input: &SchemaRef,
        layout: &[AttrId],
        keys: &[usize],
        calls: &[PythonCallPlan],
        conf: &RunnerConf,
    ) -> Result<(Vec<RelationLane>, SchemaRef)> {
        let values = Values {
            env: &self.env,
            subqueries: &[],
        };
        let mut fields: Vec<Field> = keys
            .iter()
            .map(|&k| {
                input
                    .fields()
                    .get(k)
                    .map(|f| (**f).clone())
                    .ok_or_else(|| internal(format!("key column {k} out of range")))
            })
            .collect::<Result<_>>()?;
        let mut lanes = Vec::with_capacity(calls.len());
        for (i, c) in calls.iter().enumerate() {
            let ExprNode::Resolved(ResolvedExpr::Function(f)) = &c.call.node else {
                return Err(internal(
                    "a Python aggregate call is not a function".to_owned(),
                ));
            };
            let Some(udf) = PythonUdfRef::of(&f.id) else {
                return Err(internal(
                    "a Python aggregate call is not a Python UDF".to_owned(),
                ));
            };
            let args: Vec<&Expr> = f.args.iter().map(AsRef::as_ref).collect();
            let args = compile_all(&args, layout, input, &values)?;
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
            let name = format!("pythonUDAF{i}");
            let arg_names: Vec<String> = (0..args.len()).map(|j| format!("_{j}")).collect();
            let spec = RelationCallSpec {
                name: udf.name.clone(),
                version: udf.version,
                eval_type: udf.eval_type,
                thread_safe: udf.thread_safe,
                arity: arg_types.len(),
                input_types: vec![arg_types],
                input_names: vec![arg_names.clone()],
                key_count: 0,
                output_names: vec![name.clone()],
                output_types: vec![f.data_type.clone()],
                output_arrow: Arc::new(Schema::new(vec![Field::new(&name, client.clone(), true)])),
                conf: conf.clone(),
            };
            let prepared = self.prepare_relation(&udf, &spec)?;
            fields.push(Field::new(&name, canonical, true));
            lanes.push(RelationLane {
                udf: prepared,
                names: vec![arg_names],
                sides: vec![
                    args.into_iter()
                        .map(|e| Arc::new(Compiled(e)) as Arc<dyn ScalarExpr>)
                        .collect(),
                ],
                client: vec![client],
            });
        }
        Ok((lanes, Arc::new(Schema::new(fields))))
    }
}
