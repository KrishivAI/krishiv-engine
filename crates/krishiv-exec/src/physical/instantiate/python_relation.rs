// SPDX-License-Identifier: Apache-2.0
//! Instantiation of the Python relation nodes (design P3-T085 §3.1-§3.2):
//! each node's UDFs prepared by the worker's [`UdfRuntime`] as relation call
//! sites, and a [`RelationSink`] breaker over the input.
//!
//! [`UdfRuntime`]: crate::python_eval::UdfRuntime

use super::*;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field};
use krishiv_common::config::ARROW_MAX_RECORDS_PER_BATCH;
use krishiv_common::error::{KrishivError, KrishivUdfNotAvailable};
use krishiv_plan::{ExprNode, PythonUdfRef, ResolvedExpr};
use krishiv_types::arrow::client_arrow_type;

use crate::operators::ScalarExpr;
use crate::physical::{PythonCallPlan, PythonRelationPlan, PythonUdtfPlan};
use crate::python_relation::{
    DEFAULT_WINDOW, InputColumn, PreparedRelationUdf, RelationCallSpec, RelationLane,
    RelationLimits, RelationShape, RelationSink, RunnerConf,
};

/// Spark's `spark.sql.execution.arrow.maxBytesPerBatch`, read as the
/// session wrote it (Krishiv registers no key for it yet): no limit unless
/// set to other than Spark's default.
const MAX_BYTES_PER_BATCH: &str = "spark.sql.execution.arrow.maxBytesPerBatch";

/// Spark's runner conf entry naming each window UDF's frame kind,
/// `bounded` or `unbounded` (`ArrowWindowPythonEvaluatorFactory.scala` L72
/// at v4.2.0); the worker reads it (`worker.py` L877-879).
const WINDOW_BOUND_TYPES: &str = "window_bound_types";

impl Instantiation<'_> {
    /// A Python relation node: a breaker whose sink feeds the node's tasks
    /// and whose source yields what they returned.
    pub(super) fn python_relation(&mut self, node: &PhysNode) -> Result<Built> {
        let (input, shape) = match node {
            PhysNode::PythonMap { input, .. } | PhysNode::PythonUdtf { input, .. } => {
                (input, RelationShape::Map)
            }
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
            PhysNode::PythonAggregate {
                input,
                keys,
                window: Some(_),
                ..
            } => (
                input,
                RelationShape::Aligned {
                    keys: Some(keys.clone()),
                },
            ),
            PhysNode::PythonAggregate {
                input,
                keys,
                key_values,
                window: None,
                ..
            } => (
                input,
                RelationShape::Aggregate {
                    keys: keys.clone(),
                    key_values: key_values.clone(),
                },
            ),
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
                    RelationShape::Map
                    | RelationShape::Aggregate { .. }
                    | RelationShape::Aligned { .. } => 0,
                };
                self.relation_lane(&input.schema, udf, keys, &conf)?
            }
            PhysNode::PythonUdtf { udf, call, .. } => {
                self.udtf_lane(&input.schema, udf, call, &conf)?
            }
            PhysNode::PythonAggregate {
                layout,
                key_values,
                calls,
                window: None,
                ..
            } => self.call_lanes(&input.schema, layout, key_values, calls, &conf, None, None)?,
            PhysNode::PythonAggregate {
                layout,
                calls,
                window: Some(bounded),
                ..
            } => {
                let every: Vec<usize> = (0..input.schema.fields().len()).collect();
                self.call_lanes(
                    &input.schema,
                    layout,
                    &every,
                    calls,
                    &conf,
                    None,
                    Some(bounded),
                )?
            }
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
            // Spark's default, `Int.MaxValue`, is no limit
            // (PythonArrowInput.scala:227 at v4.2.0).
            .filter(|&v| v != i64::from(i32::MAX))
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
            udtf: None,
        };
        let udf = self.prepare_relation(&plan.udf, &spec)?;
        let lane = RelationLane {
            name: plan.udf.name.clone(),
            udf,
            sides,
            names: plan.sides.iter().map(|s| s.names.clone()).collect(),
            client,
        };
        Ok((vec![lane], Arc::new(Schema::new(canonical))))
    }

    /// The lane of a Python UDTF and its schema: the frames are the call's
    /// arguments in call order (the `TABLE` argument as one struct column
    /// of its row), then the carried columns; the result is the carried
    /// columns, then the UDTF's.
    fn udtf_lane(
        &self,
        input: &SchemaRef,
        plan: &PythonRelationPlan,
        call: &PythonUdtfPlan,
        conf: &RunnerConf,
    ) -> Result<(Vec<RelationLane>, SchemaRef)> {
        let column = |c: usize| -> Result<Arc<dyn ScalarExpr>> {
            let field = input
                .fields()
                .get(c)
                .ok_or_else(|| internal(format!("frame column {c} out of range")))?;
            Ok(Arc::new(InputColumn {
                index: c,
                data_type: field.data_type().clone(),
                nullable: field.is_nullable(),
            }))
        };
        let side = plan
            .sides
            .first()
            .ok_or_else(|| internal("a Python UDTF without its argument side".to_owned()))?;
        let mut frame: Vec<Arc<dyn ScalarExpr>> = Vec::with_capacity(call.arg_names.len());
        let mut names = Vec::with_capacity(call.arg_names.len());
        let mut types = Vec::with_capacity(call.arg_names.len());
        let mut scalars = side.columns.iter().zip(&side.types);
        for i in 0..call.arg_names.len() {
            match &call.table {
                Some(t) if t.position == i => {
                    let mut fields = Vec::with_capacity(t.columns.len());
                    let mut spark = Vec::with_capacity(t.columns.len());
                    for (&c, name) in t.columns.iter().zip(&t.names) {
                        let field = input
                            .fields()
                            .get(c)
                            .ok_or_else(|| internal(format!("table column {c} out of range")))?;
                        fields.push(Field::new(
                            name,
                            field.data_type().clone(),
                            field.is_nullable(),
                        ));
                        spark.push(krishiv_types::StructField::new(
                            name.clone(),
                            krishiv_types::DataType::from_arrow(field.data_type()).map_err(
                                |e| internal(format!("a TABLE argument column's type: {e}")),
                            )?,
                            field.is_nullable(),
                        ));
                    }
                    frame.push(Arc::new(StructOf {
                        columns: t.columns.clone(),
                        fields: fields.into(),
                    }));
                    types.push(krishiv_types::DataType::Struct(
                        krishiv_types::StructType::new(spark),
                    ));
                }
                _ => {
                    let (&c, t) = scalars
                        .next()
                        .ok_or_else(|| internal("Python UDTF argument count".to_owned()))?;
                    frame.push(column(c)?);
                    types.push(t.clone());
                }
            }
            names.push(format!("_{i}"));
        }
        let args = frame.len();
        for (j, &c) in call.carried.iter().enumerate() {
            frame.push(column(c)?);
            let field = input
                .fields()
                .get(c)
                .ok_or_else(|| internal(format!("carried column {c} out of range")))?;
            types.push(
                krishiv_types::DataType::from_arrow(field.data_type())
                    .map_err(|e| internal(format!("a carried column's type: {e}")))?,
            );
            names.push(format!("_carried{j}"));
        }
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
        let udtf = crate::python_udtf::UdtfCall {
            args: call
                .arg_names
                .iter()
                .enumerate()
                .map(|(offset, name)| crate::python_udtf::UdtfArg {
                    offset,
                    name: name.clone(),
                })
                .collect(),
            partition_child_indexes: call
                .table
                .as_ref()
                .map(|t| t.partition_child_indexes.clone())
                .unwrap_or_default(),
            table_arg_offsets: call.table.iter().map(|t| t.position).collect(),
            analyze_result: call.analyzed.as_deref().map(Arc::from),
            passthrough: (args..args + call.carried.len()).collect(),
        };
        let spec = RelationCallSpec {
            name: plan.udf.name.clone(),
            version: plan.udf.version,
            eval_type: plan.udf.eval_type,
            thread_safe: plan.udf.thread_safe,
            input_types: vec![types],
            input_names: vec![names.clone()],
            key_count: 0,
            arity: frame.len(),
            output_names: plan.output.iter().map(|c| c.name.clone()).collect(),
            output_types: plan.output.iter().map(|c| c.data_type.clone()).collect(),
            output_arrow: Arc::new(Schema::new(client_fields)),
            conf: conf.clone(),
            udtf: Some(udtf),
        };
        let udf = self.prepare_relation(&plan.udf, &spec)?;
        let lane = RelationLane {
            name: plan.udf.name.clone(),
            udf,
            sides: vec![frame],
            names: vec![names],
            client,
        };
        Ok((vec![lane], Arc::new(Schema::new(canonical))))
    }

    /// A `PythonEval` of scalar iterator calls (Spark's 204, 251): a breaker
    /// whose sink sends each driver's rows to one task per call and joins
    /// every row with the value its task yields for it, in order (Spark's
    /// `ArrowEvalPythonExec` queue), whatever batches the values come in.
    pub(super) fn python_scalar_iter(
        &mut self,
        node: &PhysNode,
        ordered: bool,
        range_ordered: bool,
    ) -> Result<Built> {
        let PhysNode::PythonEval {
            input,
            layout,
            calls,
        } = node
        else {
            return Err(internal("not a PythonEval node".to_owned()));
        };
        let mut input = self.node(input)?;
        if range_ordered {
            input.node = Arc::new(OrderedNode { inner: input.node });
        }
        let conf = self.runner_conf();
        let leading: Vec<usize> = (0..input.schema.fields().len()).collect();
        let (lanes, schema) = self.call_lanes(
            &input.schema,
            layout,
            &leading,
            calls,
            &conf,
            Some("pythonUDF"),
            None,
        )?;
        // The input's metadata kept, as a scalar `PythonEval`'s output does.
        let schema = Arc::new(Schema::new_with_metadata(
            schema.fields().clone(),
            input.schema.metadata().clone(),
        ));
        let sink = RelationSink::try_new(
            Arc::clone(&input.schema),
            RelationShape::Aligned { keys: None },
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
                ordered,
            }),
        })
    }

    /// One lane per call of a Python aggregate (or aligned node), and the
    /// node's schema: the input columns `leading` (an aggregate's keys,
    /// every column of an aligned node), then one column per call, named
    /// `<prefix><i>` (`pythonUDAF<i>` by default). For a window aggregate,
    /// `window` says per call whether its frame is bounded: Spark's runner
    /// conf entry `window_bound_types` (`ArrowWindowPythonEvaluatorFactory`
    /// L220-224 at v4.2.0), one UDF per lane here.
    #[expect(
        clippy::too_many_arguments,
        reason = "the node's parts, read off three node shapes"
    )]
    fn call_lanes(
        &self,
        input: &SchemaRef,
        layout: &[AttrId],
        leading: &[usize],
        calls: &[PythonCallPlan],
        conf: &RunnerConf,
        prefix: Option<&str>,
        window: Option<&[bool]>,
    ) -> Result<(Vec<RelationLane>, SchemaRef)> {
        if window.is_some_and(|w| w.len() != calls.len()) {
            return Err(internal(
                "a Python window aggregate's bounds do not match its calls".to_owned(),
            ));
        }
        let keys = leading;
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
            let Some(udf) = PythonUdfRef::of(&f.id, f.deterministic) else {
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
            let name = format!("{}{i}", prefix.unwrap_or("pythonUDAF"));
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
                udtf: None,
            };
            let spec = match window.and_then(|w| w.get(i)) {
                Some(&bounded) => {
                    let mut spec = spec;
                    spec.conf.settings.insert(
                        WINDOW_BOUND_TYPES.to_owned(),
                        if bounded { "bounded" } else { "unbounded" }.to_owned(),
                    );
                    spec
                }
                None => spec,
            };
            let prepared = self.prepare_relation(&udf, &spec)?;
            fields.push(Field::new(&name, canonical, true));
            lanes.push(RelationLane {
                name: udf.name.clone(),
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

/// A `TABLE` argument's row: a struct of input columns (Spark's
/// `CreateStruct(subquery.output)`).
#[derive(Debug)]
struct StructOf {
    columns: Vec<usize>,
    fields: arrow_schema::Fields,
}

impl ScalarExpr for StructOf {
    fn data_type(&self) -> ArrowType {
        ArrowType::Struct(self.fields.clone())
    }

    fn nullable(&self) -> bool {
        false
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        let arrays = self
            .columns
            .iter()
            .map(|&c| {
                batch
                    .columns()
                    .get(c)
                    .cloned()
                    .ok_or_else(|| internal(format!("table column {c} out of range")))
            })
            .collect::<Result<Vec<_>>>()?;
        let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
        let rows = RecordBatch::try_new_with_options(
            Arc::new(Schema::new(self.fields.clone())),
            arrays,
            &options,
        )
        .map_err(|e| internal(format!("a TABLE argument's row: {e}")))?;
        Ok(Arc::new(arrow_array::StructArray::from(rows)))
    }
}
