// SPDX-License-Identifier: Apache-2.0
//! Registry functions, bound to their kernels by the built-in function
//! registry's execution binding (`krishiv_functions::scalar`, P2-T153).

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::DataType as ArrowType;
use krishiv_functions::scalar::{Arg, BindArg, BindOptions, ScalarFunction, bind_scalar};
use krishiv_kernels::cast::FixedOffset;
use krishiv_kernels::map::DedupPolicy;
use krishiv_plan::{Expr, ExprNode, ResolvedFunction};
use krishiv_udf_sdk::ScalarUdf;

use super::strings::to_view;
use super::{Input, PhysExpr, Value, literal_array, unsupported_expr};
use crate::error::Result;
use crate::planner::build::{internal, unsupported};

/// Binds a session UDF call: wraps the implementation (resolved from the
/// worker by the caller) like a built-in binding, over the arguments the
/// analyzer cast.
pub(super) fn compile_udf(
    udf: &Arc<dyn ScalarUdf>,
    f: &ResolvedFunction,
    args: Vec<Arc<dyn PhysExpr>>,
    input: Input<'_>,
    _e: &Expr,
) -> Result<Arc<dyn PhysExpr>> {
    let bind_args = f
        .args
        .iter()
        .zip(&args)
        .map(|(ir, compiled)| match &ir.node {
            ExprNode::Literal(l) => literal_array(l).map(BindArg::Literal),
            _ => Ok(BindArg::Column {
                ty: compiled.data_type().clone(),
                nullable: compiled.nullable(),
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    let arg_types: Vec<ArrowType> = bind_args.iter().map(BindArg::data_type).collect();
    // `bind_udf` reports user faults as `FAILED_EXECUTE_UDF` and engine
    // faults as `INTERNAL_ERROR` (never `KRISHIV_UNSUPPORTED_FEATURE`,
    // which has no planning fallback), so there is nothing to map here.
    let function = krishiv_functions::scalar::bind_udf(
        udf,
        &arg_types,
        &f.data_type,
        f.nullable,
        input.env.ansi,
        input.env.zone,
    )?;
    let out = f.data_type.to_arrow();
    // The binding returns plain carriers; Spark's `STRING` and `BINARY`
    // are views here. Both derive from the analyzer's type, so they only
    // ever match or convert to a view (by construction, no mismatch).
    let view = matches!(
        (function.return_type(), &out),
        (ArrowType::Utf8, ArrowType::Utf8View) | (ArrowType::Binary, ArrowType::BinaryView)
    );
    Ok(Arc::new(Call {
        function,
        args,
        out,
        view,
        nullable: f.nullable,
    }))
}

pub(super) fn compile(
    name: &str,
    f: &ResolvedFunction,
    args: Vec<Arc<dyn PhysExpr>>,
    input: Input<'_>,
    e: &Expr,
) -> Result<Arc<dyn PhysExpr>> {
    let bind_args = f
        .args
        .iter()
        .zip(&args)
        .map(|(ir, compiled)| match &ir.node {
            ExprNode::Literal(l) => literal_array(l).map(BindArg::Literal),
            _ => Ok(BindArg::Column {
                ty: compiled.data_type().clone(),
                nullable: compiled.nullable(),
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    let out = f.data_type.to_arrow();
    // Only an offset zone is known without the tz database (P2-T021b).
    let zoned = |t: &ArrowType| matches!(t, ArrowType::Timestamp(_, Some(_)));
    let zone = match input.env.zone {
        Some(offset) => FixedOffset(offset),
        None if zoned(&out)
            || bind_args.iter().any(|a| zoned(&a.data_type()))
            || matches!(name, "current_date" | "curdate" | "localtimestamp") =>
        {
            return Err(unsupported(
                "functions of timestamps in a session time zone other than a fixed offset",
            ));
        }
        None => FixedOffset(0),
    };
    let opts = BindOptions::new()
        .with_ansi(input.env.ansi)
        .with_kernel(input.env.kernels)
        .with_time_zone(Arc::new(zone))
        .with_query_start(input.env.query_start_micros)
        .with_map_key_dedup_policy(if input.env.map_key_dedup_last_win {
            DedupPolicy::LastWin
        } else {
            DedupPolicy::Exception
        })
        .with_user(Arc::clone(&input.env.user))
        .with_seed_salt(Arc::clone(&input.env.seed_salt));
    let function = bind_scalar(name, &bind_args, &opts).map_err(|err| {
        if err.error_class() == krishiv_common::error::ErrorClass::KrishivUnsupportedFeature {
            unsupported_expr(e)
        } else {
            err
        }
    })?;
    // The binding returns plain carriers; Spark's `STRING` is a view here.
    let view = match (function.return_type(), &out) {
        (a, b) if a == b => false,
        (ArrowType::Utf8, ArrowType::Utf8View) | (ArrowType::Binary, ArrowType::BinaryView) => true,
        (a, b) => {
            return Err(internal(&format!(
                "{name} returns {a} where the analyzer typed {b}"
            )));
        }
    };
    Ok(Arc::new(Call {
        function,
        args,
        out,
        view,
        nullable: f.nullable,
    }))
}

/// A bound registry function over its arguments.
#[derive(Debug)]
struct Call {
    function: Arc<dyn ScalarFunction>,
    args: Vec<Arc<dyn PhysExpr>>,
    out: ArrowType,
    /// Convert the result to a view (`Utf8View`, `BinaryView`).
    view: bool,
    nullable: bool,
}

impl PhysExpr for Call {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let args = self
            .args
            .iter()
            .map(|a| {
                Ok(match a.eval(batch, partition)? {
                    Value::Array(v) => Arg::Array(v),
                    Value::Scalar(v) => Arg::Scalar(v),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let out = self
            .function
            .invoke_in(&args, batch.num_rows(), partition)?;
        Ok(Value::Array(if self.view { to_view(&out)? } else { out }))
    }
}
