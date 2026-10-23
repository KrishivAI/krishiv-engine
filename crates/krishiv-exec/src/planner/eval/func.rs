// SPDX-License-Identifier: Apache-2.0
//! Registry functions, bound to their kernels by the built-in function
//! registry's execution binding (`krishiv_functions::scalar`, P2-T153).

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::DataType as ArrowType;
use krishiv_functions::scalar::{Arg, BindArg, BindOptions, ScalarFunction, bind_scalar};
use krishiv_kernels::cast::{FixedOffset, TimeZoneRules};
use krishiv_plan::{Expr, ExprNode, ResolvedFunction};

use super::strings::to_view;
use super::{Input, PhysExpr, Value, literal_array, unsupported_expr};
use crate::error::Result;
use crate::planner::build::{internal, unsupported};

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
            _ => Ok(BindArg::Column(compiled.data_type().clone())),
        })
        .collect::<Result<Vec<_>>>()?;
    let out = f.data_type.to_arrow();
    let zoned = |t: &ArrowType| matches!(t, ArrowType::Timestamp(_, Some(_)));
    let zone: Arc<dyn TimeZoneRules> = match &input.env.zone {
        Some(rules) => Arc::clone(rules),
        None if zoned(&out)
            || bind_args.iter().any(|a| zoned(&a.data_type()))
            || matches!(name, "current_date" | "curdate" | "localtimestamp") =>
        {
            return Err(unsupported(&format!(
                "functions of timestamps in the unknown session time zone `{}`",
                input.env.zone_name
            )));
        }
        None => Arc::new(FixedOffset(0)),
    };
    let opts = BindOptions::new()
        .with_ansi(input.env.ansi)
        .with_kernel(input.env.kernels)
        .with_time_zone(zone)
        .with_query_start(input.env.query_start_micros)
        .with_user(Arc::clone(&input.env.user));
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

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        let args = self
            .args
            .iter()
            .map(|a| {
                Ok(match a.eval(batch)? {
                    Value::Array(v) => Arg::Array(v),
                    Value::Scalar(v) => Arg::Scalar(v),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let out = self.function.invoke(&args, batch.num_rows())?;
        Ok(Value::Array(if self.view { to_view(&out)? } else { out }))
    }
}
