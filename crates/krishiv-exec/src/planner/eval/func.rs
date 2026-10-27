// SPDX-License-Identifier: Apache-2.0
//! Registry functions, bound to their kernels by the built-in function
//! registry's execution binding (`krishiv_functions::scalar`, P2-T153).

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::DataType as ArrowType;
use krishiv_functions::scalar::{Arg, BindArg, BindOptions, ScalarFunction, bind_scalar};
use krishiv_kernels::cast::FixedOffset;
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
            _ => Ok(BindArg::Column {
                ty: compiled.data_type().clone(),
                nullable: compiled.nullable(),
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    let out = f.data_type.to_arrow();
    // Only an offset zone is known without the tz database (P2-T021b).
    let zoned = |t: &ArrowType| matches!(t, ArrowType::Timestamp(_, Some(_)));
    // Functions whose result reads the session zone as an instant: with a
    // region session zone there are no rules to read it with. (`to_date`
    // over strings and dates, `to_timestamp_ntz` and `date_format` over
    // dates, strings and zone-less values round-trip through the zone, so
    // they run on the UTC fallback below.)
    let reads_zone = matches!(
        name,
        "to_timestamp"
            | "to_timestamp_ltz"
            | "unix_timestamp"
            | "to_unix_timestamp"
            | "from_unixtime"
            | "current_timezone"
            | "current_date"
            | "curdate"
            | "localtimestamp"
    );
    let zone = match input.env.zone {
        Some(offset) => FixedOffset(offset),
        None if zoned(&out) || bind_args.iter().any(|a| zoned(&a.data_type())) || reads_zone => {
            return Err(unsupported(
                "functions of timestamps in a session time zone other than a fixed offset",
            ));
        }
        None => FixedOffset(0),
    };
    // The zone id Spark normalises the offset to (`UTC`, `+05:30`); a
    // region session zone reaches here only for functions that do not read
    // it, which see the UTC fallback.
    let zone_name: Arc<str> = match input.env.zone {
        None | Some(0) => Arc::from("UTC"),
        Some(offset) => {
            let (sign, abs) = if offset < 0 {
                ("-", offset.unsigned_abs())
            } else {
                ("+", offset.unsigned_abs())
            };
            Arc::from(format!("{sign}{:02}:{:02}", abs / 3600, abs / 60 % 60))
        }
    };
    let opts = BindOptions::new()
        .with_ansi(input.env.ansi)
        .with_kernel(input.env.kernels)
        .with_time_zone(Arc::new(zone))
        .with_time_zone_name(zone_name)
        .with_query_start(input.env.query_start_micros)
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
