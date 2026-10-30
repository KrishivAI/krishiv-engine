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
use super::{Input, PhysExpr, Value, literal_array, rendered_zone, unsupported_expr};
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

/// The zone letters of a datetime pattern: region ids (`V`), names (`z`),
/// offsets (`X`, `x`, `Z`) and localized offsets (`O`).
const PATTERN_ZONE_LETTERS: &[char] = &['V', 'z', 'Z', 'X', 'x', 'O'];

/// The text of a one-row string literal; `None` for NULL or another type.
fn literal_text(a: &arrow_array::ArrayRef) -> Option<String> {
    use arrow_array::cast::AsArray as _;
    if a.len() != 1 || a.is_null(0) {
        return None;
    }
    match a.data_type() {
        ArrowType::Utf8 => Some(a.as_string::<i32>().value(0).to_owned()),
        ArrowType::LargeUtf8 => Some(a.as_string::<i64>().value(0).to_owned()),
        ArrowType::Utf8View => Some(a.as_string_view().value(0).to_owned()),
        _ => None,
    }
}

fn is_string(t: &ArrowType) -> bool {
    matches!(
        t,
        ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View
    )
}

/// Whether the pattern argument can name a zone: a column or any literal
/// with a zone letter (a quoted letter is text, but refusing it is safe).
/// A NULL literal parses nothing.
fn pattern_may_use_zone(arg: Option<&BindArg>) -> bool {
    match arg {
        None => false,
        Some(BindArg::Literal(a)) => literal_text(a)
            .is_some_and(|text| text.chars().any(|c| PATTERN_ZONE_LETTERS.contains(&c))),
        // A column, or any other kind of argument: its zone use is unknown.
        Some(_) => true,
    }
}

/// Calls that still read the region session zone although they are not in
/// `reads_zone`: they run on the UTC fallback, which is only right while no
/// zone is parsed or printed. Refused under a region zone: `date_format` of
/// a string (it may carry an offset) or of a date / zone-less timestamp with
/// a zone-bearing or dynamic pattern, and `to_date` / `try_to_date` of a
/// string with such a pattern. Zone-free paths stay allowed.
fn reads_region_zone(name: &str, args: &[BindArg]) -> bool {
    let first = args.first().map(BindArg::data_type);
    match name {
        "date_format" => first.as_ref().is_some_and(is_string) || pattern_may_use_zone(args.get(1)),
        "to_date" | "try_to_date" => {
            first.as_ref().is_some_and(is_string) && pattern_may_use_zone(args.get(1))
        }
        _ => false,
    }
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
    // `current_timezone()` returns the configured string; the patterns print
    // the zone id Spark normalises it to. A region session zone reaches the
    // patterns only for functions that do not read it, which see the UTC
    // fallback.
    let configured: Arc<str> = match &input.env.zone_name {
        Some(name) => Arc::clone(name),
        None => Arc::from(rendered_zone(input.env.zone)),
    };
    let zone_id: Arc<str> = Arc::from(krishiv_kernels::datetime::zone_id(&configured));
    let opts = BindOptions::new()
        .with_ansi(input.env.ansi)
        .with_kernel(input.env.kernels)
        .with_time_zone(Arc::new(zone))
        .with_time_zone_name(zone_id)
        .with_configured_time_zone(configured)
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
    // After the binding, so an invalid pattern still fails as Spark's does
    // whatever the zone.
    if input.env.zone.is_none() && reads_region_zone(name, &bind_args) {
        return Err(unsupported(
            "functions of timestamps in a session time zone other than a fixed offset",
        ));
    }
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
