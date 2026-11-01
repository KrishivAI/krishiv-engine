// SPDX-License-Identifier: Apache-2.0
//! Registry functions, bound to their kernels by the built-in function
//! registry's execution binding (`krishiv_functions::scalar`, P2-T153).

use std::sync::{Arc, OnceLock};

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, ListArray, RecordBatch, StructArray};
use arrow_schema::DataType as ArrowType;
use krishiv_functions::scalar::{Arg, BindArg, BindOptions, ScalarFunction, bind_scalar};
use krishiv_kernels::map::DedupPolicy;
use krishiv_kernels::string::CharsetOptions;
use krishiv_plan::{Expr, ExprNode, Literal, ResolvedFunction};
use krishiv_types::arrow::client_arrow_type;
use krishiv_udf_sdk::ScalarUdf;

use super::strings::to_view;
use super::{Input, PhysExpr, Value, literal_array, rendered_zone, unsupported_expr};
use crate::carrier::recarried_array;
use crate::error::Result;
use crate::planner::build::internal;

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
    let args = wide_args(args);
    let bind_args = f
        .args
        .iter()
        .zip(&args)
        .map(|(ir, compiled)| match &ir.node {
            ExprNode::Literal(l) => wide_literal(l).map(BindArg::Literal),
            _ => Ok(BindArg::Column {
                ty: client_arrow_type(compiled.data_type()),
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
    let canonical = f.data_type.to_arrow();
    let out = client_arrow_type(&canonical);
    // The binding returns plain carriers; Spark's `STRING` and `BINARY`
    // are views here. Both derive from the analyzer's type, so they only
    // ever match or convert to a view (by construction, no mismatch).
    let view = matches!(
        (function.return_type(), &out),
        (ArrowType::Utf8, ArrowType::Utf8View) | (ArrowType::Binary, ArrowType::BinaryView)
    );
    Ok(on_carrier(
        Arc::new(Call {
            function,
            args,
            out,
            view,
            relabel: false,
            nullable: f.nullable,
            folded: None,
            constant: false,
        }),
        canonical,
    ))
}

/// `e`'s values as `to`, which differs at most in decimal carriers. Functions
/// read and return `Decimal128`, as they do at every other edge (design P2-T160).
#[derive(Debug)]
struct Recarried {
    inner: Arc<dyn PhysExpr>,
    to: ArrowType,
}

impl PhysExpr for Recarried {
    fn data_type(&self) -> &ArrowType {
        &self.to
    }

    fn nullable(&self) -> bool {
        self.inner.nullable()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        Ok(match self.inner.eval(batch, partition)? {
            Value::Array(a) => Value::Array(recarried_array(&a, &self.to)?),
            Value::Scalar(a) => Value::Scalar(recarried_array(&a, &self.to)?),
        })
    }

    fn is_constant(&self) -> bool {
        self.inner.is_constant()
    }
}

fn on_carrier(e: Arc<dyn PhysExpr>, to: ArrowType) -> Arc<dyn PhysExpr> {
    if e.data_type() == &to {
        e
    } else {
        Arc::new(Recarried { inner: e, to })
    }
}

fn wide_args(args: Vec<Arc<dyn PhysExpr>>) -> Vec<Arc<dyn PhysExpr>> {
    args.into_iter()
        .map(|a| {
            let to = client_arrow_type(a.data_type());
            on_carrier(a, to)
        })
        .collect()
}

fn wide_literal(l: &Literal) -> Result<ArrayRef> {
    let a = literal_array(l)?;
    recarried_array(&a, &client_arrow_type(a.data_type()))
}

/// Whether `a` and `b` differ at most in the nullability of nested fields.
fn same_shape(a: &ArrowType, b: &ArrowType) -> bool {
    match (a, b) {
        (ArrowType::List(x), ArrowType::List(y)) => same_shape(x.data_type(), y.data_type()),
        (ArrowType::Struct(x), ArrowType::Struct(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|(p, q)| p.name() == q.name() && same_shape(p.data_type(), q.data_type()))
        }
        _ => a == b,
    }
}

/// `array` with the nested field flags of `to` (see [`same_shape`]).
fn relabeled(array: &ArrayRef, to: &ArrowType) -> Result<ArrayRef> {
    match to {
        ArrowType::List(f) if array.data_type() != to => {
            let list = array
                .as_list_opt::<i32>()
                .ok_or_else(|| internal("a relabelled list is not a list"))?;
            Ok(Arc::new(
                ListArray::try_new(
                    Arc::clone(f),
                    list.offsets().clone(),
                    relabeled(list.values(), f.data_type())?,
                    list.nulls().cloned(),
                )
                .map_err(|e| internal(&e.to_string()))?,
            ))
        }
        ArrowType::Struct(fields) if array.data_type() != to => {
            let s = array
                .as_struct_opt()
                .ok_or_else(|| internal("a relabelled struct is not a struct"))?;
            let columns = s
                .columns()
                .iter()
                .zip(fields.iter())
                .map(|(c, f)| relabeled(c, f.data_type()))
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(
                StructArray::try_new(fields.clone(), columns, s.nulls().cloned())
                    .map_err(|e| internal(&e.to_string()))?,
            ))
        }
        _ => Ok(Arc::clone(array)),
    }
}

pub(super) fn compile(
    name: &str,
    f: &ResolvedFunction,
    args: Vec<Arc<dyn PhysExpr>>,
    input: Input<'_>,
    e: &Expr,
) -> Result<Arc<dyn PhysExpr>> {
    let args = wide_args(args);
    let bind_args = f
        .args
        .iter()
        .zip(&args)
        .map(|(ir, compiled)| match &ir.node {
            ExprNode::Literal(l) => wide_literal(l).map(BindArg::Literal),
            _ => Ok(BindArg::Column {
                ty: client_arrow_type(compiled.data_type()),
                nullable: compiled.nullable(),
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    let canonical = f.data_type.to_arrow();
    let out = client_arrow_type(&canonical);
    // The session zone's rules; a function that never reads the zone sees
    // the same rules, so no per-function list of zone readers is needed.
    let zone = input.env.zone_rules()?;
    // `current_timezone()` returns the configured string; the patterns print
    // the zone id Spark normalises it to.
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
        .with_charset_options(CharsetOptions {
            java_charsets: input.env.legacy_java_charsets,
            replace_malformed: input.env.legacy_coding_error_action,
        })
        .with_user(Arc::clone(&input.env.user))
        .with_seed_salt(Arc::clone(&input.env.seed_salt));
    let args_constant = args.iter().all(|a| a.is_constant());
    let function = bind_scalar(name, &bind_args, &opts).map_err(|err| {
        if err.error_class() == krishiv_common::error::ErrorClass::KrishivUnsupportedFeature {
            unsupported_expr(e)
        } else {
            err
        }
    })?;
    // The binding returns plain carriers; Spark's `STRING` is a view here.
    let mut relabel = false;
    let view = match (function.return_type(), &out) {
        (a, b) if a == b => false,
        // The analyzer types a result nullable where the folded arguments
        // are not (a cast of a literal): the data is the same, so the
        // result takes the analyzer's field flags.
        (a, b) if same_shape(a, b) => {
            relabel = true;
            false
        }
        (ArrowType::Utf8, ArrowType::Utf8View) | (ArrowType::Binary, ArrowType::BinaryView) => true,
        (a, b) => {
            return Err(internal(&format!(
                "{name} returns {a} where the analyzer typed {b}"
            )));
        }
    };
    Ok(on_carrier(
        Arc::new(Call {
            function,
            args,
            out,
            view,
            relabel,
            nullable: f.nullable,
            // `aes_encrypt` of constants is folded once for the query, as
            // Spark's constant folding does: one IV for every row and batch
            // (a column argument draws an IV per row).
            folded: (name == "aes_encrypt" && args_constant).then(OnceLock::new),
            // A pure function of constants (`unhex('..')`) is one value for
            // every row, so an enclosing call over it folds too.
            constant: args_constant
                && (name == "aes_encrypt"
                    || krishiv_functions::scalar::is_constant_of_constants(name)),
        }),
        canonical,
    ))
}

/// A bound registry function over its arguments.
#[derive(Debug)]
struct Call {
    function: Arc<dyn ScalarFunction>,
    args: Vec<Arc<dyn PhysExpr>>,
    out: ArrowType,
    /// Convert the result to a view (`Utf8View`, `BinaryView`).
    view: bool,
    /// Give the result the analyzer's nested field flags.
    relabel: bool,
    nullable: bool,
    /// The one-row result of a call over constants that Spark folds once.
    folded: Option<OnceLock<ArrayRef>>,
    /// Whether the value is the same for every row (see `is_constant`).
    constant: bool,
}

impl PhysExpr for Call {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn is_constant(&self) -> bool {
        self.constant
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        if let Some(cell) = &self.folded {
            if let Some(row) = cell.get() {
                return Ok(Value::Scalar(Arc::clone(row)));
            }
            // Evaluated for one row; a failure is not remembered.
            let one = self.call(batch, partition, 1)?;
            let row = cell.get_or_init(|| one);
            return Ok(Value::Scalar(Arc::clone(row)));
        }
        if self.constant {
            // A pure call over constants is one value: a one-row scalar, so
            // an enclosing call sees every argument at the same length.
            return Ok(Value::Scalar(self.call(batch, partition, 1)?));
        }
        Ok(Value::Array(self.call(
            batch,
            partition,
            batch.num_rows(),
        )?))
    }
}

impl Call {
    /// The function's values for `rows` rows, in the result's carrier.
    fn call(&self, batch: &RecordBatch, partition: u32, rows: usize) -> Result<ArrayRef> {
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
        let out = self.function.invoke_in(&args, rows, partition)?;
        let out = if self.relabel {
            relabeled(&out, &self.out)?
        } else {
            out
        };
        if self.view { to_view(&out) } else { Ok(out) }
    }
}
