// SPDX-License-Identifier: Apache-2.0
//! Result types and specs of the aggregate functions (sum, avg, min/max, collect, percentile, listagg and the rest).

use super::*;

pub(super) fn bounded_decimal(precision: u32, scale: u32) -> Result<DataType, KrishivError> {
    Ok(DataType::Decimal(DecimalType::try_new(
        precision.min(38),
        scale.min(38),
    )?))
}

/// `sum`: `decimal(min(38, p + 10), s)`, `bigint` for integral input,
/// `double` for floating point.
pub(super) fn sum(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let t = match c.data_type(0) {
        Some(DataType::Decimal(d)) => bounded_decimal(
            u32::from(d.precision()).saturating_add(10),
            d.scale().into(),
        )?,
        Some(DataType::Timestamp | DataType::Double | DataType::Float) | None => DataType::Double,
        // An ANSI interval keeps its fields.
        Some(t) if is_ansi_interval(t) => t.clone(),
        Some(_) => DataType::Long,
    };
    Ok(ArgType::new(t, true))
}

/// `avg`: `decimal(min(38, p + 4), min(38, s + 4))`, an ANSI interval's
/// default fields, else `double`.
pub(super) fn avg(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let t = match c.data_type(0) {
        Some(DataType::Decimal(d)) => bounded_decimal(
            u32::from(d.precision()).saturating_add(4),
            u32::from(d.scale()).saturating_add(4),
        )?,
        Some(t) if is_ansi_interval(t) => default_interval(t),
        _ => DataType::Double,
    };
    Ok(ArgType::new(t, true))
}

/// The default fields of the ANSI interval family of `t`.
pub(super) fn default_interval(t: &DataType) -> DataType {
    if matches!(t, DataType::YearMonthInterval { .. }) {
        DataType::DEFAULT_YEAR_MONTH_INTERVAL
    } else {
        DataType::DEFAULT_DAY_TIME_INTERVAL
    }
}

pub(super) fn count(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Long, false))
}

pub(super) fn min_max(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType {
        data_type: c.data_type(0).cloned(),
        nullable: true,
    })
}

/// Whether Spark orders `t` (`RowOrdering.isOrderable`): atomic types, arrays
/// of an orderable element, structs of orderable fields (no maps, variants).
pub(in crate::registry) fn spark_orderable(t: &DataType) -> bool {
    match t {
        DataType::Map(_)
        | DataType::CalendarInterval
        | DataType::Geometry { .. }
        | DataType::Geography { .. }
        | DataType::Variant => false,
        DataType::Array(a) => spark_orderable(a.element_type()),
        DataType::Struct(s) => s.fields().iter().all(|f| spark_orderable(f.data_type())),
        _ => true,
    }
}

/// `min`/`max` over `function`'s argument: its type, which must order. A
/// struct orders field by field and an array element by element (recursively). One Spark cannot order (a
/// map field) is `DATATYPE_MISMATCH.INVALID_ORDERING_TYPE`; one Spark orders
/// but the struct comparator does not take
/// ([`krishiv_kernels::r#struct::comparable`], the same predicate the
/// executor compares with: an interval field or element) is
/// `KRISHIV_UNSUPPORTED_FEATURE`.
pub(super) fn ordered_min_max(c: &CallTypes<'_>, function: &str) -> Result<ArgType, KrishivError> {
    if let Some(t) = c.data_type(0) {
        if !spark_orderable(t) {
            return Err(DatatypeMismatchInvalidOrderingType {
                sql_expr: format!("\"{function}({})\"", c.sql(0).unwrap_or_default()),
                function_name: format!("`{function}`"),
                data_type: format!("\"{}\"", t.sql()),
            }
            .into());
        }
        if matches!(t, DataType::Struct(_) | DataType::Array(_))
            && !krishiv_kernels::r#struct::comparable(&t.to_arrow())
        {
            return Err(KrishivUnsupportedFeature {
                feature: format!(
                    "{function} over {}, a struct or array with a field the struct comparator does not take",
                    t.sql()
                ),
            }
            .into());
        }
    }
    min_max(c)
}

/// `min`: as [`ordered_min_max`].
pub(super) fn min(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    ordered_min_max(c, "min")
}

/// `max`: as [`ordered_min_max`].
pub(super) fn max(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    ordered_min_max(c, "max")
}

/// `first(x [, ignoreNulls])`: the type of `x`; the flag must be a boolean
/// literal (Spark matches `Literal(b, BooleanType)`, so `1 = 1` is rejected).
pub(super) fn pick(c: &CallTypes<'_>, function: &str) -> Result<ArgType, KrishivError> {
    if c.args().len() > 1
        && !(c.is_literal(1) && matches!(c.constant(1), Some(Constant::Boolean(_))))
    {
        return Err(LegacyErrorTemp1210 {
            func_name: function.to_owned(),
        }
        .into());
    }
    min_max(c)
}

pub(super) fn first(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    pick(c, "first")
}

pub(super) fn last(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    pick(c, "last")
}

pub(super) fn any_value(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    pick(c, "any_value")
}

/// The moment aggregates (`CentralMomentAgg`, `Covariance`, `Corr`): every
/// argument is cast to `DOUBLE`; the result is a nullable `DOUBLE`.
pub(super) fn moment(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Double, true))
}

/// `bool_and`, `bool_or` and their aliases: `boolean`, NULL for no non-NULL
/// input.
pub(super) fn nullable_boolean(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Boolean, true))
}

/// `max_by(x, y)`, `min_by(x, y)`: the type of `x`. The three-argument
/// form (`max_by(x, y, k)`) returns an array of up to `k` values, which
/// this engine has no type for yet.
pub(super) fn by(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if c.args().len() > 2 {
        return Err(KrishivUnsupportedFeature {
            feature: "max_by and min_by with a third argument (they return an array)".to_owned(),
        }
        .into());
    }
    min_max(c)
}

/// `collect_list(x)`, `collect_set(x)`, `array_agg(x)`: an array of the
/// input's type, never NULL (an empty group is `[]`; NULL elements are
/// skipped by the executor). Spark names `array_agg`'s column
/// `collect_list(...)`, so the analyzer resolves it as `collect_list`
/// (see `krishiv_plan::analyzer`).
pub(super) fn collect(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let element = c.data_type(0).cloned().unwrap_or(DataType::Null);
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(element, false).into()),
        false,
    ))
}

/// Whether `t` is or contains a map at any depth.
pub(super) fn has_map(t: &DataType) -> bool {
    match t {
        DataType::Map(_) => true,
        DataType::Array(a) => has_map(a.element_type()),
        DataType::Struct(s) => s.fields().iter().any(|f| has_map(f.data_type())),
        _ => false,
    }
}

/// `collect_set(x)`: as [`collect`], but Spark refuses an input that
/// contains a map at any depth (`DATATYPE_MISMATCH.UNSUPPORTED_INPUT_TYPE`),
/// as a map has no equality to deduplicate by.
pub(super) fn collect_set(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if let Some(t) = c.data_type(0).filter(|t| has_map(t)) {
        return Err(DatatypeMismatchUnsupportedInputType {
            sql_expr: format!("\"collect_set({})\"", c.sql(0).unwrap_or_default()),
            function_name: "`collect_set`".to_owned(),
            data_type: format!("\"{}\"", t.sql()),
        }
        .into());
    }
    collect(c)
}

/// `vector_sum(x)` / `vector_avg(x)`: element-wise aggregates over
/// `ARRAY<FLOAT>` only. Any other type is Spark's
/// `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` (no implicit cast is
/// declared, so the check below sees the written type, as `ntile`'s
/// bucket check does). The result is an array of `float`, NULL for no
/// non-NULL vector.
pub(super) fn vector(c: &CallTypes<'_>, name: &str) -> Result<ArgType, KrishivError> {
    let is_floats = matches!(
        c.data_type(0),
        Some(DataType::Array(a)) if *a.element_type() == DataType::Float
    );
    if !is_floats {
        let input = c
            .data_type(0)
            .map_or_else(|| "VOID".to_owned(), DataType::sql);
        let sql = |i: usize| c.sql(i).unwrap_or_default();
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: format!("\"{name}({})\"", sql(0)),
            param_index: "first".to_owned(),
            required_type: "\"ARRAY<FLOAT>\"".to_owned(),
            input_sql: format!("\"{}\"", sql(0)),
            input_type: format!("\"{input}\""),
        }
        .into());
    }
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(DataType::Float, false).into()),
        true,
    ))
}

pub(super) fn vector_sum(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    vector(c, "vector_sum")
}

pub(super) fn vector_avg(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    vector(c, "vector_avg")
}

/// `bit_and`, `bit_or`, `bit_xor` over `TINYINT` .. `BIGINT`: the result
/// is the input type (NULL for no input). Any other type is Spark's
/// `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` (no implicit cast is
/// declared, so the check below sees the written type, as `ntile`'s
/// bucket check and `vector` do). The analyzer calls this for aggregate
/// bit operations only; the scalar bitwise operators live in
/// `bitwise.rs`.
pub(super) fn bit(c: &CallTypes<'_>, name: &str) -> Result<ArgType, KrishivError> {
    let integral = matches!(
        c.data_type(0),
        Some(DataType::Byte | DataType::Short | DataType::Integer | DataType::Long,)
    );
    if !integral {
        let input = c
            .data_type(0)
            .map_or_else(|| "VOID".to_owned(), DataType::sql);
        let sql = |i: usize| c.sql(i).unwrap_or_default();
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: format!("\"{name}({})\"", sql(0)),
            param_index: "first".to_owned(),
            required_type: "\"BYTE\", \"SHORT\", \"INT\" or \"BIGINT\"".to_owned(),
            input_sql: format!("\"{}\"", sql(0)),
            input_type: format!("\"{input}\""),
        }
        .into());
    }
    Ok(ArgType::new(
        c.data_type(0).cloned().unwrap_or(DataType::Integer),
        true,
    ))
}

pub(super) fn bit_and(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    bit(c, "bit_and")
}

pub(super) fn bit_or(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    bit(c, "bit_or")
}

pub(super) fn bit_xor(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    bit(c, "bit_xor")
}

/// `mode(col)`: the most frequent non-NULL value, of the input's type.
/// The analyzer types the `mode(col, deterministic)` and
/// `mode() WITHIN GROUP (ORDER BY col)` forms itself (this rule only
/// types the plain call, where the argument is the value); the executor
/// breaks ties as Spark does.
pub(super) fn mode(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    min_max(c)
}

/// `regr_count(y, x)`: the number of non-NULL pairs, a non-NULL `BIGINT`.
pub(super) fn regr_count(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Long, false))
}

/// `listagg(expr[, delimiter])` / `string_agg`: the concatenation of the
/// non-NULL values, of the value's type (`STRING` or `BINARY`; NULL for
/// no input). The delimiter is implicitly cast to the value's type by the
/// declared inputs; a non-foldable delimiter is Spark's
/// `NON_FOLDABLE_INPUT`, and one that cannot cast to the value's type (a
/// string delimiter for binary values) Spark's `DATA_DIFF_TYPES`, after
/// Spark's `checkForSameTypeInputExpr`. A NULL delimiter reads as empty;
/// the analyzer fills a missing one with NULL, so the call always shows
/// it.
pub(super) fn listagg(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let sql = |i: usize| c.sql(i).unwrap_or_default();
    // The delimiter must fold (a column does not); a missing one is the
    // NULL default. A delimiter that raises while folding raises here.
    // (`is_plain_literal` covers the literals the constant folder has no
    // value for, like binary ones.)
    if c.args().len() > 1 && !c.is_plain_literal(1) && c.constant(1).is_none() {
        let input = c
            .data_type(1)
            .map_or_else(|| "VOID".to_owned(), DataType::sql);
        return Err(DatatypeMismatchNonFoldableInput {
            sql_expr: format!("\"listagg({}, {})\"", sql(0), sql(1)),
            input_name: "`delimiter`".to_owned(),
            input_type: format!("\"{input}\""),
            input_expr: format!("\"{}\"", sql(1)),
        }
        .into());
    }
    if c.args().len() > 1
        && let Some(Constant::Error(err)) = c.constant(1)
    {
        return Err(err.clone());
    }
    let (Some(value), mut delimiter) = (c.data_type(0), c.data_type(1).cloned()) else {
        return typed(DataType::String, c);
    };
    // A written NULL delimiter stays NULL (it reads as empty): the
    // implicit cast must not give it a type for the check below. This
    // matches the call as written (an explicit `CAST(NULL AS STRING)`
    // keeps its type, as Spark's check sees it; the SQL layer fills a
    // missing delimiter with NULL, which shows the same way).
    if c.sql(1).is_some_and(|s| s == "NULL") {
        delimiter = Some(DataType::Null);
    }
    // Spark renders the failing delimiter by value, not as SQL: the
    // golden quotes `listagg(c1, , )` for a `', '` delimiter (and `NULL`
    // for a NULL one, as SQL renders it too).
    let delim = match &delimiter {
        None | Some(DataType::Null) => "NULL".to_owned(),
        Some(DataType::String) if matches!(c.constant(1), Some(Constant::String(_))) => {
            match c.constant(1) {
                Some(Constant::String(s)) => s.clone(),
                _ => sql(1),
            }
        }
        Some(_) => sql(1),
    };
    if !matches!(delimiter, None | Some(DataType::Null)) && Some(value) != c.data_type(1) {
        let delim_type = c.data_type(1).cloned().unwrap_or(DataType::Null);
        return Err(DatatypeMismatchDataDiffTypes {
            sql_expr: format!("\"listagg({}, {delim})\"", sql(0)),
            function_name: "`listagg`".to_owned(),
            data_type: format!("(\"{}\" or \"{}\")", value.sql(), delim_type.sql()),
        }
        .into());
    }
    // NULL for no input: always nullable.
    Ok(ArgType::new(value.clone(), true))
}

/// `try_sum(x)`: as `sum` (NULL on overflow, ANSI on or off).
pub(super) fn try_sum(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    sum(c)
}

/// `try_avg(x)`: as `avg` (NULL on overflow, ANSI on or off).
pub(super) fn try_avg(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    avg(c)
}

/// Spark's `PercentileBase.dataType`: `double`, or an ANSI interval child's
/// default fields; an array of it for an array percentage.
pub fn exact_percentile_result(child: Option<&DataType>, array: bool) -> DataType {
    let value = match child {
        Some(t) if is_ansi_interval(t) => default_interval(t),
        _ => DataType::Double,
    };
    if array {
        DataType::Array(ArrayType::new(value, false).into())
    } else {
        value
    }
}

/// Spark's `ApproximatePercentile.dataType`: the child's type, or an
/// array of it for an array percentage.
pub fn approx_percentile_result(child: &DataType, array: bool) -> DataType {
    if array {
        DataType::Array(ArrayType::new(child.clone(), false).into())
    } else {
        child.clone()
    }
}

/// Whether the percentage argument at `i` selects an array of
/// percentiles (Spark's `returnPercentileArray`).
pub(super) fn array_percentage(c: &CallTypes<'_>, i: usize) -> bool {
    matches!(c.data_type(i), Some(DataType::Array(_)))
}

/// `percentile`, `percentile_cont`, `percentile_disc` and `median` (which
/// is `percentile(child, 0.5)`): [`exact_percentile_result`].
pub(super) fn percentile(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(
        exact_percentile_result(c.data_type(0), array_percentage(c, 1)),
        true,
    ))
}

/// Spark's `Median`: `percentile(child, 0.5)`.
pub(super) fn median(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(
        exact_percentile_result(c.data_type(0), false),
        true,
    ))
}

/// Spark's `ApproximatePercentile.dataType`: the child's type, or an
/// array of it for an array percentage.
pub(super) fn percentile_approx(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let child = c.data_type(0).cloned().unwrap_or(DataType::Null);
    Ok(ArgType::new(
        approx_percentile_result(&child, array_percentage(c, 1)),
        true,
    ))
}

/// Spark's `PercentileCont` / `PercentileDisc` result over the user
/// form (the percentage is the only argument): `double`, or
/// `array<double>` for an array percentage. The analyzer retypes it from the
/// ordering key.
pub(super) fn percentile_ordered(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(
        exact_percentile_result(None, array_percentage(c, 0)),
        true,
    ))
}

/// `approx_count_distinct`: `bigint`, never NULL (an empty group is 0).
pub(super) fn approx_count_distinct(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Long, false))
}

pub(super) fn aggregates() -> Vec<Spec> {
    let numeric = || Inputs::Expects(vec![InputType::NumericOrTimestamp]);
    let boolean_arg = || Inputs::Expects(vec![exact(DataType::Boolean)]);
    let doubles = || Inputs::Expects(vec![exact(DataType::Double)]);
    let integrals = || {
        Inputs::Expects(vec![InputType::OneOf(vec![
            DataType::Byte,
            DataType::Short,
            DataType::Integer,
            DataType::Long,
        ])])
    };
    let string_or_bytes = || {
        Inputs::Expects(vec![
            InputType::OneOf(vec![DataType::String, DataType::Binary]),
            InputType::OneOf(vec![DataType::String, DataType::Binary, DataType::Null]),
        ])
    };
    vec![
        aggregate("count", Inputs::Any, count, Aggregate::Count),
        aggregate("sum", numeric(), sum, Aggregate::Sum),
        aggregate("avg", numeric(), avg, Aggregate::Avg),
        aggregate("mean", numeric(), avg, Aggregate::Avg),
        aggregate("min", Inputs::Any, min, Aggregate::Min),
        aggregate("max", Inputs::Any, max, Aggregate::Max),
        aggregate("first", Inputs::Any, first, Aggregate::First),
        aggregate("first_value", Inputs::Any, first, Aggregate::First),
        aggregate("last", Inputs::Any, last, Aggregate::Last),
        aggregate("last_value", Inputs::Any, last, Aggregate::Last),
        aggregate("any_value", Inputs::Any, any_value, Aggregate::AnyValue),
        aggregate("stddev", doubles(), moment, Aggregate::StddevSamp),
        aggregate("std", doubles(), moment, Aggregate::StddevSamp),
        aggregate("stddev_samp", doubles(), moment, Aggregate::StddevSamp),
        aggregate("stddev_pop", doubles(), moment, Aggregate::StddevPop),
        aggregate("variance", doubles(), moment, Aggregate::VarSamp),
        aggregate("var_samp", doubles(), moment, Aggregate::VarSamp),
        aggregate("var_pop", doubles(), moment, Aggregate::VarPop),
        aggregate("skewness", doubles(), moment, Aggregate::Skewness),
        aggregate("kurtosis", doubles(), moment, Aggregate::Kurtosis),
        aggregate("corr", doubles(), moment, Aggregate::Corr),
        aggregate("covar_samp", doubles(), moment, Aggregate::CovarSamp),
        aggregate("covar_pop", doubles(), moment, Aggregate::CovarPop),
        // Spark runs these as `count`, `min` and `max` of a rewritten argument.
        aggregate("count_if", boolean_arg(), count, Aggregate::Count),
        aggregate("bool_and", boolean_arg(), nullable_boolean, Aggregate::Min),
        aggregate("every", boolean_arg(), nullable_boolean, Aggregate::Min),
        aggregate("bool_or", boolean_arg(), nullable_boolean, Aggregate::Max),
        aggregate("some", boolean_arg(), nullable_boolean, Aggregate::Max),
        aggregate("any", boolean_arg(), nullable_boolean, Aggregate::Max),
        aggregate("max_by", Inputs::Any, by, Aggregate::MaxBy),
        aggregate("min_by", Inputs::Any, by, Aggregate::MinBy),
        // The collect and vector aggregates take any single argument with
        // no implicit casts (the vector ones check the type themselves);
        // `array_agg` is `collect_list` under Spark's display name.
        aggregate("collect_list", Inputs::Any, collect, Aggregate::CollectList),
        aggregate(
            "collect_set",
            Inputs::Any,
            collect_set,
            Aggregate::CollectSet,
        ),
        aggregate("array_agg", Inputs::Any, collect, Aggregate::CollectList),
        aggregate("vector_sum", Inputs::Any, vector_sum, Aggregate::VectorSum),
        aggregate("vector_avg", Inputs::Any, vector_avg, Aggregate::VectorAvg),
        // Bitwise aggregates over `TINYINT` .. `BIGINT`.
        aggregate("bit_and", integrals(), bit_and, Aggregate::BitAnd),
        aggregate("bit_or", integrals(), bit_or, Aggregate::BitOr),
        aggregate("bit_xor", integrals(), bit_xor, Aggregate::BitXor),
        // The linear-regression aggregates read both arguments as
        // `DOUBLE`, like the other moment aggregates.
        aggregate("regr_count", doubles(), regr_count, Aggregate::RegrCount),
        aggregate("regr_avgx", doubles(), moment, Aggregate::RegrAvgx),
        aggregate("regr_avgy", doubles(), moment, Aggregate::RegrAvgy),
        aggregate("regr_sxx", doubles(), moment, Aggregate::RegrSxx),
        aggregate("regr_syy", doubles(), moment, Aggregate::RegrSyy),
        aggregate("regr_sxy", doubles(), moment, Aggregate::RegrSxy),
        aggregate("regr_slope", doubles(), moment, Aggregate::RegrSlope),
        aggregate(
            "regr_intercept",
            doubles(),
            moment,
            Aggregate::RegrIntercept,
        ),
        aggregate("regr_r2", doubles(), moment, Aggregate::RegrR2),
        // `mode(col)`: the analyzer types the deterministic and
        // `WITHIN GROUP` forms itself (see `mode_call`); the plain call
        // takes any single value.
        aggregate("mode", Inputs::Any, mode, Aggregate::Mode),
        // `listagg(expr[, delimiter])`: the analyzer fills a missing
        // delimiter with NULL and lowers `WITHIN GROUP` into the ordering
        // keys (see `listagg_call`); `string_agg` is its alias, resolved
        // as `listagg` throughout for Spark's display name.
        aggregate("listagg", string_or_bytes(), listagg, Aggregate::ListAgg),
        aggregate("string_agg", string_or_bytes(), listagg, Aggregate::ListAgg),
        // `try_sum` / `try_avg`: `sum` / `avg` that give NULL instead of
        // an overflow error, ANSI on or off.
        aggregate("try_sum", numeric(), try_sum, Aggregate::TrySum),
        aggregate("try_avg", numeric(), try_avg, Aggregate::TryAvg),
        // The ordered-set and quantile aggregates: the analyzer checks the
        // percentage, accuracy and frequency itself (they must be foldable
        // and in range, and `WITHIN GROUP` lowers to the ordering key), so
        // no implicit casts are declared here.
        aggregate("percentile", Inputs::Any, percentile, Aggregate::Percentile),
        aggregate(
            "percentile_approx",
            Inputs::Any,
            percentile_approx,
            Aggregate::PercentileApprox,
        ),
        aggregate(
            "approx_percentile",
            Inputs::Any,
            percentile_approx,
            Aggregate::PercentileApprox,
        ),
        aggregate("median", Inputs::Any, median, Aggregate::Median),
        aggregate(
            "percentile_cont",
            Inputs::Any,
            percentile_ordered,
            Aggregate::PercentileCont,
        ),
        aggregate(
            "percentile_disc",
            Inputs::Any,
            percentile_ordered,
            Aggregate::PercentileDisc,
        ),
        aggregate(
            "approx_count_distinct",
            Inputs::Any,
            approx_count_distinct,
            Aggregate::ApproxCountDistinct,
        ),
    ]
}
