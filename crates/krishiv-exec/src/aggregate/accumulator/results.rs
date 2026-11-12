// SPDX-License-Identifier: Apache-2.0
//! Result building: exact and approximate percentiles, narrowing and the struct extremum batch fold.

use super::*;

/// Spark's `NEGATIVE_VALUES_IN_FREQUENCY_EXPRESSION` for a negative
/// per-row frequency of an exact percentile.
pub(super) fn negative_frequency(value: i64) -> KrishivError {
    NegativeValuesInFrequencyExpression {
        // Spark quotes the frequency expression and writes the value as a
        // bigint literal. Only the value is known here, so a frequency that is
        // not a literal shows its value, not its text.
        frequency_expression: format!("\"{value}\""),
        negative_value: format!("{value}L"),
    }
    .into()
}

/// The final exact percentiles of groups `ids`: `evaluate` of each
/// group, as a `double` column or an array of it (with the result's
/// element field, so the planner's type check holds).
#[expect(
    clippy::too_many_arguments,
    reason = "one call site, every flag of the query"
)]
pub(super) fn exact_results(
    acc: &ExactAcc,
    ids: &[u32],
    percentages: &[f64],
    reverse: bool,
    cont: bool,
    legacy_disc: bool,
    array: bool,
    t: &DataType,
) -> Result<ArrayRef> {
    if !array {
        let it = ids.iter().map(|&i| {
            acc.evaluate(i as usize, percentages, reverse, cont, legacy_disc)
                .and_then(|mut v| v.pop())
        });
        return approx_column(it, t);
    }
    let DataType::List(field) = t else {
        return Err(internal(
            "an exact percentile array of another type than a list".to_owned(),
        ));
    };
    let mut values: Vec<Option<f64>> = Vec::new();
    let mut offsets = Vec::with_capacity(ids.len() + 1);
    let mut present = Vec::with_capacity(ids.len());
    offsets.push(0);
    for &i in ids {
        // A group with no values (or an empty list of percentages) is NULL,
        // as Spark's `generateOutput` returns it.
        let got = acc
            .evaluate(i as usize, percentages, reverse, cont, legacy_disc)
            .filter(|vs| !vs.is_empty());
        present.push(got.is_some());
        values.extend(got.into_iter().flatten().map(Some));
        offsets.push(
            i32::try_from(values.len())
                .map_err(|_| internal("a percentile of too many values".to_owned()))?,
        );
    }
    Ok(quantile::finish_lists(
        approx_column(values.into_iter(), field.data_type())?,
        offsets,
        Arc::clone(field),
        Some(&present),
    ))
}

/// The final approximate percentiles of groups `ids`: the Greenwald-Khanna
/// query of each group, converted back to the child's Arrow type `t`
/// (as Spark's `ApproximatePercentile.eval` converts its doubles), or an
/// array of it (with the result's element field).
pub(super) fn approx_results(
    gk: &GkAcc,
    ids: &[u32],
    expr: &AggregateExpr,
    relative_error: f64,
    t: &DataType,
) -> Result<ArrayRef> {
    use AggregateFunction as F;
    let (percentages, array) = match &expr.function {
        F::PercentileApprox {
            percentages, array, ..
        } => (percentages, *array),
        _ => {
            return Err(internal(
                "an approximate percentile over another aggregate".to_owned(),
            ));
        }
    };
    let values: Vec<Option<Vec<f64>>> = ids
        .iter()
        .map(|&i| gk.evaluate(i as usize, percentages, relative_error))
        .collect();
    if !array {
        let it = values.into_iter().map(|v| v.and_then(|mut v| v.pop()));
        return approx_column(it, t);
    }
    let DataType::List(field) = t else {
        return Err(internal(
            "an approximate percentile array of another type than a list".to_owned(),
        ));
    };
    let mut offsets = Vec::with_capacity(ids.len() + 1);
    offsets.push(0);
    let mut flat: Vec<Option<f64>> = Vec::new();
    let mut present = Vec::with_capacity(ids.len());
    for v in values {
        // No values, or an empty list of percentages: NULL, as in Spark.
        let got = v.filter(|vs| !vs.is_empty());
        present.push(got.is_some());
        flat.extend(got.into_iter().flatten().map(Some));
        offsets.push(
            i32::try_from(flat.len())
                .map_err(|_| internal("a percentile of too many values".to_owned()))?,
        );
    }
    let flat = approx_column(flat.into_iter(), field.data_type())?;
    Ok(quantile::finish_lists(
        flat,
        offsets,
        Arc::clone(field),
        Some(&present),
    ))
}

/// Percentile values as a column of type `t`: integers, dates, timestamps and
/// ANSI intervals truncate (as Spark's `toByte`/`toShort`/`toInt`/`toLong`
/// do), decimals truncate at their scale, floats convert.
pub(super) fn approx_column(
    it: impl Iterator<Item = Option<f64>>,
    t: &DataType,
) -> Result<ArrayRef> {
    Ok(match t {
        DataType::Float32 => {
            // `as` truncates like Spark's `toFloat`.
            #[expect(clippy::cast_possible_truncation, reason = "Spark toFloat parity")]
            let out = it.map(|v| v.map(|x| x as f32)).collect::<Float32Array>();
            Arc::new(out) as ArrayRef
        }
        DataType::Float64 => Arc::new(it.collect::<Float64Array>()) as ArrayRef,
        // `as` saturates like `toInt` and `toLong`: the largest microseconds
        // read as a double are one past the range, and come back.
        #[expect(clippy::cast_possible_truncation, reason = "Java narrowing parity")]
        DataType::Interval(IntervalUnit::YearMonth) => {
            narrow(it.map(|v| v.map(|x| i128::from(x as i32))), t)?
        }
        #[expect(clippy::cast_possible_truncation, reason = "Java narrowing parity")]
        DataType::Duration(TimeUnit::Microsecond) => {
            narrow(it.map(|v| v.map(|x| i128::from(x as i64))), t)?
        }
        DataType::Decimal64(_, s) | DataType::Decimal128(_, s) => {
            let scale = *s;
            approx_ints(it.map(|v| v.map(|x| decimal_unscaled(x, scale))), t)?
        }
        _ => {
            // `as` saturates like Java narrowing; out-of-range values fail in `narrow`.
            #[expect(clippy::cast_possible_truncation, reason = "Java narrowing parity")]
            let out = approx_ints(it.map(|v| v.map(|x| x as i128)), t)?;
            out
        }
    })
}

/// Approximate values as the `i128`s of an integral, date, timestamp or
/// decimal type `t` (inverse of `int_like`, as [`narrow`] builds them).
pub(super) fn approx_ints(
    it: impl Iterator<Item = Option<i128>>,
    t: &DataType,
) -> Result<ArrayRef> {
    narrow(it, t)
}

/// The unscaled value of the decimal `v` at `scale`: `v`'s shortest decimal
/// text rounded half up, which is how Spark builds a decimal from a double
/// (`BigDecimal.valueOf`), so `1.15` stays `1.15` at scale 2. A value too
/// big for `i128` saturates.
pub(super) fn decimal_unscaled(v: f64, scale: i8) -> i128 {
    if !v.is_finite() {
        return if v.is_sign_negative() {
            i128::MIN
        } else {
            i128::MAX
        };
    }
    // `{:e}` is the shortest text that reads back as `v`: `d.ddde<exp>`.
    let text = format!("{:e}", v.abs());
    let (mantissa, exp) = text.split_once('e').unwrap_or((&text, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let n: i128 = digits.parse().unwrap_or(0);
    let frac = i32::try_from(digits.len()).unwrap_or(0) - 1;
    // `n * 10^(exp - frac)` is `|v|`; at `scale` it is `n * 10^shift`.
    let shift = exp - frac + i32::from(scale);
    let magnitude = if shift >= 0 {
        10_i128
            .checked_pow(u32::try_from(shift).unwrap_or(u32::MAX))
            .and_then(|f| n.checked_mul(f))
            .unwrap_or(i128::MAX)
    } else {
        match 10_i128.checked_pow(u32::try_from(-shift).unwrap_or(u32::MAX)) {
            Some(d) => n / d + i128::from(n % d >= d - d / 2),
            None => 0,
        }
    };
    if v.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

/// Builds an array of type `t` from `i128` values (inverse of `int_like`).
pub(super) fn narrow(it: impl Iterator<Item = Option<i128>>, t: &DataType) -> Result<ArrayRef> {
    fn prim<T>(it: impl Iterator<Item = Option<i128>>, t: &DataType) -> Result<ArrayRef>
    where
        T: arrow_array::ArrowPrimitiveType,
        T::Native: TryFrom<i128>,
    {
        let v: Vec<Option<T::Native>> = it
            .map(|x| x.map(T::Native::try_from).transpose())
            .collect::<Result<_, _>>()
            .map_err(|_| internal(format!("value out of range of {t}")))?;
        let a: PrimitiveArray<T> = v.into_iter().collect();
        Ok(Arc::new(a.with_data_type(t.clone())))
    }
    match t {
        DataType::Boolean => Ok(Arc::new(
            it.map(|x| x.map(|v| v != 0)).collect::<BooleanArray>(),
        )),
        DataType::Int8 => prim::<Int8Type>(it, t),
        DataType::Int16 => prim::<Int16Type>(it, t),
        DataType::Int32 => prim::<Int32Type>(it, t),
        DataType::Int64 => prim::<Int64Type>(it, t),
        DataType::Date32 => prim::<Date32Type>(it, t),
        DataType::Timestamp(..) => prim::<TimestampMicrosecondType>(it, t),
        DataType::Interval(_) => prim::<IntervalYearMonthType>(it, t),
        DataType::Duration(_) => prim::<DurationMicrosecondType>(it, t),
        DataType::Decimal64(..) => prim::<Decimal64Type>(it, t),
        _ => prim::<Decimal128Type>(it, t),
    }
}

/// `a` with the string and binary views it holds copied out of the shared
/// buffers of the batch it was taken from: a one-row winner would otherwise
/// keep every buffer of that batch alive, and the reservation would count
/// them once per group. Nested structs and arrays are compacted field by field; other
/// types already own what `take` copied.
pub(super) fn compact_views(a: ArrayRef) -> ArrayRef {
    match a.data_type() {
        DataType::Utf8View => Arc::new(a.as_string_view().gc()),
        DataType::BinaryView => Arc::new(a.as_binary_view().gc()),
        DataType::Struct(fields) => {
            let s = a.as_struct();
            let children: Vec<ArrayRef> = s.columns().iter().cloned().map(compact_views).collect();
            match arrow_array::StructArray::try_new(fields.clone(), children, s.nulls().cloned()) {
                Ok(compacted) => Arc::new(compacted),
                Err(_) => a,
            }
        }
        DataType::List(field) => {
            let l = a.as_list::<i32>();
            let values = compact_views(Arc::clone(l.values()));
            match arrow_array::ListArray::try_new(
                Arc::clone(field),
                l.offsets().clone(),
                values,
                l.nulls().cloned(),
            ) {
                Ok(compacted) => Arc::new(compacted),
                Err(_) => a,
            }
        }
        DataType::LargeList(field) => {
            let l = a.as_list::<i64>();
            let values = compact_views(Arc::clone(l.values()));
            match arrow_array::LargeListArray::try_new(
                Arc::clone(field),
                l.offsets().clone(),
                values,
                l.nulls().cloned(),
            ) {
                Ok(compacted) => Arc::new(compacted),
                Err(_) => a,
            }
        }
        _ => a,
    }
}

/// Folds one batch of a struct `MIN`/`MAX` into the per-group extrema
/// without a comparison per row. Per group, the candidate rows (in input
/// order, NULL structs skipped) meet in rounds: neighbours are compared in
/// one vectorized comparison across all groups and the strictly better one
/// advances (so a tie keeps the earlier row, as the row-at-a-time fold
/// does). The winner of each group then meets the group's current extremum
/// in one more comparison, and each group that improved takes its winning
/// row once.
#[expect(
    clippy::too_many_arguments,
    reason = "the accumulator's fields travel together"
)]
pub(super) fn struct_extremum_batch(
    a: &ArrayRef,
    groups: &[u32],
    valid: &dyn Fn(usize) -> bool,
    order: CompareKind,
    values: &mut [Option<ArrayRef>],
    bytes: &mut [usize],
    set: &mut [bool],
    heap: &mut usize,
    kernels: &KernelCtx,
) -> Result<()> {
    let wins = |candidate: &ArrayRef, current: &ArrayRef| -> Result<Vec<bool>> {
        let out = cmp(
            order,
            &Value::Array(Arc::clone(candidate)),
            &Value::Array(Arc::clone(current)),
            kernels,
        )?;
        Ok((0..out.len())
            .map(|k| out.is_valid(k) && out.value(k))
            .collect())
    };
    let take_rows = |rows: &[u32]| -> Result<ArrayRef> {
        arrow_select::take::take(a.as_ref(), &UInt32Array::from(rows.to_vec()), None)
            .map_err(|e| internal(e.to_string()))
    };
    // The valid rows, grouped (stable: input order within a group).
    let mut candidates: Vec<(u32, u32)> = Vec::new();
    for (i, &g) in groups.iter().enumerate() {
        if valid(i) {
            let row = u32::try_from(i)
                .map_err(|_| internal("a struct extremum past row u32::MAX".to_owned()))?;
            candidates.push((g, row));
        }
    }
    candidates.sort_by_key(|(g, _)| *g);
    // Tournament rounds until each group has one candidate left.
    loop {
        let mut left: Vec<u32> = Vec::new();
        let mut right: Vec<u32> = Vec::new();
        let mut next: Vec<(u32, u32)> = Vec::with_capacity(candidates.len());
        let mut start = 0;
        let mut pending = false;
        while start < candidates.len() {
            let g = candidates[start].0;
            let mut end = start;
            while end < candidates.len() && candidates[end].0 == g {
                end += 1;
            }
            let mut k = start;
            while k + 1 < end {
                left.push(candidates[k].1);
                right.push(candidates[k + 1].1);
                // Placeholder; resolved after the comparison.
                next.push((g, u32::MAX));
                k += 2;
                pending = true;
            }
            if k < end {
                next.push(candidates[k]);
            }
            start = end;
        }
        if !pending {
            break;
        }
        let better = wins(&take_rows(&right)?, &take_rows(&left)?)?;
        let mut pair = 0;
        for slot in &mut next {
            if slot.1 == u32::MAX {
                slot.1 = if better[pair] {
                    right[pair]
                } else {
                    left[pair]
                };
                pair += 1;
            }
        }
        candidates = next;
    }
    if candidates.is_empty() {
        return Ok(());
    }
    // Each group's winner against its current extremum.
    let winners: Vec<u32> = candidates.iter().map(|(_, row)| *row).collect();
    let winner_rows = take_rows(&winners)?;
    let held: Vec<usize> = (0..candidates.len())
        .filter(|k| values[candidates[*k].0 as usize].is_some())
        .collect();
    let mut improves = vec![true; candidates.len()];
    if !held.is_empty() {
        let candidate = arrow_select::take::take(
            winner_rows.as_ref(),
            &UInt32Array::from(
                held.iter()
                    .map(|k| u32::try_from(*k).unwrap_or(u32::MAX))
                    .collect::<Vec<_>>(),
            ),
            None,
        )
        .map_err(|e| internal(e.to_string()))?;
        let currents: Vec<ArrayRef> = held
            .iter()
            .filter_map(|k| values[candidates[*k].0 as usize].clone())
            .collect();
        let refs: Vec<&dyn Array> = currents.iter().map(AsRef::as_ref).collect();
        let current = arrow_select::concat::concat(&refs).map_err(|e| internal(e.to_string()))?;
        let better = wins(&candidate, &current)?;
        for (pos, k) in held.iter().enumerate() {
            improves[*k] = better[pos];
        }
    }
    for (k, (g, row)) in candidates.iter().enumerate() {
        if !improves[k] {
            continue;
        }
        let g = *g as usize;
        let taken = compact_views(take_rows(&[*row])?);
        let size = taken.get_buffer_memory_size();
        *heap = heap.saturating_sub(bytes[g]) + size;
        bytes[g] = size;
        values[g] = Some(taken);
        set[g] = true;
    }
    Ok(())
}
