// SPDX-License-Identifier: Apache-2.0
//! The output side of an accumulator: its partial state and its result column.

use super::*;

impl Accumulator {
    /// The partial states of groups `ids`.
    pub(crate) fn state(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        let pick = |i: &u32| *i as usize;
        Ok(match &self.kind {
            Kind::Count(c) => vec![Arc::new(Int64Array::from_iter_values(
                ids.iter().map(|i| c[pick(i)]),
            ))],
            Kind::SumInt { sum, seen } => vec![Arc::new(Int64Array::from_iter(
                ids.iter().map(|i| seen[pick(i)].then(|| sum[pick(i)])),
            ))],
            Kind::SumFloat { sum, seen } => vec![Arc::new(Float64Array::from_iter(
                ids.iter().map(|i| seen[pick(i)].then(|| sum[pick(i)])),
            ))],
            Kind::SumDec { sum, state } => vec![
                self.decimal(
                    ids.iter()
                        .map(|i| (state[pick(i)] == 1).then(|| sum[pick(i)])),
                    &self.result,
                )?,
                Arc::new(BooleanArray::from_iter(
                    ids.iter().map(|i| Some(state[pick(i)] != 0)),
                )),
            ],
            Kind::AvgFloat { sum, count } => vec![
                Arc::new(Float64Array::from_iter_values(
                    ids.iter().map(|i| sum[pick(i)]),
                )),
                Arc::new(Int64Array::from_iter_values(
                    ids.iter().map(|i| count[pick(i)]),
                )),
            ],
            Kind::AvgDec {
                sum,
                count,
                overflow,
            } => vec![
                self.decimal(
                    ids.iter()
                        .map(|i| (!overflow[pick(i)]).then(|| sum[pick(i)])),
                    &self.expr.avg_sum_type(),
                )?,
                Arc::new(Int64Array::from_iter_values(
                    ids.iter().map(|i| count[pick(i)]),
                )),
            ],
            Kind::MinMax { values, set, .. } => {
                vec![self.value_array(values, &self.result, |i| set[i], ids)?]
            }
            Kind::MinMaxStruct { values, set, .. } => {
                vec![self.struct_values(values, &self.result, |i| set[i], ids)?]
            }
            Kind::Pick {
                values, set, null, ..
            } => vec![
                self.value_array(values, &self.result, |i| set[i] && !null[i], ids)?,
                Arc::new(BooleanArray::from_iter(
                    ids.iter().map(|i| Some(set[pick(i)])),
                )),
            ],
            Kind::Moments(m) => m.state(ids)?,
            Kind::Collect(c) => vec![c.state(ids, &self.result)?],
            Kind::Vector(v) => vec![v.state_sums(ids)?, v.state_count(ids)],
            Kind::TrySumInt { sum, state } => vec![
                Arc::new(Int64Array::from_iter(
                    ids.iter()
                        .map(|i| (state[pick(i)] == 1).then(|| sum[pick(i)])),
                )),
                Arc::new(BooleanArray::from_iter(
                    ids.iter().map(|i| Some(state[pick(i)] != 0)),
                )),
            ],
            Kind::SumInterval {
                sum, count, over, ..
            } => vec![
                Arc::new(Int64Array::from_iter(ids.iter().map(|i| {
                    let g = pick(i);
                    (count[g] > 0 && !over[g]).then(|| sum[g])
                }))),
                Arc::new(Int64Array::from_iter_values(
                    ids.iter().map(|i| count[pick(i)]),
                )),
            ],
            Kind::Bit { values, set, .. } => {
                vec![self.value_array(values, &self.result, |i| set[i], ids)?]
            }
            Kind::Mode(m) => m.state(ids)?,
            Kind::ListAgg(l) => l.state(ids)?,
            Kind::Exact(acc) => acc.state(ids)?,
            Kind::Approx(gk) => {
                let AggregateFunction::PercentileApprox { accuracy, .. } = &self.expr.function
                else {
                    return Err(internal(
                        "an approximate percentile over another aggregate".to_owned(),
                    ));
                };
                gk.state(ids, 1.0 / *accuracy as f64)
            }
            Kind::Hll(h) => h.state(ids),
            Kind::Udf(u) => u.state(ids)?,
            Kind::By {
                values,
                keys,
                set,
                null,
                ..
            } => {
                let (_, key_type) = self
                    .expr
                    .by_types()
                    .ok_or_else(|| internal("max_by without value and key types".to_owned()))?;
                vec![
                    self.value_array(values, &self.result, |i| set[i] && !null[i], ids)?,
                    self.value_array(keys, key_type, |i| set[i], ids)?,
                ]
            }
        })
    }

    /// A decimal state or result column of type `t` (either carrier).
    pub(super) fn decimal(
        &self,
        v: impl Iterator<Item = Option<i128>>,
        t: &DataType,
    ) -> Result<ArrayRef> {
        match t {
            DataType::Decimal64(..) => narrow(v, t),
            _ => Ok(Arc::new(
                Decimal128Array::from_iter(v).with_data_type(t.clone()),
            )),
        }
    }

    /// The struct winners of groups `ids` as type `t`; `on(group)` says
    /// whether the group has a value (else a NULL struct row). The
    /// partial state is the struct value itself, so partial and final
    /// states share this column.
    pub(super) fn struct_values(
        &self,
        values: &[Option<ArrayRef>],
        t: &DataType,
        on: impl Fn(usize) -> bool,
        ids: &[u32],
    ) -> Result<ArrayRef> {
        let mut parts = Vec::with_capacity(ids.len());
        for i in ids {
            let g = *i as usize;
            match on(g).then(|| values[g].as_ref()).flatten() {
                Some(winner) => parts.push(Arc::clone(winner)),
                None => parts.push(arrow_array::new_null_array(t, 1)),
            }
        }
        if parts.is_empty() {
            return Ok(arrow_array::new_null_array(t, 0));
        }
        let refs: Vec<&dyn Array> = parts.iter().map(AsRef::as_ref).collect();
        arrow_select::concat::concat(&refs).map_err(|e| internal(e.to_string()))
    }

    /// `values` of groups `ids` as type `t`; `on(group)` says whether the
    /// group has a value (else NULL).
    pub(super) fn value_array(
        &self,
        values: &Values,
        t: &DataType,
        on: impl Fn(usize) -> bool,
        ids: &[u32],
    ) -> Result<ArrayRef> {
        let on = |i: &u32| on(*i as usize);
        let at = |i: &u32| *i as usize;
        Ok(match values {
            Values::Float(v) => {
                let it = ids.iter().map(|i| on(i).then(|| v[at(i)]));
                if *t == DataType::Float32 {
                    #[expect(clippy::cast_possible_truncation, reason = "the value was a float")]
                    let a: PrimitiveArray<Float32Type> = it.map(|x| x.map(|x| x as f32)).collect();
                    Arc::new(a)
                } else {
                    Arc::new(it.collect::<Float64Array>())
                }
            }
            Values::Bytes(v) => {
                let it = ids.iter().map(|i| on(i).then(|| &*v[at(i)]));
                match t {
                    DataType::Binary => Arc::new(it.collect::<BinaryArray>()),
                    DataType::BinaryView => Arc::new(it.collect::<BinaryViewArray>()),
                    _ => {
                        let s: Vec<Option<&str>> = it
                            .map(|b| b.map(std::str::from_utf8).transpose())
                            .collect::<Result<_, _>>()
                            .map_err(|e| internal(e.to_string()))?;
                        if *t == DataType::Utf8 {
                            Arc::new(StringArray::from(s))
                        } else {
                            Arc::new(StringViewArray::from(s))
                        }
                    }
                }
            }
            Values::Int(v) => {
                let it = ids.iter().map(|i| on(i).then(|| v[at(i)]));
                narrow(it, t)?
            }
        })
    }

    /// The final values of groups `ids`.
    ///
    /// # Errors
    /// ANSI overflow errors.
    pub(crate) fn evaluate(&self, ids: &[u32]) -> Result<ArrayRef> {
        let at = |i: &u32| *i as usize;
        let ansi = self.ansi;
        Ok(match &self.kind {
            Kind::SumDec { sum, state } => {
                let (DataType::Decimal64(p, s) | DataType::Decimal128(p, s)) = self.result else {
                    return Err(internal("decimal sum type".to_owned()));
                };
                let mut out = Vec::with_capacity(ids.len());
                for i in ids {
                    let v = sum[at(i)];
                    out.push(match state[at(i)] {
                        0 => None,
                        1 if v.unsigned_abs() < pow10(p).unsigned_abs() => Some(v),
                        _ if !ansi => None,
                        1 => return Err(out_of_range(plain(v, s), p, s)),
                        _ => return Err(decimal_sum_overflow("sum")),
                    });
                }
                self.decimal(out.into_iter(), &self.result)?
            }
            Kind::AvgFloat { sum, count } => {
                Arc::new(Float64Array::from_iter(ids.iter().map(|i| {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "Spark divides by count as double"
                    )]
                    let n = count[at(i)] as f64;
                    (count[at(i)] != 0).then(|| sum[at(i)] / n)
                })))
            }
            Kind::AvgDec {
                sum,
                count,
                overflow,
            } => {
                let (
                    DataType::Decimal64(p, s) | DataType::Decimal128(p, s),
                    DataType::Decimal64(_, ss) | DataType::Decimal128(_, ss),
                ) = (&self.result, self.expr.avg_sum_type())
                else {
                    return Err(internal("decimal avg type".to_owned()));
                };
                let mut out = Vec::with_capacity(ids.len());
                for i in ids {
                    let g = at(i);
                    out.push(if count[g] == 0 {
                        None
                    } else if overflow[g] {
                        if ansi {
                            return Err(decimal_sum_overflow("avg"));
                        }
                        None
                    } else if let Some(v) = avg_fast(sum[g], ss, count[g], *p, *s) {
                        Some(v)
                    } else {
                        match avg_decimal(sum[g], ss, count[g], *p, *s) {
                            (Some(v), _) => Some(v),
                            (None, text) if ansi => return Err(out_of_range(text, *p, *s)),
                            (None, _) => None,
                        }
                    });
                }
                self.decimal(out.into_iter(), &self.result)?
            }
            Kind::Moments(m) => {
                let array = m.evaluate(ids)?;
                // `regr_count` counts the pairs: its state is the pair
                // buffer, its value the count as a `bigint`.
                if self.expr.function == AggregateFunction::RegrCount {
                    let doubles = array.as_primitive::<Float64Type>();
                    Arc::new(
                        doubles
                            .iter()
                            .map(|v| {
                                v.map(|v| {
                                    #[expect(
                                        clippy::cast_possible_truncation,
                                        reason = "a pair count below 2^53 is whole and exact"
                                    )]
                                    let n = v as i64;
                                    n
                                })
                            })
                            .collect::<Int64Array>(),
                    )
                } else {
                    array
                }
            }
            // The collect final values are the state lists; the vector
            // ones the sums (or means) as `float`.
            Kind::Collect(c) => c.state(ids, &self.result)?,
            Kind::Vector(v) => v.evaluate(ids, &self.result)?,
            Kind::TrySumInt { sum, state } => Arc::new(Int64Array::from_iter(
                ids.iter().map(|i| (state[at(i)] == 1).then(|| sum[at(i)])),
            )),
            Kind::SumInterval {
                sum, count, over, ..
            } => {
                let mean = matches!(
                    self.expr.function,
                    AggregateFunction::Avg | AggregateFunction::TryAvg
                );
                narrow(
                    ids.iter().map(|i| {
                        let g = at(i);
                        (count[g] > 0 && !over[g]).then(|| {
                            if mean {
                                interval_mean(sum[g], count[g])
                            } else {
                                i128::from(sum[g])
                            }
                        })
                    }),
                    &self.result,
                )?
            }
            Kind::Bit { values, set, .. } => {
                self.value_array(values, &self.result, |i| set[i], ids)?
            }
            Kind::Mode(m) => m.evaluate(ids)?,
            Kind::ListAgg(l) => l.evaluate(ids)?,
            Kind::Exact(acc) => {
                let (percentages, reverse, cont, legacy, array) = self.exact_params()?;
                exact_results(
                    acc,
                    ids,
                    &percentages,
                    reverse,
                    cont,
                    legacy,
                    array,
                    &self.result,
                )?
            }
            Kind::Approx(gk) => {
                let AggregateFunction::PercentileApprox { accuracy, .. } = &self.expr.function
                else {
                    return Err(internal(
                        "an approximate percentile over another aggregate".to_owned(),
                    ));
                };
                approx_results(gk, ids, &self.expr, 1.0 / *accuracy as f64, &self.result)?
            }
            Kind::Hll(h) => Arc::new(Int64Array::from_iter_values(
                ids.iter().map(|i| h.estimate(at(i))),
            )) as ArrayRef,
            Kind::Udf(u) => u.evaluate(ids)?,
            // Only the value: the ordering key is state, not a result.
            Kind::By {
                values, set, null, ..
            } => self.value_array(values, &self.result, |i| set[i] && !null[i], ids)?,
            Kind::Count(_)
            | Kind::SumInt { .. }
            | Kind::SumFloat { .. }
            | Kind::MinMax { .. }
            | Kind::MinMaxStruct { .. }
            | Kind::Pick { .. } => {
                let mut s = self.state(ids)?;
                s.swap_remove(0)
            }
        })
    }
}
