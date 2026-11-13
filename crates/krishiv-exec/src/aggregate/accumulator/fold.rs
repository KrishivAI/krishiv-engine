// SPDX-License-Identifier: Apache-2.0
//! The batch fold: one match over the accumulator kinds.

use super::*;

impl Accumulator {
    pub(super) fn fold(
        &mut self,
        groups: &[u32],
        args: &[&ArrayRef],
        state: bool,
        w: &Weights,
        run: Option<&Running<'_>>,
        mask: Option<&BooleanArray>,
    ) -> Result<()> {
        debug_assert!(
            mask.is_none() || !state,
            "a merge over partial states carries no mask"
        );
        if let Kind::Udf(u) = &mut self.kind {
            if run.is_some() {
                return Err(internal("a running fold of an aggregate UDF".to_owned()));
            }
            if state {
                let states: Vec<ArrayRef> = args.iter().map(|a| Arc::clone(a)).collect();
                return u.merge(groups, &states);
            }
            return u.update(groups, args, mask, w);
        }
        let want = if state {
            self.state_len()
        } else {
            self.expr.args.len()
        };
        if args.len() != want || (want == 0 && !matches!(self.kind, Kind::Count(_))) {
            return Err(internal(format!(
                "{:?} got {} argument columns, wants {want}",
                self.expr.function,
                args.len()
            )));
        }
        if (state
            || matches!(
                self.kind,
                Kind::MinMax { .. }
                    | Kind::MinMaxStruct { .. }
                    | Kind::Pick { .. }
                    | Kind::By { .. }
                    | Kind::Moments(_)
                    | Kind::TrySumInt { .. }
                    | Kind::SumInterval { .. }
                    | Kind::Collect(_)
                    | Kind::Vector(_)
                    | Kind::Bit { .. }
                    | Kind::Mode(_)
                    | Kind::ListAgg(_)
                    | Kind::Exact(_)
                    | Kind::Approx(_)
                    | Kind::Hll(_)
            ))
            && !is_unit(w)
        {
            return Err(unsupported(format!(
                "{:?} over rows with weights other than 1 (needs incremental view maintenance)",
                self.expr.function
            )));
        }
        let ansi = self.ansi;
        let bound = self.sum_bound;
        let (proofs, max_sum) = (self.proofs, self.max_sum);
        // Validity bitmaps, read once per batch (no virtual call per row);
        // logical, so an untyped NULL column (`count(NULL)`) counts nothing.
        let nulls: Vec<Option<NullBuffer>> =
            args.iter().map(arrow_array::Array::logical_nulls).collect();
        let first = nulls.first().and_then(Option::as_ref);
        // Rows the mask drops (`FILTER (WHERE ...)`); a NULL is not true.
        let active = |i: usize| mask.is_none_or(|m| m.is_valid(i) && m.value(i));
        // A row the mask drops is skipped like a NULL argument.
        let valid = |i: usize| first.is_none_or(|n| n.is_valid(i)) && active(i);
        // After row `i` of a running fold: the state of its partition's
        // group is copied into the row's snapshot group, if it has one.
        macro_rules! snapshot {
            ($i:expr, $($v:ident),+) => {
                if let Some(r) = run {
                    let to = r.snapshots[$i];
                    if to != Running::NONE {
                        let (to, from) = (to as usize, r.parts[$i] as usize);
                        $( $v[to] = $v[from].clone(); )+
                    }
                }
            };
        }
        if run.is_some()
            && matches!(
                self.kind,
                Kind::SumDec { .. }
                    | Kind::AvgDec { .. }
                    | Kind::Exact(_)
                    | Kind::Approx(_)
                    | Kind::Hll(_)
                    | Kind::MinMax {
                        values: Values::Bytes(_),
                        ..
                    }
                    | Kind::Pick {
                        values: Values::Bytes(_),
                        ..
                    }
            )
        {
            return Err(internal(format!(
                "{:?} over {:?} has no running fold",
                self.expr.function, self.expr.arg_type
            )));
        }
        if run.is_some()
            && matches!(
                &self.kind,
                Kind::By { values, keys, .. }
                    if matches!(values, Values::Bytes(_)) || matches!(keys, Values::Bytes(_))
            )
        {
            return Err(unsupported(format!(
                "{:?} over strings or binaries in a running window frame",
                self.expr.function
            )));
        }
        match (&mut self.kind, state) {
            // Handled at the top of `fold`.
            (Kind::Udf(_), _) => return Err(internal("an aggregate UDF fold".to_owned())),
            // Every row counts once with one weight: no per-row test.
            (Kind::Count(c), false)
                if run.is_none()
                    && mask.is_none()
                    && matches!(w, Weights::Const(_))
                    && nulls
                        .iter()
                        .all(|n| n.as_ref().is_none_or(|n| n.null_count() == 0)) =>
            {
                small_fold::dense_count(groups, weight(w, 0), c);
            }
            (Kind::Count(c), false) => {
                for (i, &g) in groups.iter().enumerate() {
                    if active(i)
                        && nulls
                            .iter()
                            .all(|n| n.as_ref().is_none_or(|n| n.is_valid(i)))
                    {
                        let x = &mut c[g as usize];
                        *x = x.wrapping_add(weight(w, i));
                    }
                    snapshot!(i, c);
                }
            }
            (Kind::Count(c), true) => {
                let s = counts_of(args[0])?;
                for (&g, &v) in groups.iter().zip(s.iter()) {
                    let x = &mut c[g as usize];
                    *x = x.wrapping_add(v);
                }
            }
            (Kind::SumInt { sum, seen }, _) => {
                let a = args[0];
                let v = ints(a)?;
                for (i, &g) in groups.iter().enumerate() {
                    if active(i) && valid(i) {
                        let g = g as usize;
                        let wv = if state {
                            Some(v[i])
                        } else {
                            v[i].checked_mul(weight(w, i))
                        };
                        let s = wv.and_then(|x| sum[g].checked_add(x));
                        sum[g] = match s {
                            Some(s) => s,
                            None if ansi => return Err(long_overflow()),
                            None => sum[g].wrapping_add(v[i].wrapping_mul(if state {
                                1
                            } else {
                                weight(w, i)
                            })),
                        };
                        seen[g] = true;
                    }
                    snapshot!(i, sum, seen);
                }
            }
            (Kind::SumFloat { sum, seen }, _) => {
                let a = args[0];
                let v = floats(a)?;
                for (i, &g) in groups.iter().enumerate() {
                    if active(i) && valid(i) {
                        #[expect(clippy::cast_precision_loss, reason = "a weight scales a double")]
                        let wv = if state { 1.0 } else { weight(w, i) as f64 };
                        sum[g as usize] += v[i] * wv;
                        seen[g as usize] = true;
                    }
                    snapshot!(i, sum, seen);
                }
            }
            // `try_sum` of integers: an overflow sticks (the value reads
            // as NULL), ANSI on or off.
            (Kind::TrySumInt { sum, state: st }, _) => {
                let a = args[0];
                let v = ints(a)?;
                if state {
                    let nonempty = booleans_of(args[1])?;
                    for (i, &g) in groups.iter().enumerate() {
                        let g = g as usize;
                        if nonempty.value(i) {
                            // An overflowed partial (sum NULL, nonempty)
                            // is sticky.
                            if !valid(i) {
                                st[g] = 2;
                            } else if st[g] != 2 {
                                (sum[g], st[g]) =
                                    sum[g].checked_add(v[i]).map_or((0, 2), |s| (s, 1));
                            }
                        }
                    }
                } else {
                    for (i, &g) in groups.iter().enumerate() {
                        if valid(i) {
                            let g = g as usize;
                            if st[g] != 2 {
                                let s = v[i]
                                    .checked_mul(weight(w, i))
                                    .and_then(|x| sum[g].checked_add(x));
                                (sum[g], st[g]) = s.map_or((0, 2), |s| (s, 1));
                            }
                        }
                        snapshot!(i, sum, st);
                    }
                }
            }
            (
                Kind::SumInterval {
                    sum,
                    count,
                    over,
                    months,
                },
                _,
            ) => {
                let raises = matches!(
                    self.expr.function,
                    AggregateFunction::Sum | AggregateFunction::Avg
                );
                let months = *months;
                let v = interval_values(args[0])?;
                // A partial's count says it holds rows; its NULL sum then
                // says it overflowed.
                let counts = if state { Some(ints(args[1])?) } else { None };
                for (i, &g) in groups.iter().enumerate() {
                    let g = g as usize;
                    let rows = match &counts {
                        Some(c) => c[i],
                        None => i64::from(valid(i)),
                    };
                    if rows > 0 {
                        count[g] = count[g].saturating_add(rows);
                        if state && !valid(i) {
                            over[g] = true;
                        } else if !over[g] {
                            match interval_add(sum[g], v[i], months) {
                                Some(s) => sum[g] = s,
                                None if raises => return Err(interval_sum_overflow()),
                                None => over[g] = true,
                            }
                        }
                    }
                    snapshot!(i, sum, count, over);
                }
            }
            // The bitwise aggregates: NULLs skipped, NULL for no input.
            // Weights other than 1 need incremental view maintenance,
            // as for `min`/`max` (see the check above).
            (Kind::Bit { op, values, set }, false) => {
                let op = *op;
                let a = args[0];
                match values {
                    Values::Int(vals) => {
                        let v = ints(a)?;
                        for (i, &g) in groups.iter().enumerate() {
                            let g = g as usize;
                            if valid(i) {
                                if !set[g] {
                                    vals[g] = i128::from(v[i]);
                                    set[g] = true;
                                } else {
                                    vals[g] = bit(op, vals[g], i128::from(v[i]));
                                }
                            }
                            snapshot!(i, vals, set);
                        }
                    }
                    _ => {
                        return Err(internal(
                            "a bitwise aggregate over non-integral values".to_owned(),
                        ));
                    }
                }
            }
            (Kind::Bit { op, values, set }, true) => {
                let op = *op;
                match values {
                    Values::Int(vals) => {
                        let s = args[0];
                        if s.len() != groups.len() {
                            return Err(internal(
                                "a bitwise state of another length than the group ids".to_owned(),
                            ));
                        }
                        let v = ints(s)?;
                        let nulls = s.logical_nulls();
                        for (i, &g) in groups.iter().enumerate() {
                            let g = g as usize;
                            if nulls.as_ref().is_none_or(|n| n.is_valid(i)) {
                                if !set[g] {
                                    vals[g] = i128::from(v[i]);
                                    set[g] = true;
                                } else {
                                    vals[g] = bit(op, vals[g], i128::from(v[i]));
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(internal(
                            "a bitwise aggregate over non-integral values".to_owned(),
                        ));
                    }
                }
            }
            (Kind::SumDec { sum, state: st }, _) => {
                let nonempty = args.get(1).map(|b| booleans_of(b)).transpose()?;
                let f = DecFold {
                    groups,
                    nulls: first,
                    mask,
                    w: (!state).then_some(w),
                    bound,
                    max_sum,
                    proofs,
                };
                match decimals(args[0])? {
                    Decimals::I64(v) => f.sum(sum, st, v, nonempty),
                    Decimals::I128(v) => f.sum(sum, st, v, nonempty),
                }
            }
            (Kind::AvgFloat { sum, count }, _) => {
                let a = args[0];
                let v = floats(a)?;
                let counts = args.get(1).map(|c| counts_of(c)).transpose()?;
                for (i, &g) in groups.iter().enumerate() {
                    let g = g as usize;
                    if let Some(c) = &counts {
                        if valid(i) {
                            sum[g] += v[i];
                        }
                        count[g] = count[g].wrapping_add(c[i]);
                    } else if active(i) && valid(i) {
                        let wt = weight(w, i);
                        #[expect(clippy::cast_precision_loss, reason = "a weight scales a double")]
                        let wv = wt as f64;
                        sum[g] += v[i] * wv;
                        count[g] = count[g].wrapping_add(wt);
                    }
                    snapshot!(i, sum, count);
                }
            }
            (
                Kind::AvgDec {
                    sum,
                    count,
                    overflow,
                },
                _,
            ) => {
                let counts = args.get(1).map(|c| counts_of(c)).transpose()?;
                let f = DecFold {
                    groups,
                    nulls: first,
                    mask,
                    w: counts.is_none().then_some(w),
                    bound,
                    max_sum,
                    proofs,
                };
                match decimals(args[0])? {
                    Decimals::I64(v) => f.avg(sum, count, overflow, v, counts.as_deref()),
                    Decimals::I128(v) => f.avg(sum, count, overflow, v, counts.as_deref()),
                }
            }
            (
                Kind::MinMax {
                    max,
                    values,
                    set,
                    heap,
                },
                _,
            ) => {
                let a = args[0];
                let max = *max;
                match values {
                    Values::Int(vals) => {
                        let v = int_like(a)?;
                        for (i, &g) in groups.iter().enumerate() {
                            let g = g as usize;
                            if active(i)
                                && valid(i)
                                && (!set[g] || (v[i] > vals[g]) == max && v[i] != vals[g])
                            {
                                vals[g] = v[i];
                                set[g] = true;
                            }
                            snapshot!(i, vals, set);
                        }
                    }
                    Values::Float(vals) => {
                        let v = floats(a)?;
                        for (i, &g) in groups.iter().enumerate() {
                            let g = g as usize;
                            let better = if max {
                                float_lt(vals[g], v[i])
                            } else {
                                float_lt(v[i], vals[g])
                            };
                            if active(i) && valid(i) && (!set[g] || better) {
                                vals[g] = v[i];
                                set[g] = true;
                            }
                            snapshot!(i, vals, set);
                        }
                    }
                    Values::Bytes(vals) => {
                        let v = byte_values(a)?;
                        for (i, &g) in groups.iter().enumerate() {
                            let g = g as usize;
                            let cur: &[u8] = &vals[g];
                            let better = if max { v[i] > cur } else { v[i] < cur };
                            if active(i) && valid(i) && (!set[g] || better) {
                                *heap = heap.saturating_sub(vals[g].len()) + v[i].len();
                                vals[g] = v[i].into();
                                set[g] = true;
                            }
                        }
                    }
                }
            }
            (
                Kind::MinMaxStruct {
                    max,
                    values,
                    bytes,
                    set,
                    heap,
                    kernels,
                },
                _,
            ) => {
                let a = args[0];
                if a.len() != groups.len() {
                    return Err(internal(
                        "a struct extremum of another length than the group ids".to_owned(),
                    ));
                }
                if !struct_min_max_supported(a.data_type()) {
                    return Err(internal(format!("a struct extremum of {}", a.data_type())));
                }
                let order = if *max {
                    CompareKind::Greater
                } else {
                    CompareKind::Less
                };
                // Without running-window snapshots the batch is reduced at
                // once: a tournament per group over whole-chunk comparisons,
                // then one comparison of the group winners against the
                // current extrema and one `take` per winning group.
                // A `FILTER` mask drops a row like a NULL does.
                let live = |i: usize| active(i) && valid(i);
                if run.is_none() {
                    struct_extremum_batch(
                        a, groups, &live, order, values, bytes, set, heap, kernels,
                    )?;
                    return Ok(());
                }
                for (i, &g) in groups.iter().enumerate() {
                    let g = g as usize;
                    if live(i) {
                        // A strict win keeps the first row on ties, so the
                        // value is the same whatever the grouping.
                        let better = match &values[g] {
                            None => true,
                            Some(current) => {
                                let candidate = a.slice(i, 1);
                                let out = cmp(
                                    order,
                                    &Value::Array(candidate),
                                    &Value::Array(Arc::clone(current)),
                                    kernels,
                                )?;
                                out.is_valid(0) && out.value(0)
                            }
                        };
                        if better {
                            let index = u32::try_from(i).map_err(|_| {
                                internal("a struct extremum past row u32::MAX".to_owned())
                            })?;
                            let taken = compact_views(
                                arrow_select::take::take(a, &UInt32Array::from(vec![index]), None)
                                    .map_err(|e| internal(e.to_string()))?,
                            );
                            *heap = heap.saturating_sub(bytes[g]) + taken.get_buffer_memory_size();
                            bytes[g] = taken.get_buffer_memory_size();
                            values[g] = Some(taken);
                            set[g] = true;
                        }
                    }
                    if let Some(r) = run {
                        let to = r.snapshots[i];
                        if to != Running::NONE {
                            let (to, from) = (to as usize, r.parts[i] as usize);
                            *heap = heap.saturating_sub(bytes[to]) + bytes[from];
                            bytes[to] = bytes[from];
                            values[to] = values[from].clone();
                            set[to] = set[from];
                        }
                    }
                }
            }
            (
                Kind::Pick {
                    last,
                    ignore_nulls,
                    values,
                    set,
                    null,
                    heap,
                },
                _,
            ) => {
                let a = args[0];
                let (last, ignore_nulls) = (*last, *ignore_nulls);
                // A partial state's row is taken when its `valueSet` says so
                // (its value may be NULL); a raw row unless it is a NULL to
                // skip.
                let taken = if state {
                    Some(booleans_of(args[1])?)
                } else {
                    None
                };
                let take = |i: usize, g: usize, set: &[bool]| {
                    let present = taken.map_or(valid(i) || !ignore_nulls, |t| t.value(i));
                    active(i) && present && (last || !set[g])
                };
                match values {
                    Values::Int(vals) => {
                        let v = int_like(a)?;
                        for (i, &g) in groups.iter().enumerate() {
                            let g = g as usize;
                            if take(i, g, set) {
                                set[g] = true;
                                null[g] = !valid(i);
                                if valid(i) {
                                    vals[g] = v[i];
                                } else {
                                    vals[g] = 0;
                                }
                            }
                            snapshot!(i, vals, set, null);
                        }
                    }
                    Values::Float(vals) => {
                        let v = floats(a)?;
                        for (i, &g) in groups.iter().enumerate() {
                            let g = g as usize;
                            if take(i, g, set) {
                                set[g] = true;
                                null[g] = !valid(i);
                                if valid(i) {
                                    vals[g] = v[i];
                                } else {
                                    vals[g] = 0.0;
                                }
                            }
                            snapshot!(i, vals, set, null);
                        }
                    }
                    Values::Bytes(vals) => {
                        let v = byte_values(a)?;
                        for (i, &g) in groups.iter().enumerate() {
                            let g = g as usize;
                            if take(i, g, set) {
                                set[g] = true;
                                null[g] = !valid(i);
                                if valid(i) {
                                    *heap = heap.saturating_sub(vals[g].len()) + v[i].len();
                                    vals[g] = v[i].into();
                                } else {
                                    *heap = heap.saturating_sub(vals[g].len());
                                    vals[g] = Box::default();
                                }
                            }
                        }
                    }
                }
            }
            (Kind::Moments(m), false) => {
                let cols = args.iter().map(|a| floats(a)).collect::<Result<Vec<_>>>()?;
                let cols: Vec<&[f64]> = cols.iter().map(AsRef::as_ref).collect();
                // A row the mask drops (`FILTER (WHERE ...)`) is skipped like
                // a NULL argument.
                let all_valid = |i: usize| {
                    active(i)
                        && nulls
                            .iter()
                            .all(|n| n.as_ref().is_none_or(|n| n.is_valid(i)))
                };
                m.update(groups, &cols, all_valid, run)?;
            }
            (Kind::Moments(m), true) => {
                let cols = args
                    .iter()
                    .map(|a| state_doubles(a))
                    .collect::<Result<Vec<_>>>()?;
                m.merge(groups, &cols)?;
            }
            (Kind::Collect(c), false) => {
                let a = args[0];
                if a.len() != groups.len() {
                    return Err(internal(
                        "a collect column of another length than the group ids".to_owned(),
                    ));
                }
                c.update(groups, a, &valid, run)?;
            }
            (Kind::Collect(c), true) => {
                let [state] = args else {
                    return Err(internal(
                        "a collect state of another shape than its values".to_owned(),
                    ));
                };
                c.merge(groups, state)?;
            }
            (Kind::Vector(v), false) => {
                let a = args[0];
                if a.len() != groups.len() {
                    return Err(internal(
                        "a vector column of another length than the group ids".to_owned(),
                    ));
                }
                v.update(groups, a, &valid, run)?;
            }
            (Kind::Vector(v), true) => {
                v.merge(groups, args)?;
            }
            (Kind::Mode(m), false) => {
                let a = args[0];
                if a.len() != groups.len() {
                    return Err(internal(
                        "a mode column of another length than the group ids".to_owned(),
                    ));
                }
                if let Some(r) = run {
                    // A running fold keeps the state after chosen rows:
                    // fold one row at a time, snapshotting after each row
                    // the frame keeps.
                    for (i, &g) in groups.iter().enumerate() {
                        let col = a.slice(i, 1);
                        m.fold(&[g], &col, &|_| valid(i))?;
                        let to = r.snapshots[i];
                        if to != Running::NONE {
                            m.snapshot(r.parts[i] as usize, to as usize);
                        }
                    }
                } else {
                    m.fold(groups, a, &valid)?;
                }
            }
            (Kind::Mode(m), true) => {
                m.merge(groups, args)?;
            }
            (Kind::ListAgg(l), false) => {
                if args.iter().any(|a| a.len() != groups.len()) {
                    return Err(internal(
                        "a listagg column of another length than the group ids".to_owned(),
                    ));
                }
                if let Some(r) = run {
                    // As for `mode`: fold one row at a time, snapshotting
                    // after each row the frame keeps.
                    for (i, &g) in groups.iter().enumerate() {
                        let row: Vec<ArrayRef> = args.iter().map(|a| a.slice(i, 1)).collect();
                        let refs: Vec<&ArrayRef> = row.iter().collect();
                        l.fold(&[g], &refs, &|_| valid(i))?;
                        let to = r.snapshots[i];
                        if to != Running::NONE {
                            l.snapshot(r.parts[i] as usize, to as usize);
                        }
                    }
                } else {
                    l.fold(groups, args, &valid)?;
                }
            }
            (Kind::ListAgg(l), true) => {
                l.merge(groups, args)?;
            }
            (Kind::Exact(acc), false) => {
                if !matches!(
                    self.expr.function,
                    AggregateFunction::Percentile { .. } | AggregateFunction::PercentileDisc { .. }
                ) {
                    return Err(internal(
                        "an exact percentile over another aggregate".to_owned(),
                    ));
                }
                let a = args[0];
                if a.len() != groups.len() {
                    return Err(internal(
                        "a quantile value column of another length than the group ids".to_owned(),
                    ));
                }
                let freq = match args.get(1) {
                    None => None,
                    Some(f) => {
                        if f.len() != groups.len() {
                            return Err(internal(
                                "a quantile frequency column of another length than the group ids"
                                    .to_owned(),
                            ));
                        }
                        Some(ints(f)?)
                    }
                };
                let freq_nulls = args.get(1).and_then(Array::logical_nulls);
                for (i, &g) in groups.iter().enumerate() {
                    if !valid(i) {
                        continue;
                    }
                    let wv = match (&freq, &freq_nulls) {
                        (None, _) => 1,
                        (Some(_), Some(n)) if !n.is_valid(i) => continue,
                        (Some(f), _) => f[i],
                    };
                    if wv < 0 {
                        return Err(negative_frequency(wv));
                    }
                    if let Some(v) = quantile::double_of(a, i)? {
                        acc.add(g as usize, v, wv);
                    }
                }
            }
            (Kind::Exact(acc), true) => {
                let [values, counts] = args else {
                    return Err(internal(
                        "an exact percentile state of other columns than (values, counts)"
                            .to_owned(),
                    ));
                };
                let (Some(values), Some(counts)) = (
                    values.as_any().downcast_ref::<ListArray>(),
                    counts.as_any().downcast_ref::<ListArray>(),
                ) else {
                    return Err(internal(
                        "an exact percentile state of other lists than (values, counts)".to_owned(),
                    ));
                };
                acc.merge(groups, values, counts)?;
            }
            (Kind::Approx(gk), false) => {
                let AggregateFunction::PercentileApprox { accuracy, .. } = &self.expr.function
                else {
                    return Err(internal(
                        "an approximate percentile over another aggregate".to_owned(),
                    ));
                };
                let relative_error = 1.0 / *accuracy as f64;
                let a = args[0];
                if a.len() != groups.len() {
                    return Err(internal(
                        "a quantile value column of another length than the group ids".to_owned(),
                    ));
                }
                for (i, &g) in groups.iter().enumerate() {
                    if let Some(v) = quantile::double_of(a, i)?
                        && valid(i)
                    {
                        gk.insert(g as usize, v, relative_error);
                    }
                }
            }
            (Kind::Approx(gk), true) => {
                let [states] = args else {
                    return Err(internal(
                        "an approximate percentile state of another column than its summary"
                            .to_owned(),
                    ));
                };
                let Some(states) = states.as_any().downcast_ref::<BinaryArray>() else {
                    return Err(internal(
                        "an approximate percentile state of another type than binary".to_owned(),
                    ));
                };
                let AggregateFunction::PercentileApprox { accuracy, .. } = &self.expr.function
                else {
                    return Err(internal(
                        "an approximate percentile over another aggregate".to_owned(),
                    ));
                };
                gk.merge(groups, states, 1.0 / *accuracy as f64)?;
            }
            (Kind::Hll(h), false) => {
                let a = args[0];
                if a.len() != groups.len() {
                    return Err(internal(
                        "a HyperLogLog value column of another length than the group ids"
                            .to_owned(),
                    ));
                }
                for (i, &g) in groups.iter().enumerate() {
                    // A row the `FILTER` mask drops is skipped like a NULL.
                    if !valid(i) {
                        continue;
                    }
                    if let Some(x) = hll::hash_of(a, i)? {
                        h.add(g as usize, x);
                    }
                }
            }
            (Kind::Hll(h), true) => {
                let [states] = args else {
                    return Err(internal(
                        "a HyperLogLog state of another column than its registers".to_owned(),
                    ));
                };
                let Some(states) = states.as_any().downcast_ref::<BinaryArray>() else {
                    return Err(internal(
                        "a HyperLogLog state of another type than binary".to_owned(),
                    ));
                };
                h.merge(groups, states)?;
            }
            (
                Kind::By {
                    max,
                    first_wins,
                    values,
                    keys,
                    set,
                    null,
                    heap,
                },
                _,
            ) => {
                let &[xa, ya] = args else {
                    return Err(internal("max_by without a value and a key".to_owned()));
                };
                if xa.len() != groups.len() || ya.len() != groups.len() {
                    return Err(internal(
                        "max_by columns of other lengths than the group ids".to_owned(),
                    ));
                }
                let (x, y) = (Col::of(xa)?, Col::of(ya)?);
                let key_valid = |i: usize| nulls[1].as_ref().is_none_or(|n| n.is_valid(i));
                for (i, &g) in groups.iter().enumerate() {
                    let g = g as usize;
                    if active(i)
                        && key_valid(i)
                        && (!set[g] || keys.beats(g, &y, i, *max, *first_wins)?)
                    {
                        keys.put(g, &y, i, heap)?;
                        if valid(i) {
                            values.put(g, &x, i, heap)?;
                        } else {
                            values.clear(g, heap);
                        }
                        set[g] = true;
                        null[g] = !valid(i);
                    }
                    if let Some(r) = run {
                        let to = r.snapshots[i];
                        if to != Running::NONE {
                            let (to, from) = (to as usize, r.parts[i] as usize);
                            values.copy_slot(from, to)?;
                            keys.copy_slot(from, to)?;
                            set[to] = set[from];
                            null[to] = null[from];
                        }
                    }
                }
            }
        }
        if let Kind::SumDec { sum, .. } | Kind::AvgDec { sum, .. } = &self.kind {
            self.max_sum = refresh_max(self.max_sum, sum, groups);
        }
        Ok(())
    }
}
