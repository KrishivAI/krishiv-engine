// SPDX-License-Identifier: Apache-2.0
//! The sort-with-comparator higher-order function: stable merge sort and its state machine.

use super::*;

/// A bound `array_sort` (see the module doc): without a comparator the
/// array sorts in ascending order with NULLs last (Spark's default
/// comparator); with one, a stable merge sort over the list's offsets
/// calls it per comparison (a NULL result is a `COMPARATOR_RETURNS_NULL`
/// error, as Spark raises it).
#[derive(Debug)]
pub(super) struct Sort {
    pub(super) array: Arc<dyn PhysExpr>,
    pub(super) comparator: Option<Arc<dyn PhysExpr>>,
    pub(super) schema: Arc<Schema>,
    pub(super) sources: Vec<Source>,
    pub(super) out: ArrowType,
    pub(super) nullable: bool,
}

/// Compiles `array_sort(arr[, (l, r) -> ...])`.
pub(super) fn compile_sort(
    _name: &str,
    f: &ResolvedFunction,
    input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    if f.args.len() != 1 && f.args.len() != 2 {
        return Err(internal(
            "a higher-order call without its values and lambda",
        ));
    }
    let array = super::super::compile(&f.args[0], input)?;
    let ArrowType::List(_) = array.data_type().clone() else {
        return Err(internal("a higher-order call over a non-list"));
    };
    if f.args.len() == 1 {
        return Ok(Arc::new(Sort {
            array,
            comparator: None,
            schema: Arc::new(Schema::empty()),
            sources: Vec::new(),
            out: f.data_type.to_arrow(),
            nullable: f.nullable,
        }));
    }
    let (params, body) = lambda_of(f, 1, 1)?;
    if params.len() != 2 {
        return Err(internal("a higher-order call without a lambda"));
    }
    let mut refs = Refs::default();
    collect(body, params, &mut Vec::new(), &mut refs, 2, None);
    // Both parameters are bound per comparison, never flattened: the
    // layout below carries the captured columns first (from `flat_parts`)
    // and the two elements after (their types ride the analyzer's
    // variables).
    let (mut layout, schema, mut sources) = flat_parts(input, &refs, params, &[], 2, None, false)?;
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    for (var, at) in params.iter().take(2).zip([0, 1]) {
        layout.push(var.id);
        fields.push(Field::new(
            format!("c{}", fields.len()),
            var.data_type.to_arrow(),
            var.nullable,
        ));
        sources.push(Source::Element(at));
    }
    let schema = Arc::new(Schema::new(fields));
    let body = compile_body(body, &layout, &schema, input)?;
    Ok(Arc::new(Sort {
        array,
        comparator: Some(body),
        schema,
        sources,
        out: f.data_type.to_arrow(),
        nullable: f.nullable,
    }))
}

/// Merges the adjacent runs `left` and `right` of `perm` stably through
/// `buf`: on ties the left run's element goes first. A comparison failure
/// aborts the sort with its error.
pub(super) fn merge_runs(
    perm: &mut [usize],
    buf: &mut [usize],
    left: (usize, usize),
    right: (usize, usize),
    cmp: &dyn Fn(usize, usize) -> Result<Ordering, KrishivError>,
) -> Result<()> {
    let (mut i, mut j, mut k) = (left.0, right.0, left.0);
    while i < left.1 && j < right.1 {
        if cmp(perm[i], perm[j])? != Ordering::Greater {
            buf[k] = perm[i];
            i += 1;
        } else {
            buf[k] = perm[j];
            j += 1;
        }
        k += 1;
    }
    while i < left.1 {
        buf[k] = perm[i];
        i += 1;
        k += 1;
    }
    while j < right.1 {
        buf[k] = perm[j];
        j += 1;
        k += 1;
    }
    perm[left.0..right.1].copy_from_slice(&buf[left.0..right.1]);
    Ok(())
}

/// Stably sorts `perm` by `cmp`: ascending runs (and strictly descending
/// ones, reversed) merge pairwise, so an already ordered run never moves
/// (a constant comparator leaves every array in place, as Spark's sort
/// does).
pub(super) fn stable_sort(
    perm: &mut [usize],
    cmp: &dyn Fn(usize, usize) -> Result<Ordering, KrishivError>,
) -> Result<()> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < perm.len() {
        let mut j = i + 1;
        if j < perm.len() && cmp(perm[j], perm[i])? == Ordering::Less {
            // Strictly descending: extend, then reverse back to ascending.
            j += 1;
            while j < perm.len() && cmp(perm[j], perm[j - 1])? == Ordering::Less {
                j += 1;
            }
            perm[i..j].reverse();
        } else {
            while j < perm.len() && cmp(perm[j], perm[j - 1])? != Ordering::Less {
                j += 1;
            }
        }
        runs.push((i, j));
        i = j;
    }
    let mut buf = vec![0_usize; perm.len()];
    while runs.len() > 1 {
        let mut next: Vec<(usize, usize)> = Vec::with_capacity(runs.len().div_ceil(2));
        let (pairs, rest) = runs.as_chunks::<2>();
        for pair in pairs {
            merge_runs(perm, &mut buf, pair[0], pair[1], cmp)?;
            next.push((pair[0].0, pair[1].1));
        }
        next.extend(rest.iter().copied());
        runs = next;
    }
    Ok(())
}

/// Where a [`Machine`] is in the sort of one array (the sequential
/// algorithm of [`stable_sort`], one comparison at a time).
#[derive(Debug)]
pub(super) enum Phase {
    /// Looking for the next run from `i`.
    Start {
        i: usize,
    },
    /// The first pair of a run at `i` is out for comparison.
    First {
        i: usize,
    },
    /// Extending the run `i..j`, strictly descending when `desc`; the pair
    /// at `j` is out for comparison.
    Extend {
        i: usize,
        j: usize,
        desc: bool,
    },
    /// Merging the pair of runs at `pair`; the heads `i` and `j` are out.
    Merge {
        pair: usize,
        i: usize,
        j: usize,
        k: usize,
    },
    Done,
}

/// One array's sort as a state machine, so every array's pending
/// comparison can be evaluated in one batch per round: [`Machine::request`]
/// names the elements to compare next and [`Machine::answer`] takes the
/// result. The order of comparisons and the outcome are those of
/// [`stable_sort`].
#[derive(Debug)]
pub(super) struct Machine {
    pub(super) perm: Vec<usize>,
    pub(super) buf: Vec<usize>,
    pub(super) runs: Vec<(usize, usize)>,
    pub(super) next: Vec<(usize, usize)>,
    pub(super) phase: Phase,
}

impl Machine {
    pub(super) fn new(perm: Vec<usize>) -> Self {
        let n = perm.len();
        Self {
            perm,
            buf: vec![0; n],
            runs: Vec::new(),
            next: Vec::new(),
            phase: Phase::Start { i: 0 },
        }
    }

    /// The pair of elements to compare next, or `None` when sorted.
    pub(super) fn request(&mut self) -> Option<(usize, usize)> {
        loop {
            match self.phase {
                Phase::Done => return None,
                Phase::Start { i } => {
                    if i >= self.perm.len() {
                        self.begin_round();
                        continue;
                    }
                    if i + 1 >= self.perm.len() {
                        self.runs.push((i, i + 1));
                        self.phase = Phase::Start { i: i + 1 };
                        continue;
                    }
                    self.phase = Phase::First { i };
                    return Some((self.perm[i + 1], self.perm[i]));
                }
                Phase::First { i } => return Some((self.perm[i + 1], self.perm[i])),
                Phase::Extend { i, j, desc } => {
                    if j >= self.perm.len() {
                        self.finish_run(i, j, desc);
                        continue;
                    }
                    return Some((self.perm[j], self.perm[j - 1]));
                }
                Phase::Merge { pair, i, j, k } => {
                    let Some(&left) = self.runs.get(2 * pair) else {
                        self.end_round();
                        continue;
                    };
                    let Some(&right) = self.runs.get(2 * pair + 1) else {
                        self.next.push(left);
                        self.end_round();
                        continue;
                    };
                    if i < left.1 && j < right.1 {
                        return Some((self.perm[i], self.perm[j]));
                    }
                    let (mut i, mut j, mut k) = (i, j, k);
                    while i < left.1 {
                        self.buf[k] = self.perm[i];
                        i += 1;
                        k += 1;
                    }
                    while j < right.1 {
                        self.buf[k] = self.perm[j];
                        j += 1;
                        k += 1;
                    }
                    self.perm[left.0..right.1].copy_from_slice(&self.buf[left.0..right.1]);
                    self.next.push((left.0, right.1));
                    self.start_pair(pair + 1);
                }
            }
        }
    }

    /// Takes the result of the comparison [`Machine::request`] named.
    pub(super) fn answer(&mut self, ord: Ordering) {
        match self.phase {
            Phase::First { i } => {
                self.phase = Phase::Extend {
                    i,
                    j: i + 2,
                    desc: ord == Ordering::Less,
                };
            }
            Phase::Extend { i, j, desc } => {
                if (ord == Ordering::Less) == desc {
                    self.phase = Phase::Extend { i, j: j + 1, desc };
                } else {
                    self.finish_run(i, j, desc);
                }
            }
            Phase::Merge { pair, i, j, k } => {
                if ord != Ordering::Greater {
                    self.buf[k] = self.perm[i];
                    self.phase = Phase::Merge {
                        pair,
                        i: i + 1,
                        j,
                        k: k + 1,
                    };
                } else {
                    self.buf[k] = self.perm[j];
                    self.phase = Phase::Merge {
                        pair,
                        i,
                        j: j + 1,
                        k: k + 1,
                    };
                }
            }
            Phase::Start { .. } | Phase::Done => {}
        }
    }

    pub(super) fn finish_run(&mut self, i: usize, j: usize, desc: bool) {
        if desc {
            self.perm[i..j].reverse();
        }
        self.runs.push((i, j));
        self.phase = Phase::Start { i: j };
    }

    pub(super) fn begin_round(&mut self) {
        if self.runs.len() > 1 {
            self.next.clear();
            self.start_pair(0);
        } else {
            self.phase = Phase::Done;
        }
    }

    pub(super) fn end_round(&mut self) {
        self.runs = std::mem::take(&mut self.next);
        self.begin_round();
    }

    pub(super) fn start_pair(&mut self, pair: usize) {
        let (i, j) = match (self.runs.get(2 * pair), self.runs.get(2 * pair + 1)) {
            (Some(l), Some(r)) => (l.0, r.0),
            _ => (0, 0),
        };
        let k = i;
        self.phase = Phase::Merge { pair, i, j, k };
    }
}

impl PhysExpr for Sort {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let rows = batch.num_rows();
        let array = self.array.eval(batch, partition)?.into_array(rows)?;
        if array.len() != rows {
            return Err(internal("a higher-order array of the wrong row count"));
        }
        let list = array
            .as_list_opt::<i32>()
            .ok_or_else(|| internal("a higher-order array that is no list"))?;
        let list = rebase_list(list, rows)?;
        let offsets = list.value_offsets();
        let values = list.values();
        // One permutation per row; every row keeps its length (a NULL row
        // keeps its elements, unsorted), so the offsets rebuild unchanged.
        let mut perms: Vec<Vec<usize>> = Vec::with_capacity(rows);
        for r in 0..rows {
            let (start, stop) = (
                usize::try_from(offsets[r])
                    .map_err(|_| internal("a higher-order list with a negative offset"))?,
                usize::try_from(offsets[r + 1])
                    .map_err(|_| internal("a higher-order list with a negative offset"))?,
            );
            perms.push((start..stop).collect());
        }
        if let Some(comparator) = &self.comparator {
            // The comparator over pairs: the captured row with the two
            // elements. All arrays advance together, one comparison each
            // per round, evaluated as one batch. A NULL result is Spark's
            // `COMPARATOR_RETURNS_NULL` (both values render the left
            // element, as Spark's call does).
            let mut machines: Vec<(usize, Machine)> = Vec::new();
            for (r, perm) in perms.iter_mut().enumerate() {
                if list.is_valid(r) && perm.len() > 1 {
                    machines.push((r, Machine::new(std::mem::take(perm))));
                }
            }
            loop {
                let mut asked: Vec<(usize, usize, usize)> = Vec::new();
                let mut slots: Vec<usize> = Vec::new();
                for (m, (r, machine)) in machines.iter_mut().enumerate() {
                    if let Some((a, b)) = machine.request() {
                        asked.push((*r, a, b));
                        slots.push(m);
                    }
                }
                if asked.is_empty() {
                    break;
                }
                let to_u32 = |n: usize| u32::try_from(n).map_err(|_| too_many(n));
                let roots = UInt32Array::from(
                    asked
                        .iter()
                        .map(|&(r, ..)| to_u32(r))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                let lefts = UInt32Array::from(
                    asked
                        .iter()
                        .map(|&(_, a, _)| to_u32(a))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                let rights = UInt32Array::from(
                    asked
                        .iter()
                        .map(|&(_, _, b)| to_u32(b))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                let take = |column: &ArrayRef, at: &UInt32Array| {
                    arrow_select::take::take(column.as_ref(), at, None)
                        .map_err(|e| internal(&e.to_string()))
                };
                let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.sources.len());
                for source in &self.sources {
                    columns.push(match source {
                        Source::Root(index) => take(batch.column(*index), &roots)?,
                        Source::Element(0) => take(values, &lefts)?,
                        Source::Element(_) => take(values, &rights)?,
                        Source::Index | Source::Acc => {
                            return Err(internal("a higher-order fold outside its fold"));
                        }
                    });
                }
                let pairs = RecordBatch::try_new(Arc::clone(&self.schema), columns)
                    .map_err(|e| internal(&e.to_string()))?;
                let cmp = comparator
                    .eval(&pairs, partition)?
                    .into_array(asked.len())?;
                let cmp = cmp
                    .as_primitive_opt::<arrow_array::types::Int32Type>()
                    .ok_or_else(|| internal("a comparator that is no INT"))?;
                for (n, (&m, &(_, a, _))) in slots.iter().zip(&asked).enumerate() {
                    if cmp.is_null(n) {
                        return Err(KrishivError::from(ComparatorReturnsNull {
                            first_value: render_value(values.as_ref(), a),
                            second_value: render_value(values.as_ref(), a),
                        }));
                    }
                    machines[m].1.answer(cmp.value(n).cmp(&0));
                }
            }
            for (r, machine) in machines {
                perms[r] = machine.perm;
            }
        } else {
            // Ascending with NULLs last (Spark's default comparator): NULL
            // is greater than every value.
            let one = |a: usize, b: usize| -> Result<Ordering, KrishivError> {
                match (values.is_null(a), values.is_null(b)) {
                    (true, true) => Ok(Ordering::Equal),
                    (true, false) => Ok(Ordering::Greater),
                    (false, true) => Ok(Ordering::Less),
                    (false, false) => {
                        cmp_rows(values.as_ref(), a, values.as_ref(), b).map_err(Into::into)
                    }
                }
            };
            for (r, perm) in perms.iter_mut().enumerate() {
                if list.is_valid(r) {
                    stable_sort(perm, &one)?;
                }
            }
        }
        let mut taken: Vec<u32> = Vec::new();
        for perm in &perms {
            for &e in perm {
                taken.push(u32::try_from(e).map_err(|_| too_many(e))?);
            }
        }
        let sorted = arrow_select::take::take(values.as_ref(), &UInt32Array::from(taken), None)
            .map_err(|e| internal(&e.to_string()))?;
        let ArrowType::List(field) = &self.out else {
            return Err(internal("a higher-order call over a non-list"));
        };
        let out = Arc::new(ListArray::new(
            Arc::clone(field),
            list.offsets().clone(),
            sorted,
            list.nulls().cloned(),
        )) as ArrayRef;
        if out.data_type() != &self.out {
            return Err(internal(&format!(
                "array_sort returned {} where the analyzer typed {}",
                out.data_type(),
                self.out
            )));
        }
        Ok(Value::Array(out))
    }
}
