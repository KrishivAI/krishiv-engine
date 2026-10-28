// SPDX-License-Identifier: Apache-2.0
//! The value service: completeness over producer partitions, retries,
//! filters that never make a consumer wait, and errors.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array};
use arrow_schema::DataType;
use krishiv_common::error::ErrorClass;
use krishiv_io::{
    ColumnPath, DisabledReason, FilterId, FilterSnapshot, KeyFilterBuilder, KeyFilterOptions,
};

use super::{
    BroadcastBuild, LocalValues, Partial, Producer, Publication, Value, ValueId, ValueService,
    Version,
};
use crate::physical::{ExchangeId, SubqueryId};
use crate::pipeline::{ExecNode, PipelineBuilder};

const SUBQUERY: ValueId = ValueId::Subquery(SubqueryId(1));
const FILTER: ValueId = ValueId::Filter(FilterId(1));
const BROADCAST: ValueId = ValueId::Broadcast(ExchangeId(3));

fn array(v: i64) -> ArrayRef {
    Arc::new(Int64Array::from(vec![v]))
}

fn partial(value: ValueId, partition: u32, attempt: u32, payload: Value) -> Partial {
    Partial {
        value,
        version: Version::BATCH,
        producer: Producer { partition, attempt },
        payload,
    }
}

fn of(producers: &[(ValueId, u32)]) -> LocalValues {
    LocalValues::with_producers(producers.iter().copied().collect()).unwrap()
}

fn subquery(values: &LocalValues) -> Option<Vec<i64>> {
    let (version, v) = values.subscribe(SUBQUERY).unwrap().current()?;
    assert_eq!(version, Version::BATCH);
    let Value::Subquery(a) = v.as_ref() else {
        panic!("{v:?}")
    };
    let Some(a) = a.as_any().downcast_ref::<Int64Array>() else {
        panic!("subquery column is not Int64")
    };
    Some(a.values().to_vec())
}

/// A filter on keys `keys`: drops every other key.
fn keys(keys: &[i64]) -> FilterSnapshot {
    let mut b = KeyFilterBuilder::new(DataType::Int64, KeyFilterOptions::default()).unwrap();
    b.add(&Int64Array::from(keys.to_vec()), None).unwrap();
    FilterSnapshot::Final(Arc::new(b.finish()))
}

/// Broadcast inputs, told apart by pointer.
fn inputs<const N: usize>() -> [Arc<dyn ExecNode>; N] {
    std::array::from_fn(|_| Arc::new(Input) as Arc<dyn ExecNode>)
}

fn build(input: &Arc<dyn ExecNode>) -> Value {
    Value::Broadcast(BroadcastBuild::new(input))
}

/// Whether the value `id` is `expected`'s inputs, in that order.
fn is_broadcast_of(values: &LocalValues, id: ValueId, expected: &[&Arc<dyn ExecNode>]) -> bool {
    let Some((_, v)) = values.subscribe(id).unwrap().current() else {
        return false;
    };
    let Value::Broadcast(b) = v.as_ref() else {
        panic!("{v:?}")
    };
    assert_eq!(format!("{b:?}"), "BroadcastBuild");
    let got = b.inputs().unwrap();
    got.len() == expected.len() && got.iter().zip(expected).all(|(g, e)| Arc::ptr_eq(g, e))
}

#[test]
fn a_value_is_invisible_until_every_producer_partition_published() {
    let [i0, i1] = inputs();
    let values = of(&[(BROADCAST, 2)]);
    let read = values.subscribe(BROADCAST).unwrap();
    assert!(read.current().is_none());
    let p = values
        .publish(partial(BROADCAST, 1, 0, build(&i1)))
        .unwrap();
    assert_eq!(p, Publication::Waiting);
    assert!(read.current().is_none());
    let p = values
        .publish(partial(BROADCAST, 0, 0, build(&i0)))
        .unwrap();
    assert_eq!(p, Publication::Complete);
    // Merged in partition order, whatever the order of arrival.
    assert!(is_broadcast_of(&values, BROADCAST, &[&i0, &i1]));
    assert!(read.current().is_some());
}

#[test]
fn one_producer_partition_by_default() {
    let values = LocalValues::new();
    let p = values
        .publish(partial(SUBQUERY, 0, 0, Value::Subquery(array(7))))
        .unwrap();
    assert_eq!(p, Publication::Complete);
    assert_eq!(subquery(&values), Some(vec![7]));
}

#[test]
fn a_subquery_has_one_producer() {
    let e = LocalValues::with_producers(BTreeMap::from([(SUBQUERY, 2)])).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
    assert!(e.to_string().contains("a subquery has one"), "{e}");
    // A second publication of a subquery is ignored (its sink reports it).
    let values = LocalValues::new();
    let first = partial(SUBQUERY, 0, 0, Value::Subquery(array(1)));
    assert_eq!(values.publish(first).unwrap(), Publication::Complete);
    let again = partial(SUBQUERY, 0, 0, Value::Subquery(array(2)));
    assert_eq!(values.publish(again).unwrap(), Publication::Ignored);
    assert_eq!(subquery(&values), Some(vec![1]));
}

#[test]
fn a_retried_partial_is_ignored() {
    let [i0, i1, retry] = inputs();
    let values = of(&[(BROADCAST, 2)]);
    let first = partial(BROADCAST, 0, 0, build(&i0));
    assert_eq!(values.publish(first).unwrap(), Publication::Waiting);
    // Partition 0 again, from attempt 1: ignored, the value still waits.
    let again = partial(BROADCAST, 0, 1, build(&retry));
    assert_eq!(values.publish(again).unwrap(), Publication::Ignored);
    assert!(!is_broadcast_of(&values, BROADCAST, &[]));
    let other = partial(BROADCAST, 1, 0, build(&i1));
    assert_eq!(values.publish(other).unwrap(), Publication::Complete);
    assert!(is_broadcast_of(&values, BROADCAST, &[&i0, &i1]));
    // After completion as well.
    let late = partial(BROADCAST, 1, 1, build(&retry));
    assert_eq!(values.publish(late).unwrap(), Publication::Ignored);
    assert!(is_broadcast_of(&values, BROADCAST, &[&i0, &i1]));
}

#[test]
fn a_filter_stays_pending_until_complete() {
    let values = of(&[(FILTER, 2)]);
    values
        .declare_filter(FilterId(1), ColumnPath::new("k"), DataType::Int64)
        .unwrap();
    let cell = values.filter(FilterId(1)).unwrap();
    let read = values.subscribe(FILTER).unwrap();
    // One partition's filter alone would drop the other's keys: it is
    // not applied.
    let p = values
        .publish(partial(FILTER, 0, 0, Value::Filter(keys(&[1]))))
        .unwrap();
    assert_eq!(p, Publication::Waiting);
    assert!(matches!(cell.published().snapshot, FilterSnapshot::Pending));
    assert_eq!(cell.generation_hint(), 0);
    assert!(read.current().is_none());
    let p = values
        .publish(partial(FILTER, 1, 0, Value::Filter(keys(&[2]))))
        .unwrap();
    assert_eq!(p, Publication::Complete);
    // Filters do not merge yet: the complete value applies nothing.
    assert!(matches!(
        cell.published().snapshot,
        FilterSnapshot::Disabled(DisabledReason::Off)
    ));
    assert!(matches!(
        read.current().unwrap().1.as_ref(),
        Value::Filter(FilterSnapshot::Disabled(_))
    ));
}

#[test]
fn a_disabled_partial_disables_the_filter() {
    let values = of(&[(FILTER, 2)]);
    values
        .declare_filter(FilterId(1), ColumnPath::new("k"), DataType::Int64)
        .unwrap();
    let failed = FilterSnapshot::Disabled(DisabledReason::ProducerFailed);
    values
        .publish(partial(FILTER, 1, 0, Value::Filter(failed)))
        .unwrap();
    values
        .publish(partial(FILTER, 0, 0, Value::Filter(keys(&[1]))))
        .unwrap();
    assert!(matches!(
        values.filter(FilterId(1)).unwrap().published().snapshot,
        FilterSnapshot::Disabled(DisabledReason::ProducerFailed)
    ));
}

#[test]
fn a_filter_of_one_producer_is_published_to_its_cell() {
    let values = LocalValues::new();
    values
        .declare_filter(FilterId(1), ColumnPath::new("k"), DataType::Int64)
        .unwrap();
    let p = values
        .publish(partial(FILTER, 0, 0, Value::Filter(keys(&[1]))))
        .unwrap();
    assert_eq!(p, Publication::Complete);
    let cell = values.filter(FilterId(1)).unwrap();
    assert!(matches!(
        cell.published().snapshot,
        FilterSnapshot::Final(_)
    ));
    // The failure path after publishing changes nothing.
    let failed = FilterSnapshot::Disabled(DisabledReason::ProducerFailed);
    let p = values
        .publish(partial(FILTER, 0, 0, Value::Filter(failed)))
        .unwrap();
    assert_eq!(p, Publication::Ignored);
    assert!(matches!(
        cell.published().snapshot,
        FilterSnapshot::Final(_)
    ));
}

#[test]
fn services_never_share_values() {
    let (a, b) = (LocalValues::new(), LocalValues::new());
    for v in [&a, &b] {
        v.declare_filter(FilterId(1), ColumnPath::new("k"), DataType::Int64)
            .unwrap();
    }
    a.publish(partial(FILTER, 0, 0, Value::Filter(keys(&[1]))))
        .unwrap();
    a.publish(partial(SUBQUERY, 0, 0, Value::Subquery(array(1))))
        .unwrap();
    assert!(!Arc::ptr_eq(
        &a.filter(FilterId(1)).unwrap(),
        &b.filter(FilterId(1)).unwrap()
    ));
    assert!(matches!(
        b.filter(FilterId(1)).unwrap().published().snapshot,
        FilterSnapshot::Pending
    ));
    assert_eq!(subquery(&b), None);
}

/// A broadcast input.
struct Input;

impl ExecNode for Input {
    fn build_pipelines(self: Arc<Self>, _b: &mut PipelineBuilder) -> crate::Result<()> {
        Ok(())
    }
}

#[test]
fn a_broadcast_value_does_not_own_its_inputs() {
    let [input] = inputs();
    let values = LocalValues::new();
    values
        .publish(partial(BROADCAST, 0, 0, build(&input)))
        .unwrap();
    assert!(is_broadcast_of(&values, BROADCAST, &[&input]));
    // The input may hold the service: owning it here would be a cycle.
    drop(input);
    let (_, v) = values.subscribe(BROADCAST).unwrap().current().unwrap();
    let Value::Broadcast(b) = v.as_ref() else {
        panic!("{v:?}")
    };
    assert!(b.inputs().is_none());
}

#[test]
fn malformed_partials_are_internal_errors() {
    let [input] = inputs();
    let values = of(&[(BROADCAST, 2)]);
    values
        .declare_filter(FilterId(1), ColumnPath::new("k"), DataType::Int64)
        .unwrap();
    let check = |p: Partial, text: &str| {
        let e = values.publish(p).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
        assert!(e.to_string().contains(text), "{e}");
    };
    check(
        partial(SUBQUERY, 0, 0, Value::Filter(keys(&[1]))),
        "another kind",
    );
    check(
        partial(FILTER, 0, 0, Value::Filter(FilterSnapshot::Pending)),
        "pending partial",
    );
    check(
        partial(
            ValueId::Filter(FilterId(9)),
            0,
            0,
            Value::Filter(keys(&[1])),
        ),
        "has no scan",
    );
    check(
        partial(BROADCAST, 2, 0, build(&input)),
        "producer partition 2",
    );
    values
        .publish(partial(BROADCAST, 0, 0, build(&input)))
        .unwrap();
    check(
        Partial {
            version: Version { epoch: 1 },
            ..partial(BROADCAST, 1, 0, build(&input))
        },
        "epochs 0 and 1",
    );
    let e = values
        .declare_filter(FilterId(1), ColumnPath::new("k"), DataType::Int64)
        .unwrap_err();
    assert!(e.to_string().contains("declared twice"), "{e}");
    let e = values.subscribe(ValueId::Filter(FilterId(9))).unwrap_err();
    assert!(e.to_string().contains("has no scan"), "{e}");
    let e = values.filter(FilterId(9)).unwrap_err();
    assert!(e.to_string().contains("has no scan"), "{e}");
    let e = LocalValues::with_producers(BTreeMap::from([(SUBQUERY, 0)])).unwrap_err();
    assert!(e.to_string().contains("has no producer"), "{e}");
}

#[test]
fn a_join_publishes_only_into_its_services_cells() {
    use crate::join::{HashJoin, JoinRuntimeFilter, JoinSpec, JoinType};
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "k",
        DataType::Int64,
        true,
    )]));
    let join = || {
        HashJoin::try_new(JoinSpec {
            join_type: JoinType::Inner,
            probe_schema: Arc::clone(&schema),
            build_schema: Arc::clone(&schema),
            probe_keys: vec![0],
            build_keys: vec![0],
            null_safe: vec![false],
            filter: None,
            single_match: false,
            spill: None,
        })
        .unwrap()
    };
    let values = Arc::new(LocalValues::new());
    values
        .declare_filter(FilterId(1), ColumnPath::new("k"), DataType::Int64)
        .unwrap();
    let filter = |cell| JoinRuntimeFilter {
        key: 0,
        filter: cell,
        options: KeyFilterOptions::default(),
    };
    // A cell of its own is not the service's.
    let foreign = Arc::new(krishiv_io::RuntimeFilter::new(
        FilterId(1),
        ColumnPath::new("k"),
        DataType::Int64,
    ));
    let e = join()
        .with_filter_values(
            vec![filter(foreign)],
            Arc::clone(&values) as _,
            Producer::default(),
        )
        .unwrap_err();
    assert!(e.to_string().contains("not the value service's"), "{e}");
    // The service's cell is taken; two filters of one id are not.
    let cell = values.filter(FilterId(1)).unwrap();
    join()
        .with_filter_values(
            vec![filter(Arc::clone(&cell))],
            Arc::clone(&values) as _,
            Producer::default(),
        )
        .unwrap();
    let e = join()
        .with_runtime_filters(vec![filter(Arc::clone(&cell)), filter(cell)])
        .unwrap_err();
    assert!(e.to_string().contains("declared twice"), "{e}");
}
