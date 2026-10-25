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

#[test]
fn a_value_is_invisible_until_every_producer_partition_published() {
    let values = of(&[(SUBQUERY, 2)]);
    let read = values.subscribe(SUBQUERY).unwrap();
    assert!(read.current().is_none());
    let p = values
        .publish(partial(SUBQUERY, 1, 0, Value::Subquery(array(20))))
        .unwrap();
    assert_eq!(p, Publication::Waiting);
    assert!(read.current().is_none());
    assert_eq!(subquery(&values), None);
    let p = values
        .publish(partial(SUBQUERY, 0, 0, Value::Subquery(array(10))))
        .unwrap();
    assert_eq!(p, Publication::Complete);
    // Merged in partition order, whatever the order of arrival.
    assert_eq!(subquery(&values), Some(vec![10, 20]));
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
fn a_retried_partial_is_ignored() {
    let values = of(&[(SUBQUERY, 2)]);
    let first = partial(SUBQUERY, 0, 0, Value::Subquery(array(10)));
    assert_eq!(values.publish(first).unwrap(), Publication::Waiting);
    // Partition 0 again, from attempt 1: ignored, the value still waits.
    let retry = partial(SUBQUERY, 0, 1, Value::Subquery(array(99)));
    assert_eq!(values.publish(retry).unwrap(), Publication::Ignored);
    assert_eq!(subquery(&values), None);
    let other = partial(SUBQUERY, 1, 0, Value::Subquery(array(20)));
    assert_eq!(values.publish(other).unwrap(), Publication::Complete);
    assert_eq!(subquery(&values), Some(vec![10, 20]));
    // After completion as well.
    let late = partial(SUBQUERY, 1, 1, Value::Subquery(array(99)));
    assert_eq!(values.publish(late).unwrap(), Publication::Ignored);
    assert_eq!(subquery(&values), Some(vec![10, 20]));
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

fn broadcast(values: &LocalValues, id: ValueId) -> Arc<dyn ExecNode> {
    let (_, v) = values.subscribe(id).unwrap().current().unwrap();
    let Value::Broadcast(b) = v.as_ref() else {
        panic!("{v:?}")
    };
    assert_eq!(format!("{b:?}"), "BroadcastBuild");
    Arc::clone(&b.0)
}

#[test]
fn broadcast_partials_are_one_input() {
    let id = ValueId::Broadcast(ExchangeId(3));
    let inputs: [Arc<dyn ExecNode>; 2] = [Arc::new(Input), Arc::new(Input)];
    let input = |i: usize| Value::Broadcast(BroadcastBuild(Arc::clone(&inputs[i])));
    // One producer partition: its input is the value.
    let one = LocalValues::new();
    one.publish(partial(id, 0, 0, input(0))).unwrap();
    assert!(Arc::ptr_eq(&broadcast(&one, id), &inputs[0]));
    // Two: the union of their inputs.
    let two = of(&[(id, 2)]);
    assert_eq!(
        two.publish(partial(id, 1, 0, input(1))).unwrap(),
        Publication::Waiting
    );
    assert_eq!(
        two.publish(partial(id, 0, 0, input(0))).unwrap(),
        Publication::Complete
    );
    let both = broadcast(&two, id);
    assert!(!Arc::ptr_eq(&both, &inputs[0]) && !Arc::ptr_eq(&both, &inputs[1]));
    let mut b = PipelineBuilder::new(Arc::new(crate::testing::Count::default()));
    both.build_pipelines(&mut b).unwrap();
}

#[test]
fn malformed_partials_are_internal_errors() {
    let values = of(&[(SUBQUERY, 2)]);
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
        partial(SUBQUERY, 2, 0, Value::Subquery(array(1))),
        "producer partition 2",
    );
    values
        .publish(partial(SUBQUERY, 0, 0, Value::Subquery(array(1))))
        .unwrap();
    check(
        Partial {
            version: Version { epoch: 1 },
            ..partial(SUBQUERY, 1, 0, Value::Subquery(array(1)))
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
