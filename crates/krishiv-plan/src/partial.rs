// SPDX-License-Identifier: Apache-2.0
//! The state columns of a [`PlanNode::PartialAggregate`](crate::PlanNode).
//!
//! The executor owns the states; this names their types for the aggregates
//! whose states are simple enough to move below a join, so the optimizer can
//! type the node. The physical planner refuses a plan where the two disagree.

use krishiv_types::{DataType, DecimalType};

use crate::{AttrId, BoundColumn, FunctionId, FunctionKind, ResolvedFunction};

/// The state columns of `call` over a raw argument of type `raw_type`, numbered
/// from `first`, or `None` when its state cannot be computed apart from its
/// result: anything but the registry's aggregates `count`, `sum`, `avg`, `min`
/// and `max`, a `DISTINCT` or a `FILTER`.
#[must_use]
pub fn partial_state_columns(
    call: &ResolvedFunction,
    raw_type: &DataType,
    first: AttrId,
) -> Option<Vec<BoundColumn>> {
    if call.distinct
        || call.filter.is_some()
        || !call.within_group.is_empty()
        || call.kind != FunctionKind::Aggregate
        || !matches!(call.id, FunctionId::Registry { .. })
    {
        return None;
    }
    let name = call.id.name();
    let out = call.data_type.clone();
    let parts: Vec<(&str, DataType, bool)> = match (name, &out) {
        ("count", _) => vec![("count", DataType::Long, false)],
        ("sum", DataType::Decimal(_)) => {
            vec![("sum", out, true), ("nonempty", DataType::Boolean, false)]
        }
        ("sum", _) => vec![("sum", out, true)],
        ("avg" | "mean", _) => vec![
            ("sum", avg_sum_type(raw_type)?, true),
            ("count", DataType::Long, false),
        ],
        ("min" | "max", _) => vec![("value", out, true)],
        _ => return None,
    };
    let mut columns = Vec::with_capacity(parts.len());
    for (offset, (suffix, data_type, nullable)) in (0_u64..).zip(parts) {
        columns.push(BoundColumn {
            id: AttrId(first.0.checked_add(offset)?),
            name: format!("{name}#{suffix}"),
            data_type,
            nullable,
        });
    }
    Some(columns)
}

/// The running sum of an `avg`: a decimal widened by 10 digits, else a double.
fn avg_sum_type(raw_type: &DataType) -> Option<DataType> {
    Some(match raw_type {
        DataType::Decimal(d) => {
            let precision = (u32::from(d.precision()) + 10).min(38);
            DataType::Decimal(DecimalType::try_new(precision, u32::from(d.scale())).ok()?)
        }
        _ => DataType::Double,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, data_type: DataType) -> ResolvedFunction {
        ResolvedFunction {
            id: FunctionId::Registry {
                name: name.to_owned(),
                id: 1,
            },
            kind: FunctionKind::Aggregate,
            args: Vec::new(),
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type,
            nullable: true,
            deterministic: true,
        }
    }

    fn decimal(p: u32, s: u32) -> DataType {
        DataType::Decimal(DecimalType::try_new(p, s).unwrap())
    }

    fn parts(call: &ResolvedFunction, raw: &DataType) -> Vec<(String, DataType, bool)> {
        partial_state_columns(call, raw, AttrId(7))
            .unwrap()
            .into_iter()
            .map(|c| (c.name, c.data_type, c.nullable))
            .collect()
    }

    #[test]
    fn states_match_the_functions() {
        let long = DataType::Long;
        let n = |s: &str| s.to_owned();
        assert_eq!(
            parts(&call("count", long.clone()), &DataType::Integer),
            [(n("count#count"), long.clone(), false)]
        );
        assert_eq!(
            parts(&call("sum", long.clone()), &DataType::Integer),
            [(n("sum#sum"), long.clone(), true)]
        );
        assert_eq!(
            parts(&call("sum", decimal(20, 2)), &decimal(10, 2)),
            [
                (n("sum#sum"), decimal(20, 2), true),
                (n("sum#nonempty"), DataType::Boolean, false)
            ]
        );
        assert_eq!(
            parts(&call("avg", DataType::Double), &DataType::Integer),
            [
                (n("avg#sum"), DataType::Double, true),
                (n("avg#count"), long.clone(), false)
            ]
        );
        assert_eq!(
            parts(&call("mean", decimal(14, 6)), &decimal(36, 2)),
            [
                (n("mean#sum"), decimal(38, 2), true),
                (n("mean#count"), long, false)
            ]
        );
        assert_eq!(
            parts(&call("max", DataType::String), &DataType::String),
            [(n("max#value"), DataType::String, true)]
        );
    }

    #[test]
    fn states_are_numbered_from_first() {
        let avg = call("avg", DataType::Double);
        let columns = partial_state_columns(&avg, &DataType::Long, AttrId(7)).unwrap();
        let ids: Vec<u64> = columns.iter().map(|c| c.id.0).collect();
        assert_eq!(ids, [7, 8]);
        assert!(partial_state_columns(&avg, &DataType::Long, AttrId(u64::MAX)).is_none());
    }

    #[test]
    fn other_calls_have_no_state_here() {
        let raw = DataType::Long;
        assert!(partial_state_columns(&call("first", raw.clone()), &raw, AttrId(0)).is_none());
        assert!(partial_state_columns(&call("stddev", raw.clone()), &raw, AttrId(0)).is_none());
        let mut distinct = call("sum", raw.clone());
        distinct.distinct = true;
        assert!(partial_state_columns(&distinct, &raw, AttrId(0)).is_none());
        let mut filtered = call("sum", raw.clone());
        filtered.filter = Some(std::sync::Arc::new(crate::Expr::new(
            crate::ExprNode::Literal(crate::Literal::Boolean(true)),
        )));
        assert!(partial_state_columns(&filtered, &raw, AttrId(0)).is_none());
        let mut window = call("sum", raw.clone());
        window.kind = FunctionKind::Window;
        assert!(partial_state_columns(&window, &raw, AttrId(0)).is_none());
        let mut builtin = call("sum", raw.clone());
        builtin.id = FunctionId::Builtin(crate::Builtin::Add);
        assert!(partial_state_columns(&builtin, &raw, AttrId(0)).is_none());
    }
}
