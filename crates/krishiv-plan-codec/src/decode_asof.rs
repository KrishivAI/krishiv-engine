// SPDX-License-Identifier: Apache-2.0
//! The wire form of a sort-merge join's as-of match (`SortMergeJoinNode.as_of`, codec feature `as_of_join`), checked
//! on decode.

use krishiv_exec::{AsOfMergeDirection, AsOfMergeSpec};

use crate::error::{Result, corrupt};
use crate::proto as pb;

/// An as-of join's match.
pub(crate) fn as_of_merge(a: &pb::AsOfMerge) -> Result<AsOfMergeSpec> {
    let direction = match pb::AsOfMergeDirection::try_from(a.direction) {
        Ok(pb::AsOfMergeDirection::Backward) => AsOfMergeDirection::Backward,
        Ok(pb::AsOfMergeDirection::Forward) => AsOfMergeDirection::Forward,
        _ => return Err(corrupt("an as-of join without its direction")),
    };
    let column = |c: u64| usize::try_from(c).map_err(|_| corrupt("an as-of column out of range"));
    Ok(AsOfMergeSpec {
        left_column: column(a.left_column)?,
        right_column: column(a.right_column)?,
        direction,
        allow_exact: a.allow_exact,
        distance: None,
        distance_on_every_pair: a.distance_on_every_pair,
    })
}
