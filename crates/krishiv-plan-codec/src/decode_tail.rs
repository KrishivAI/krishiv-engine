// SPDX-License-Identifier: Apache-2.0
//! The `Tail` node from its wire form (kept out of `decode.rs`, which is at the file-size cap).

use krishiv_exec::physical::PhysNode;

use crate::proto as pb;

/// `Tail` over the decoded `input`.
pub(crate) fn tail(input: Box<PhysNode>, t: &pb::TailNode) -> PhysNode {
    PhysNode::Tail {
        input,
        limit: t.limit,
    }
}
