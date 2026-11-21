// SPDX-License-Identifier: Apache-2.0
//! The wire form of a user-defined type, checked on decode.

use krishiv_types::DataType;

use crate::error::{Result, corrupt};
use crate::proto as pb;

/// A user-defined type from its JSON form, in the canonical text the encoder writes.
pub(crate) fn udt(t: &pb::DataType) -> Result<DataType> {
    let d = DataType::from_canonical_json(&t.json).ok();
    d.filter(|d| matches!(d, DataType::Udt(_)) && d.json() == t.json)
        .ok_or_else(|| corrupt("an invalid user-defined type"))
}
