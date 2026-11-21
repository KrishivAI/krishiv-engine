// SPDX-License-Identifier: Apache-2.0
//! The wire form of a bucketed write (`WritePlan.bucket`), checked on decode.

use krishiv_exec::physical::BucketWrite;

use crate::decode::{usize_of, usizes};
use crate::error::Result;
use crate::proto as pb;

/// A bucketed write's bucket column and sort columns from their wire form (none: not bucketed).
pub(crate) fn bucket_of(b: Option<&pb::WriteBucket>) -> Result<Option<BucketWrite>> {
    b.map(|b| {
        Ok(BucketWrite {
            column: usize_of(b.column)?,
            sort_by: usizes(&b.sort_by)?,
        })
    })
    .transpose()
}
