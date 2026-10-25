// SPDX-License-Identifier: Apache-2.0
//! The `text` schema: always one nullable `value` string column, as in
//! Spark. There is nothing to infer.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::future::BoxFuture;

use crate::error::FormatResult;
use crate::io::{FileRef, IoContext};
use crate::options::FormatOptions;

/// The schema of `text` `files`.
pub(crate) fn infer<'a>(
    _io: &'a IoContext,
    files: &'a [FileRef],
    _o: &'a FormatOptions,
) -> BoxFuture<'a, FormatResult<SchemaRef>> {
    let mut result: FormatResult<SchemaRef> = Ok(Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Utf8View,
        true,
    )])));
    for f in files {
        if let Err(e) = crate::text::check_plain(f.path.as_ref()) {
            result = Err(e.into());
            break;
        }
    }
    Box::pin(std::future::ready(result))
}
