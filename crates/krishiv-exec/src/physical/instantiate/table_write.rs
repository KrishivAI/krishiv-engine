// SPDX-License-Identifier: Apache-2.0
//! The instantiation of a [`PhysNode::TableWrite`](crate::physical::PhysNode::TableWrite): the
//! input's columns checked by storage against the request, the write id required, and the
//! table's writer or row-level rewriter resolved on the worker.

use std::sync::Arc;

use super::{Built, Instantiation};
use crate::error::{Result, internal};
use crate::physical::PhysNode;

impl Instantiation<'_> {
    /// A [`PhysNode::TableWrite`]: the input built and its columns checked against the request's
    /// schema (by storage: see [`same_storage`]), the write id required (a retried task must
    /// commit once), then the table resolved through the worker's resolver and its sink asked
    /// for the request.
    pub(super) fn table_write(
        &mut self,
        input: &PhysNode,
        table: &krishiv_plan::Scan,
        request: &crate::physical::TableWriteRequest,
    ) -> Result<Built> {
        use crate::physical::TableWriteRequest;
        let input = self.node(input)?;
        let want = request.input_schema();
        let same = input.schema.fields().len() == want.fields().len()
            && input
                .schema
                .fields()
                .iter()
                .zip(want.fields())
                .all(|(a, b)| same_storage(a.data_type(), b.data_type()));
        if !same {
            return Err(internal(format!(
                "a write into {} gets columns {:?} where its request says {:?}",
                request.table_name(),
                input.schema.fields(),
                want.fields()
            )));
        }
        let write_id = match request {
            TableWriteRequest::Write(w) => &w.write_id,
            TableWriteRequest::RowLevel(r) => &r.write_id,
        };
        if write_id.is_none() {
            return Err(internal(format!(
                "a write into {} without a write id",
                request.table_name()
            )));
        }
        let provider = self.worker.tables.provider(table)?;
        let sink = match request {
            TableWriteRequest::Write(w) => provider.writer(w)?,
            TableWriteRequest::RowLevel(r) => provider
                .row_level(r.op)
                .ok_or_else(|| {
                    krishiv_common::error::KrishivError::from(
                        krishiv_common::error::UnsupportedFeatureTableOperation {
                            table_name: r.table_name.clone(),
                            operation: r.op.name().to_owned(),
                        },
                    )
                })?
                .rewriter(r)?,
        };
        Ok(Built {
            schema: crate::operators::table_write::written_schema(),
            node: Arc::new(crate::operators::table_write::TableWrite {
                input: input.node,
                sink,
            }),
        })
    }
}

/// Whether a column of type `a` holds what a column of type `b` holds: the same type up to the
/// engine's interchangeable carriers (string and binary layouts, decimal widths of one precision
/// and scale, list layouts), and ignoring nested field names and nullability.
fn same_storage(a: &arrow_schema::DataType, b: &arrow_schema::DataType) -> bool {
    use arrow_schema::DataType as T;
    let strings = |t: &T| matches!(t, T::Utf8 | T::LargeUtf8 | T::Utf8View);
    let binaries = |t: &T| matches!(t, T::Binary | T::LargeBinary | T::BinaryView);
    let decimal = |t: &T| match t {
        T::Decimal32(p, s) | T::Decimal64(p, s) | T::Decimal128(p, s) | T::Decimal256(p, s) => {
            Some((*p, *s))
        }
        _ => None,
    };
    let element = |t: &T| match t {
        T::List(f) | T::LargeList(f) | T::ListView(f) | T::LargeListView(f) => {
            Some(f.data_type().clone())
        }
        _ => None,
    };
    if strings(a) || strings(b) {
        return strings(a) && strings(b);
    }
    if binaries(a) || binaries(b) {
        return binaries(a) && binaries(b);
    }
    if let (Some(x), Some(y)) = (decimal(a), decimal(b)) {
        return x == y;
    }
    if let (Some(x), Some(y)) = (element(a), element(b)) {
        return same_storage(&x, &y);
    }
    match (a, b) {
        (T::Struct(x), T::Struct(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|(f, g)| same_storage(f.data_type(), g.data_type()))
        }
        (T::Map(x, _), T::Map(y, _)) => same_storage(x.data_type(), y.data_type()),
        (T::FixedSizeList(x, n), T::FixedSizeList(y, m)) => {
            n == m && same_storage(x.data_type(), y.data_type())
        }
        _ => a == b,
    }
}
