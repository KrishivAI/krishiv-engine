// SPDX-License-Identifier: Apache-2.0
//! [`LocalRelation`]: Arrow data a plan reads (Spark's `LocalRelation`).
//!
//! The batches live once, in the session's [`krishiv_memory::RelationStore`];
//! the plan holds the content hash, the counts and (when built in this
//! process) a lease that keeps the entry alive while the plan does. A plan
//! decoded from the plan codec holds no lease: the provider resolves it by
//! hash in the same process's store or fails loudly.

use std::fmt;

use arrow_schema::SchemaRef;
use krishiv_memory::{RelationHash, RelationLease, RelationRef};

/// Arrow data a plan reads. See the module docs.
#[derive(Clone)]
pub struct LocalRelation {
    schema: SchemaRef,
    relation: RelationRef,
    batches: usize,
    rows: u64,
    bytes: u64,
    lease: Option<RelationLease>,
}

impl LocalRelation {
    /// A relation over the entry `lease` holds. `schema` is the Arrow
    /// schema of the stored batches; the hash is the lease's, so it is
    /// correct by construction.
    #[must_use]
    pub fn from_lease(schema: SchemaRef, lease: &RelationLease, rows: u64, bytes: u64) -> Self {
        Self {
            schema,
            relation: lease.reference(),
            batches: lease.batches(),
            rows,
            bytes,
            lease: Some(lease.clone()),
        }
    }

    /// A relation named by its 32-byte content hash, holding no lease (the
    /// plan codec's decode: the entry is found by hash where the plan runs).
    #[must_use]
    pub fn from_decoded(
        schema: SchemaRef,
        hash: [u8; 32],
        batches: usize,
        rows: u64,
        bytes: u64,
    ) -> Self {
        Self {
            schema,
            relation: RelationRef {
                hash: RelationHash::from_bytes(hash),
            },
            batches,
            rows,
            bytes,
            lease: None,
        }
    }

    /// The Arrow schema of the stored batches.
    #[must_use]
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// The content hash the data is found by.
    #[must_use]
    pub fn relation(&self) -> RelationRef {
        self.relation
    }

    /// The number of stored batches.
    #[must_use]
    pub fn batches(&self) -> usize {
        self.batches
    }

    /// The number of rows.
    #[must_use]
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// The decoded size in bytes at insertion.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The lease keeping the entry alive; `None` after a codec decode.
    #[must_use]
    pub fn lease(&self) -> Option<&RelationLease> {
        self.lease.as_ref()
    }
}

impl PartialEq for LocalRelation {
    fn eq(&self, other: &Self) -> bool {
        self.relation == other.relation
            && self.schema == other.schema
            && self.batches == other.batches
            && self.rows == other.rows
    }
}

impl Eq for LocalRelation {}

impl fmt::Debug for LocalRelation {
    /// The shape, counts and a hash prefix; never values.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hex = self.relation.hash.hex();
        f.debug_struct("LocalRelation")
            .field("columns", &self.schema.fields().len())
            .field("batches", &self.batches)
            .field("rows", &self.rows)
            .field("hash", &hex.get(..12).unwrap_or(&hex))
            .finish_non_exhaustive()
    }
}

/// The Spark type of `field` as text for the canonical encodings; the Arrow
/// type's name when it has no Spark type (such a relation does not resolve).
pub(crate) fn type_text(field: &arrow_schema::Field) -> String {
    krishiv_types::DataType::from_arrow_field(field).map_or_else(
        |_| format!("arrow:{:?}", field.data_type()),
        |t| t.simple_string(),
    )
}
