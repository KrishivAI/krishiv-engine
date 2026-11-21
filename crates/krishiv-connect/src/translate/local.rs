// SPDX-License-Identifier: Apache-2.0
//! `LocalRelation` and `CachedLocalRelation` (`createDataFrame`): the host's local relations.
//!
//! The translator is synchronous and building a relation is not (decoding the Arrow IPC data and
//! applying a user schema run on the session's executor), so the host runs an async pre-pass
//! (design P3-T012a part 2b, re-review nit N4): it translates the plan once with
//! [`LocalRelations::collecting`], which records a [`LocalRequest`] per local relation and
//! answers with placeholders; builds the relations (`Session::local_relation`); and translates
//! again with [`LocalRelations::resolved`], which hands the translator the n-th relation for the
//! n-th request. A plan without local relations needs the first translation only. Translation
//! visits the relations in a fixed order, so the n-th request of both passes is the same relation.

use std::sync::{Arc, Mutex, PoisonError};

use krishiv_common::error::{
    ConnectInvalidPlanSchemaRequiredForLocalRelation, KrishivError, KrishivUnsupportedFeature,
};
use krishiv_plan::LocalRelation;
use krishiv_types::StructType;

use super::schema::parse_local_schema;
use crate::proto::v4_3 as pb;

/// A local relation the host builds before the second translation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalRequest {
    /// `LocalRelation`: Arrow IPC data with its own schema and a user schema, either of which may
    /// be absent (not both).
    Inline {
        /// The Arrow IPC stream.
        data: Option<Vec<u8>>,
        /// The parsed `schema` (DDL or JSON; a type that is not a struct is `struct<value: t>`).
        schema: Option<StructType>,
    },
    /// `CachedLocalRelation`: the artifact `cache/<hash>` holding a serialized `LocalRelation`.
    Cached {
        /// The hash the client names.
        hash: String,
    },
    /// `Relation.catalog`: a call of `spark.catalog` the host runs, answering its result rows.
    Catalog(super::catalog::CatalogCall),
}

/// Where the translator gets its local relations from.
#[derive(Clone, Default)]
pub struct LocalRelations(Mode);

#[derive(Clone, Default)]
enum Mode {
    /// No host: a local relation is refused.
    #[default]
    None,
    /// The first pass: record the requests, answer with placeholders.
    Collect(Arc<Mutex<Vec<LocalRequest>>>),
    /// The second pass: the relation of the n-th request.
    Resolved(Arc<[LocalRelation]>),
}

impl LocalRelations {
    /// The first pass: records what the plan needs.
    #[must_use]
    pub fn collecting() -> Self {
        Self(Mode::Collect(Arc::default()))
    }

    /// The second pass: `relations[n]` answers the n-th request of the first pass.
    #[must_use]
    pub fn resolved(relations: Vec<LocalRelation>) -> Self {
        Self(Mode::Resolved(relations.into()))
    }

    /// Takes the requests recorded so far, in order (empty unless [`Self::collecting`]), leaving
    /// none behind: the inline data is moved out, never copied.
    #[must_use]
    pub fn take_requests(&self) -> Vec<LocalRequest> {
        match &self.0 {
            Mode::Collect(r) => {
                std::mem::take(&mut *r.lock().unwrap_or_else(PoisonError::into_inner))
            }
            _ => Vec::new(),
        }
    }

    /// The relation for the request `ordinal`: recorded (and a placeholder returned) in the first
    /// pass, looked up in the second.
    pub(super) fn get(
        &self,
        ordinal: usize,
        request: impl FnOnce() -> Result<LocalRequest, KrishivError>,
    ) -> Result<LocalRelation, KrishivError> {
        match &self.0 {
            Mode::None => Err(KrishivUnsupportedFeature {
                feature: "local relations without a host that builds them \
                          (TranslateConf::local_relations)"
                    .to_owned(),
            }
            .into()),
            Mode::Collect(requests) => {
                let request = request()?;
                requests
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(request);
                // Never run: the first pass's plan is dropped when there were requests.
                Ok(LocalRelation::from_decoded(
                    Arc::new(arrow_schema::Schema::empty()),
                    [0; 32],
                    0,
                    0,
                    0,
                ))
            }
            Mode::Resolved(relations) => relations.get(ordinal).cloned().ok_or_else(|| {
                krishiv_common::error::InternalError {
                    message: format!("no local relation was built for request {ordinal}"),
                }
                .into()
            }),
        }
    }
}

impl std::fmt::Debug for LocalRelations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Mode::None => f.write_str("LocalRelations::None"),
            Mode::Collect(_) => f.write_str("LocalRelations::Collect"),
            Mode::Resolved(r) => write!(f, "LocalRelations::Resolved({})", r.len()),
        }
    }
}

impl PartialEq for LocalRelations {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Mode::None, Mode::None) => true,
            (Mode::Collect(a), Mode::Collect(b)) => Arc::ptr_eq(a, b),
            (Mode::Resolved(a), Mode::Resolved(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for LocalRelations {}

/// The request of a `LocalRelation`, as Spark's `transformLocalRelation` reads it: a schema,
/// when set, is parsed (DDL, else JSON); with neither data nor schema the relation is invalid.
///
/// # Errors
/// `CONNECT_INVALID_PLAN.SCHEMA_REQUIRED_FOR_LOCAL_RELATION` for neither, the parse error of a
/// schema that is neither DDL nor JSON.
pub(super) fn inline_request(l: &pb::LocalRelation) -> Result<LocalRequest, KrishivError> {
    let schema = l.schema.as_deref().map(parse_local_schema).transpose()?;
    if l.data.is_none() && schema.is_none() {
        return Err(ConnectInvalidPlanSchemaRequiredForLocalRelation {}.into());
    }
    Ok(LocalRequest::Inline {
        data: l.data.clone(),
        schema,
    })
}

#[cfg(test)]
mod tests;
