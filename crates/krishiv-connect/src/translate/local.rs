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
    ConnectInvalidPlanCannotParse, ConnectInvalidPlanSchemaRequiredForLocalRelation, KrishivError,
    KrishivUnsupportedFeature,
};
use krishiv_plan::{Literal, LocalRelation};
use krishiv_types::StructType;

use super::schema::parse_local_schema;
use crate::proto::v4_3 as pb;

/// A local relation the host builds before the second translation.
#[derive(Debug, Clone, PartialEq)]
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
    /// A pivot without values: the host runs Spark's `collectPivotValues` over the pivot's input
    /// (`select(col).distinct().limit(max + 1).sort(col)`) and answers the values.
    PivotValues {
        /// The pivot's input relation, as the client sent it (it may hold local relations or UDFs
        /// of its own: the host runs it through the whole pass).
        input: crate::Deep<pb::Relation>,
        /// The pivot column.
        column: crate::Deep<pb::Expression>,
    },
}

/// What the host built for one request.
#[derive(Debug, Clone, PartialEq)]
pub enum Built {
    /// The relation of an inline, cached or catalog request.
    Relation(LocalRelation),
    /// The values of a [`LocalRequest::PivotValues`], sorted.
    PivotValues(Vec<Literal>),
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
    Resolved(Arc<[Built]>),
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
        Self::resolved_with(relations.into_iter().map(Built::Relation).collect())
    }

    /// The second pass: `built[n]` answers the n-th request of the first pass.
    #[must_use]
    pub fn resolved_with(built: Vec<Built>) -> Self {
        Self(Mode::Resolved(built.into()))
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
            Mode::Resolved(built) => match built.get(ordinal) {
                Some(Built::Relation(r)) => Ok(r.clone()),
                _ => Err(no_answer(ordinal)),
            },
        }
    }

    /// The values of the pivot request `ordinal`: recorded (and one placeholder value returned)
    /// in the first pass, looked up in the second.
    pub(super) fn pivot_values(
        &self,
        ordinal: usize,
        request: impl FnOnce() -> Result<LocalRequest, KrishivError>,
    ) -> Result<Vec<Literal>, KrishivError> {
        match &self.0 {
            Mode::None => Err(KrishivUnsupportedFeature {
                feature: "a pivot without values, without a host that computes them \
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
                Ok(vec![Literal::Null(None)])
            }
            Mode::Resolved(built) => match built.get(ordinal) {
                Some(Built::PivotValues(v)) => Ok(v.clone()),
                _ => Err(no_answer(ordinal)),
            },
        }
    }
}

fn no_answer(ordinal: usize) -> KrishivError {
    krishiv_common::error::InternalError {
        message: format!("no local relation was built for request {ordinal}"),
    }
    .into()
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
pub(super) fn inline_request(
    l: &pb::LocalRelation,
    opts: &krishiv_sql::ParserOptions,
) -> Result<LocalRequest, KrishivError> {
    let (data, schema) = inline_parts(l.data.clone(), l.schema.as_deref(), opts)?;
    Ok(LocalRequest::Inline { data, schema })
}

/// The data and parsed schema of a `LocalRelation`, as [`inline_request`] checks them.
fn inline_parts(
    data: Option<Vec<u8>>,
    schema: Option<&str>,
    opts: &krishiv_sql::ParserOptions,
) -> Result<(Option<Vec<u8>>, Option<StructType>), KrishivError> {
    let schema = schema.map(|s| parse_local_schema(s, opts)).transpose()?;
    if data.is_none() && schema.is_none() {
        return Err(ConnectInvalidPlanSchemaRequiredForLocalRelation {}.into());
    }
    Ok((data, schema))
}

/// The inline relation (Arrow IPC data, parsed schema) a `CachedLocalRelation`'s `cache/<hash>`
/// artifact holds in `bytes`: a serialized `LocalRelation`, read as an inline one (Spark's `transformCachedLocalRelation`
/// parses the block and calls `transformLocalRelation`). The host takes the bytes only from the
/// requesting session's own artifacts, never from the relation store (design P3-T012a §1,
/// "Security: a client hash never reaches the relation store").
///
/// # Errors
/// `CONNECT_INVALID_PLAN.CANNOT_PARSE` for bytes that are not a `LocalRelation` (Spark raises an
/// unclassed protobuf parse error there), and the errors of an inline relation.
#[expect(
    clippy::disallowed_methods,
    reason = "a flat message (bytes and string fields), checked by `flat_message` first"
)]
pub fn cached_request(
    bytes: &[u8],
    opts: &krishiv_sql::ParserOptions,
) -> Result<(Option<Vec<u8>>, Option<StructType>), KrishivError> {
    use prost::Message as _;
    crate::wire::flat_message(bytes)?;
    let relation = pb::LocalRelation::decode(bytes).map_err(|e| {
        KrishivError::from(ConnectInvalidPlanCannotParse {
            error_msg: format!("the cached local relation does not decode: {e}"),
        })
    })?;
    // The decoded payload is moved, not copied again.
    inline_parts(relation.data, relation.schema.as_deref(), opts)
}

#[cfg(test)]
mod tests;
