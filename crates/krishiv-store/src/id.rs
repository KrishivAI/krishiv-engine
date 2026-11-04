// SPDX-License-Identifier: Apache-2.0
//! The identity of a registered store.

use std::fmt;
use std::sync::Arc;

use krishiv_common::secret::OptionValue;

/// Identifies a registered store, e.g. `s3://bucket`. Its text is printed in
/// paths, so it must not hold credentials (they are redacted if it does).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct StoreId(Arc<str>);

impl StoreId {
    /// A store id.
    #[must_use]
    pub fn new(id: impl AsRef<str>) -> Self {
        Self(Arc::from(id.as_ref()))
    }

    /// Whether the store is an object store or HTTP (by the id's URL
    /// scheme), where a request costs latency: local files are `file`.
    #[must_use]
    pub fn is_remote(&self) -> bool {
        const REMOTE: [&str; 10] = [
            "s3", "s3a", "gs", "gcs", "az", "azure", "abfs", "abfss", "http", "https",
        ];
        self.0
            .split_once("://")
            .is_some_and(|(scheme, _)| REMOTE.iter().any(|r| scheme.eq_ignore_ascii_case(r)))
    }
}

/// Separates a store's name from the alias of one write's private
/// registration of it (see [`StoreId::private_to`]).
const PRIVATE_MARK: &str = "#private-";

impl StoreId {
    /// The id under which one write (`job`) registers its own store for the
    /// same location as `self`: a distinct registry key whose text, in every
    /// message and in `EXPLAIN`, is still `self`'s.
    #[must_use]
    pub fn private_to(&self, job: &str) -> Self {
        Self::new(format!("{}{PRIVATE_MARK}{job}", self.0))
    }

    /// The full registry key: [`Display`](fmt::Display) hides the alias of
    /// [`Self::private_to`] and redacts credentials.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for StoreId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shown = self
            .0
            .split_once(PRIVATE_MARK)
            .map_or(&*self.0, |(name, _)| name);
        // Typed redaction (ADR 0038): URL credentials never print.
        OptionValue::from_user("path", shown).fmt(f)
    }
}

impl fmt::Debug for StoreId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StoreId({self})")
    }
}
