// SPDX-License-Identifier: Apache-2.0
//! Cloud object stores: S3, GCS, Azure and the read-only web stores.
//!
//! URL parsing, the `fs.*` keys, credential lookup and store building live in the
//! `krishiv-store` crate (ADR 0044), which has no scheduler, cache or memory accounting in its
//! API; this module re-exports its items at their old paths and holds the one piece that needs the
//! [`IoScheduler`]: [`register`] puts a store the registry built on the scheduler, so every
//! request still goes through the layer's retry, timeout, pacing and accounting wrappers.
//! The module docs of `krishiv_store::StoreRegistry` describe the URLs, keys and secret handling.

use std::collections::BTreeMap;

use krishiv_common::secret::OptionValue;
pub(crate) use krishiv_store::is_store_option;
pub use krishiv_store::{
    AZURE_ACCOUNT_KEY, AZURE_ACCOUNT_NAME, AZURE_ENDPOINT, AZURE_SAS, GCS_ENDPOINT, GCS_KEY,
    GCS_KEYFILE, GCS_STORAGE_ROOT_URL, HF_ENDPOINT, HF_TOKEN, ObjectScheme, OpenedStore, ParsedUrl,
    S3_ACCESS_KEY, S3_ENDPOINT, S3_ENDPOINT_REGION, S3_PATH_STYLE_ACCESS, S3_REGION, S3_SECRET_KEY,
    S3_SESSION_TOKEN, StoreRegistry, StoreSettings, parse_object_url,
};
use object_store::path::Path;

use super::{IoScheduler, StoreId};
use crate::error::Result;

/// Parses `text`, builds its store with `registry` and registers it on `sched` under its store
/// id, returning the id, the key and the display path.
///
/// # Errors
/// As [`parse_object_url`] and [`StoreRegistry::store`].
pub fn register(
    sched: &IoScheduler,
    registry: &StoreRegistry,
    text: &str,
) -> Result<(StoreId, Path, String)> {
    register_with_options(sched, registry, text, &BTreeMap::new())
}

/// As [`register`], with per-read `fs.*` data-source options overriding the registry's settings
/// (see [`StoreRegistry::store_with_options`]).
///
/// # Errors
/// As [`parse_object_url`] and [`StoreRegistry::store_with_options`].
pub fn register_with_options(
    sched: &IoScheduler,
    registry: &StoreRegistry,
    text: &str,
    options: &BTreeMap<String, OptionValue>,
) -> Result<(StoreId, Path, String)> {
    let opened = registry.open_with_options(text, options)?;
    sched.register_store_scoped(
        opened.id.clone(),
        opened.store,
        opened.policy,
        &opened.scope,
    );
    Ok((opened.id, opened.key, opened.display))
}
