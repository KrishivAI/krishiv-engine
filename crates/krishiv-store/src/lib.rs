// SPDX-License-Identifier: Apache-2.0
//! The object-store layer shared by the Krishiv engine and platform (ADR 0044).
//!
//! It holds URL parsing (`s3://`, `gs://`, `abfss://`, `https://`, `hf://`), store building from
//! Spark-style `fs.*` settings with typed secrets, and the pacing budgets per scheme. Request
//! scheduling, caches and memory accounting stay in `krishiv-io`. Settings come through
//! [`StoreSettings`], so an outside user needs no engine type:
//!
//! ```
//! use std::collections::BTreeMap;
//! use std::sync::Arc;
//! use krishiv_store::{StoreRegistry, parse_object_url};
//!
//! // `s3a://bucket/key` names the store `s3://bucket` and the key `key`.
//! let url = parse_object_url("s3a://bucket/warehouse/t/part-0.parquet")?;
//! assert_eq!(url.store_id.to_string(), "s3://bucket");
//! assert_eq!(url.key.as_ref(), "warehouse/t/part-0.parquet");
//!
//! // Settings are a plain map of Hadoop-style `fs.*` keys.
//! let mut settings = BTreeMap::new();
//! settings.insert("fs.s3a.endpoint".to_owned(), "http://127.0.0.1:9000".to_owned());
//! let registry = StoreRegistry::new(Arc::new(settings));
//!
//! // Without the `cloud` feature no cloud store is built, and the error says so.
//! if cfg!(not(feature = "cloud")) {
//!     assert!(registry.store(&url).is_err());
//! }
//! # Ok::<(), krishiv_common::error::KrishivError>(())
//! ```
//!
//! # Features
//! - `cloud`: the S3, GCS, Azure and web builders (`object_store`'s `aws`, `azure`, `gcp` and
//!   `http` features). Without it URLs still parse, so plans can qualify paths, but building a
//!   cloud store fails with `KRISHIV_UNSUPPORTED_FEATURE`.
#![forbid(unsafe_code)]

mod credentials;
mod error;
mod id;
mod policy;
mod refreshing;
mod scope;
mod settings;
mod stores;

pub use credentials::{
    CredentialScope, CredentialSource, DEFAULT_REFRESH_SKEW, RefreshingProvider, ScopedCredential,
};
pub use error::Result;
pub use id::StoreId;
pub use policy::StorePolicy;
pub use refreshing::RefreshingStore;
pub use scope::CacheScope;
pub use settings::{SettingValue, StoreSettings};
pub use stores::{
    AZURE_ACCOUNT_KEY, AZURE_ACCOUNT_NAME, AZURE_ENDPOINT, AZURE_SAS, GCS_ENDPOINT, GCS_KEY,
    GCS_KEYFILE, GCS_STORAGE_ROOT_URL, HF_ENDPOINT, HF_TOKEN, ObjectScheme, OpenedStore, ParsedUrl,
    S3_ACCESS_KEY, S3_ENDPOINT, S3_ENDPOINT_REGION, S3_PATH_STYLE_ACCESS, S3_REGION, S3_SECRET_KEY,
    S3_SESSION_TOKEN, StoreRegistry, is_store_option, parse_object_url,
};
