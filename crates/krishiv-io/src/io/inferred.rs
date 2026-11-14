// SPDX-License-Identifier: Apache-2.0
//! The schemas formats inferred from files, kept next to the file-metadata
//! cache so a later query over the same file versions does not infer again.
//!
//! Inferring a CSV or JSON schema reads every row (`samplingRatio` 1), so a
//! DataFrame read with `inferSchema` would read its files twice per action.
//! A key names the format, its parsed options (session fallbacks included,
//! so a changed time zone or parse policy is another key) and every file by
//! its metadata-cache key (store, path, size and version), so a changed,
//! added or removed file is a miss, never a stale schema. The cache is
//! bounded by the file keys it holds: when full, it starts over (a miss only
//! costs the inference again).

use std::sync::{Mutex, PoisonError};

use arrow_schema::SchemaRef;

use crate::error::FormatResult;
use crate::format::FileFormat;
use crate::io::cache::CacheKey;
use crate::io::{FileRef, IoContext};
use crate::options::FormatOptions;

/// What an inference read: the format, its options and the files.
#[derive(Debug, Clone)]
pub struct InferKey {
    files: Vec<CacheKey>,
    options: FormatOptions,
}

impl InferKey {
    /// The key of inferring `files` as `format` with `options`.
    #[must_use]
    pub fn new(format: &'static str, files: &[FileRef], options: &FormatOptions) -> Self {
        Self {
            files: files
                .iter()
                .map(|f| CacheKey::new(format, f, None))
                .collect(),
            options: options.clone(),
        }
    }

    fn same(&self, other: &Self) -> bool {
        self.files == other.files && self.options.same_read(&other.options)
    }
}

/// The schemas inferred in a process (one cache per I/O scheduler).
#[derive(Debug)]
pub struct InferredSchemas {
    entries: Mutex<Entries>,
    max_files: usize,
}

#[derive(Debug, Default)]
struct Entries {
    held: Vec<(InferKey, SchemaRef)>,
    files: usize,
}

impl InferredSchemas {
    /// File keys held before the cache starts over.
    pub const MAX_FILES: usize = 16_384;

    /// An empty cache holding at most `max_files` file keys.
    #[must_use]
    pub fn new(max_files: usize) -> Self {
        Self {
            entries: Mutex::new(Entries::default()),
            max_files: max_files.max(1),
        }
    }

    /// The schema stored under `key`.
    #[must_use]
    pub fn get(&self, key: &InferKey) -> Option<SchemaRef> {
        let e = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        e.held
            .iter()
            .find(|(k, _)| k.same(key))
            .map(|(_, s)| SchemaRef::clone(s))
    }

    /// Stores `schema` under `key`. A key of more files than the cache holds
    /// is not stored; a full cache starts over first.
    pub fn insert(&self, key: InferKey, schema: SchemaRef) {
        let n = key.files.len();
        if n > self.max_files {
            return;
        }
        let mut e = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(at) = e.held.iter().position(|(k, _)| k.same(&key)) {
            e.held[at].1 = schema;
            return;
        }
        if e.files + n > self.max_files {
            e.held.clear();
            e.files = 0;
        }
        e.files += n;
        e.held.push((key, schema));
    }

    /// The number of schemas held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .held
            .len()
    }

    /// Whether no schema is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for InferredSchemas {
    fn default() -> Self {
        Self::new(Self::MAX_FILES)
    }
}

/// `format`'s schema of `files` with `options`, inferred once per file
/// versions: a later call with the same key takes the stored schema.
///
/// # Errors
/// The format's inference errors (which are not stored).
pub async fn infer_schema_cached(
    format: &dyn FileFormat,
    io: &IoContext,
    files: &[FileRef],
    options: &FormatOptions,
) -> FormatResult<SchemaRef> {
    let cache = io.scheduler().inferred_schemas();
    let key = InferKey::new(format.name(), files, options);
    if let Some(schema) = cache.get(&key) {
        return Ok(schema);
    }
    let schema = format.infer_schema(io, files, options).await?;
    cache.insert(key, SchemaRef::clone(&schema));
    Ok(schema)
}
