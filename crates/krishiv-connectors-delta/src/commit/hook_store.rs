// SPDX-License-Identifier: Apache-2.0
//! A store for the commit tests that runs a hook at the moments the commit loop must survive:
//! after a put landed but before its answer (the answer is then lost), and after a `head`
//! found no object (so another writer's commit lands right after a winner read).

use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex, PoisonError};

use async_trait::async_trait;
use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as StoreResult,
};

/// A hook: the path suffix it fires on, and what it runs then.
type Hook = (String, Box<dyn FnOnce() + Send>);

/// See the module docs.
pub(super) struct HookStore {
    inner: Arc<dyn ObjectStore>,
    after_put: Mutex<Option<(Hook, bool)>>,
    after_missing_head: Mutex<Option<Hook>>,
}

impl fmt::Debug for HookStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HookStore({})", self.inner)
    }
}

impl fmt::Display for HookStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HookStore({})", self.inner)
    }
}

fn lost() -> object_store::Error {
    object_store::Error::Generic {
        store: "HookStore",
        source: "the connection was reset".into(),
    }
}

impl HookStore {
    pub(super) fn new(inner: Arc<dyn ObjectStore>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            after_put: Mutex::new(None),
            after_missing_head: Mutex::new(None),
        })
    }

    /// The next put of a path ending in `suffix` runs `hook` and fails; it is written first
    /// when `write` (a lost answer), not at all otherwise.
    pub(super) fn on_put(&self, suffix: &str, write: bool, hook: impl FnOnce() + Send + 'static) {
        *self
            .after_put
            .lock()
            .unwrap_or_else(PoisonError::into_inner) =
            Some(((suffix.to_owned(), Box::new(hook)), write));
    }

    /// The next `head` of a path ending in `suffix` that finds nothing runs `hook` after.
    pub(super) fn on_missing_head(&self, suffix: &str, hook: impl FnOnce() + Send + 'static) {
        *self
            .after_missing_head
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some((suffix.to_owned(), Box::new(hook)));
    }
}

#[async_trait]
impl ObjectStore for HookStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> StoreResult<PutResult> {
        let hook = {
            let mut slot = self
                .after_put
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            match slot.as_ref() {
                Some(((suffix, _), _)) if location.as_ref().ends_with(suffix.as_str()) => {
                    slot.take()
                }
                _ => None,
            }
        };
        let Some(((_, run), write)) = hook else {
            return self.inner.put_opts(location, payload, opts).await;
        };
        if write {
            self.inner.put_opts(location, payload, opts).await?;
        }
        run();
        Err(lost())
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> StoreResult<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> StoreResult<GetResult> {
        let head = options.head;
        let result = self.inner.get_opts(location, options).await;
        if head && matches!(result, Err(object_store::Error::NotFound { .. })) {
            let hook = {
                let mut slot = self
                    .after_missing_head
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                match slot.as_ref() {
                    Some((suffix, _)) if location.as_ref().ends_with(suffix.as_str()) => {
                        slot.take()
                    }
                    _ => None,
                }
            };
            if let Some((_, run)) = hook {
                run();
            }
        }
        result
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> StoreResult<Vec<bytes::Bytes>> {
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, StoreResult<Path>>,
    ) -> BoxStream<'static, StoreResult<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, StoreResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> StoreResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> StoreResult<()> {
        self.inner.copy_opts(from, to, options).await
    }
}
