// SPDX-License-Identifier: Apache-2.0
//! Krishiv's implementation of the kernel's `Engine` (design note P3-T021 section 3).
//!
//! [`DeltaEngine`] holds what every handler needs: the query's `IoContext`, the table's store
//! and root, the runtime handle the handlers bridge to, the parse limits and the query's
//! cancellation. Kernel calls are entered only through [`DeltaEngine::delta_blocking`], which
//! runs them on the process's [`LogPool`]. Handlers: storage ([`storage`]), JSON ([`json`]),
//! Parquet ([`columnar`]) and evaluation ([`eval`]).

pub mod columnar;
pub mod data;
pub mod eval;
pub mod json;
pub mod paths;
pub mod pool;
pub(crate) mod relabel;
pub mod schema;
pub mod storage;

use std::future::Future;
use std::sync::Arc;

use delta_kernel::cancellation::{
    CancellationToken as KernelToken, CancellationTokenRef, CancelledFuture,
};
use delta_kernel::{KernelError, KernelResult};
use krishiv_common::config::{
    CATALOG_MAX_COLUMNS, CATALOG_MAX_METADATA_BYTES, Config, ConfigValue,
};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::CancellationToken;
use krishiv_io::{IoContext, StoreId};
use tokio::runtime::{Handle, RuntimeFlavor};
use url::Url;

use self::paths::TableRoot;
use self::pool::{LogPool, on_log_pool};
use crate::error::to_kernel;

pub use self::columnar::KrishivParquetHandler;
pub use self::data::KrishivEngineData;
pub use self::eval::KrishivEvaluationHandler;
pub use self::json::KrishivJsonHandler;
pub use self::storage::KrishivStorageHandler;

/// The bounds on what the handlers parse (lakehouse design section 5.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaLimits {
    /// Largest commit file, `_last_checkpoint` or `.crc` file, or JSON string parsed, in bytes
    /// (`krishiv.catalog.maxMetadataBytes`).
    pub max_metadata_bytes: u64,
    /// Deepest JSON nesting of anything parsed.
    pub max_depth: usize,
    /// Most columns, nested fields included, of a `metaData` schema string
    /// (`krishiv.catalog.maxColumns`).
    pub max_columns: usize,
}

/// The fixed JSON nesting limit.
pub const MAX_JSON_DEPTH: usize = 64;

impl Default for DeltaLimits {
    fn default() -> Self {
        Self {
            max_metadata_bytes: 256 << 20,
            max_depth: MAX_JSON_DEPTH,
            max_columns: 10_000,
        }
    }
}

impl DeltaLimits {
    /// The limits `config` sets.
    #[must_use]
    pub fn from_config(config: &Config) -> Self {
        let d = Self::default();
        Self {
            max_metadata_bytes: config
                .get(&CATALOG_MAX_METADATA_BYTES)
                .and_then(ConfigValue::as_size)
                .unwrap_or(d.max_metadata_bytes),
            max_depth: MAX_JSON_DEPTH,
            max_columns: config
                .get(&CATALOG_MAX_COLUMNS)
                .and_then(ConfigValue::as_int)
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(d.max_columns),
        }
    }
}

/// What every handler of one table, for one query, shares.
#[derive(Debug)]
pub(crate) struct Shared {
    pub(crate) io: IoContext,
    pub(crate) store: StoreId,
    pub(crate) root: TableRoot,
    pub(crate) handle: Handle,
    pub(crate) limits: DeltaLimits,
    pub(crate) cancel: Arc<CancellationToken>,
}

impl Shared {
    /// Fails with the kernel's `Cancelled` once the query is cancelled: every handler call
    /// starts here.
    pub(crate) fn check_cancelled(&self) -> KernelResult<()> {
        if self.cancel.is_cancelled() {
            Err(KernelError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Runs the I/O `fut` from a log-pool thread, racing the query's cancellation: a pure read,
    /// which may be dropped mid-flight. Refused anywhere but on the log pool: a runtime worker
    /// must never block.
    pub(crate) fn block_on<T>(
        &self,
        fut: impl Future<Output = Result<T, KrishivError>>,
    ) -> KernelResult<T> {
        self.enter_io()?;
        let cancel = &self.cancel;
        self.handle.block_on(async {
            tokio::select! {
                biased;
                () = cancel.cancelled() => Err(KernelError::Cancelled),
                r = fut => r.map_err(to_kernel),
            }
        })
    }

    /// Runs the write `fut` to completion from a log-pool thread. Cancellation is checked before
    /// it starts, never during: a put dropped mid-flight may still land (a request already sent,
    /// a local write on a detached blocking thread), and a commit reported as cancelled after it
    /// landed would mislead the kernel (design note section 3.2: everything after the commit put
    /// has succeeded is exempt from cancellation).
    pub(crate) fn block_on_to_completion<T>(
        &self,
        fut: impl Future<Output = Result<T, KrishivError>>,
    ) -> KernelResult<T> {
        self.enter_io()?;
        self.handle.block_on(fut).map_err(to_kernel)
    }

    /// The checks before any handler I/O: not cancelled, and on the log pool.
    fn enter_io(&self) -> KernelResult<()> {
        self.check_cancelled()?;
        if !on_log_pool() {
            debug_assert!(false, "a Delta handler ran outside the log pool");
            return Err(to_kernel(
                InternalError {
                    message: "a Delta handler ran outside the log pool".to_owned(),
                }
                .into(),
            ));
        }
        Ok(())
    }
}

/// The Delta engine of one table for one query.
#[derive(Debug, Clone)]
pub struct DeltaEngine {
    shared: Arc<Shared>,
    pool: &'static LogPool,
}

impl DeltaEngine {
    /// An engine for the table at `root` in `store`, reading through `io`, bridging to `handle`,
    /// running kernel calls on `pool` and ending them when `cancel` fires.
    ///
    /// `handle` must belong to a multi-thread runtime: the handlers `block_on` it from pool
    /// threads, and a current-thread runtime drives its I/O only inside its own `block_on`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a current-thread `handle`.
    pub fn new(
        io: IoContext,
        store: StoreId,
        root: Url,
        handle: Handle,
        limits: DeltaLimits,
        cancel: Arc<CancellationToken>,
        pool: &'static LogPool,
    ) -> Result<Self, KrishivError> {
        if handle.runtime_flavor() == RuntimeFlavor::CurrentThread {
            return Err(InternalError {
                message: "the Delta engine needs a multi-thread runtime".to_owned(),
            }
            .into());
        }
        Ok(Self {
            shared: Arc::new(Shared {
                io,
                store,
                root: TableRoot::new(root),
                handle,
                limits,
                cancel,
            }),
            pool,
        })
    }

    /// The table root (ends with `/`).
    #[must_use]
    pub fn table_root(&self) -> &Url {
        self.shared.root.url()
    }

    /// The table root, confining every path read under it.
    pub(crate) fn root(&self) -> &TableRoot {
        &self.shared.root
    }

    /// The query's I/O.
    #[must_use]
    pub fn io(&self) -> &IoContext {
        &self.shared.io
    }

    /// The store holding the table.
    #[must_use]
    pub fn store(&self) -> &StoreId {
        &self.shared.store
    }

    /// The storage handler.
    #[must_use]
    pub fn storage(&self) -> Arc<KrishivStorageHandler> {
        Arc::new(KrishivStorageHandler::new(Arc::clone(&self.shared)))
    }

    /// The JSON handler.
    #[must_use]
    pub fn json(&self) -> Arc<KrishivJsonHandler> {
        Arc::new(KrishivJsonHandler::new(Arc::clone(&self.shared)))
    }

    /// The Parquet handler.
    #[must_use]
    pub fn parquet(&self) -> Arc<KrishivParquetHandler> {
        Arc::new(KrishivParquetHandler::new(Arc::clone(&self.shared)))
    }

    /// The evaluation handler.
    #[must_use]
    pub fn evaluation(&self) -> Arc<KrishivEvaluationHandler> {
        Arc::new(KrishivEvaluationHandler::new(self.shared.limits))
    }

    /// The query's cancellation as the kernel's token, for `with_cancellation_token`.
    #[must_use]
    pub fn kernel_cancellation(&self) -> CancellationTokenRef {
        Arc::new(KernelCancel(Arc::clone(&self.shared.cancel)))
    }

    /// Runs the kernel call `f` with this engine on the log pool (see [`LogPool::run`]). The
    /// only way kernel calls are entered; planning code awaits it.
    ///
    /// Every lazy kernel iterator (scan metadata, log replay, a handler's `read_files`) must be
    /// drained inside `f`: pulled later on a runtime worker, its handler I/O is refused as off the
    /// pool (`INTERNAL_ERROR`; a debug assertion in debug builds).
    ///
    /// A call that has started returns its own outcome, a commit that landed included, even when
    /// the query is cancelled meanwhile.
    ///
    /// # Errors
    /// `OPERATION_CANCELED` when the query is cancelled before the call starts; `INTERNAL_ERROR`
    /// if `f` panics.
    pub async fn delta_blocking<T, F>(&self, f: F) -> Result<T, KrishivError>
    where
        T: Send + 'static,
        F: FnOnce(&DeltaEngine) -> T + Send + 'static,
    {
        let engine = self.clone();
        self.pool.run(&self.shared.cancel, move || f(&engine)).await
    }
}

/// The assembled engine: the kernel's four handlers over Krishiv's I/O and Arrow compute.
/// Every kernel call that takes it still goes through [`DeltaEngine::delta_blocking`].
impl delta_kernel::Engine for DeltaEngine {
    fn evaluation_handler(&self) -> Arc<dyn delta_kernel::EvaluationHandler> {
        self.evaluation()
    }

    fn storage_handler(&self) -> Arc<dyn delta_kernel::StorageHandler> {
        self.storage()
    }

    fn json_handler(&self) -> Arc<dyn delta_kernel::JsonHandler> {
        self.json()
    }

    fn parquet_handler(&self) -> Arc<dyn delta_kernel::ParquetHandler> {
        self.parquet()
    }
}

/// The query's cancellation, as the kernel sees it.
struct KernelCancel(Arc<CancellationToken>);

impl KernelToken for KernelCancel {
    fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }

    fn cancelled_future(&self) -> CancelledFuture<'_> {
        Box::pin(self.0.cancelled())
    }
}

#[cfg(test)]
mod tests;
