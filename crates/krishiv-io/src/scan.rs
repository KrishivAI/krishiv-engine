// SPDX-License-Identifier: Apache-2.0
//! [`ScanStream`]: the framework's read loop for one split of one file —
//! what the executor's file source runs per claimed split.
//!
//! Opening: metadata (cached) → [`SchemaMapper`] → file-level request →
//! [`FileFormat::plan_units`] → [`FileFormat::open`]. Reading: step the
//! decoder; fetch what it needs through the scheduler; prefetch what it
//! will need ([`Decoder::upcoming`]) within `krishiv.io.prefetchBytes`; map
//! each batch to the table schema.
//!
//! **Memory (ADR 0007).** The stream's [`Reservation`] always covers the
//! decoder's buffered bytes plus prefetched bytes. Bytes the decoder needs
//! to progress (one unit's column chunks, in dependency order) are reserved
//! with waiting ([`Reservation::grow`], which may arbitrate); prefetch only
//! with [`Reservation::try_grow`], and is the first thing dropped
//! ([`ScanStream::release_prefetch`], the scan's spill). Dropping the stream
//! returns everything.
//!
//! **Cancellation.** Every await is drop-safe: dropping a pending
//! [`ScanStream::next_batch`] cancels its requests; the decoder never saw a
//! partial push, so the next call asks again.

use std::ops::Range;

use arrow_array::RecordBatch;
use bytes::Bytes;
use futures::future::BoxFuture;
use krishiv_common::config::{Config, ConfigValue, IO_PREFETCH_BYTES};
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_memory::Reservation;

use crate::error::{Result, internal};
use crate::format::{Decoder, FileFormat, MetadataNeeds, ScanUnitMetrics, Step};
use crate::io::{ByteRange, FileRef, IoContext};
use crate::options::FormatOptions;
use crate::request::ScanRequest;
use crate::schema_map::SchemaMapper;

/// Per-scan settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanOptions {
    /// `krishiv.io.prefetchBytes`.
    pub prefetch_bytes: u64,
    /// `spark.sql.files.ignoreCorruptFiles` / the `ignoreCorruptFiles`
    /// option: a file that cannot be decoded (bad footer, undecodable data)
    /// ends early, keeping the rows already returned. Storage failures
    /// (`KRISHIV_OBJECT_STORE_REQUEST_FAILED`) and changed files are never
    /// skipped (a divergence from Spark, which skips any I/O error).
    pub ignore_corrupt_files: bool,
    /// `spark.sql.files.ignoreMissingFiles` / `ignoreMissingFiles`: a file
    /// deleted after listing reads as empty.
    pub ignore_missing_files: bool,
    /// Metadata beyond the footer (the page index when the predicate can use it).
    pub needs: MetadataNeeds,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self::from_config(&Config::new())
    }
}

impl ScanOptions {
    /// The settings in session config `c`.
    #[must_use]
    pub fn from_config(c: &Config) -> Self {
        let flag = |k: &str| {
            c.get_unregistered(k)
                .is_some_and(|v| v.trim().eq_ignore_ascii_case("true"))
        };
        Self {
            prefetch_bytes: c
                .get(&IO_PREFETCH_BYTES)
                .and_then(ConfigValue::as_size)
                .unwrap_or(64 << 20),
            ignore_corrupt_files: flag("spark.sql.files.ignoreCorruptFiles"),
            ignore_missing_files: flag("spark.sql.files.ignoreMissingFiles"),
            needs: MetadataNeeds::default(),
        }
    }
}

/// Prefetched bytes, held until the decoder asks for them.
struct Chunk {
    range: Range<u64>,
    bytes: Bytes,
    /// Bytes of it already handed to the decoder.
    served: u64,
}

type Pending = (u64, BoxFuture<'static, Result<Vec<Bytes>>>, Vec<Range<u64>>);

/// Reads one split of one file (see the module docs).
pub struct ScanStream {
    decoder: Option<Box<dyn Decoder>>,
    mapper: Option<SchemaMapper>,
    io: IoContext,
    file: FileRef,
    mem: Reservation,
    options: ScanOptions,
    ready: Vec<Chunk>,
    in_flight: Option<Pending>,
    done: bool,
}

impl std::fmt::Debug for ScanStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanStream")
            .field("file", &self.file.display_path())
            .field("reserved", &self.mem.size())
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

fn is_class(e: &KrishivError, classes: &[ErrorClass]) -> bool {
    classes.contains(&e.error_class())
}

const CORRUPT: &[ErrorClass] = &[
    ErrorClass::FailedReadFileNoHint,
    ErrorClass::FailedReadFileCannotReadFileFooter,
];
const MISSING: &[ErrorClass] = &[ErrorClass::FailedReadFileFileNotExist];

impl ScanStream {
    /// Opens `split` of `file` for the table-level request `req`, holding
    /// its memory in `mem`.
    ///
    /// # Errors
    /// Metadata, mapping or open errors (`FAILED_READ_FILE.*` and friends), unless
    /// `options` ignores missing or corrupt files: the stream is then empty.
    #[expect(
        clippy::too_many_arguments,
        reason = "one call per split, all inputs distinct"
    )]
    pub async fn open(
        io: &IoContext,
        format: &dyn FileFormat,
        file: FileRef,
        split: ByteRange,
        req: &ScanRequest,
        format_options: &FormatOptions,
        mem: Reservation,
        options: ScanOptions,
    ) -> Result<Self> {
        let mut stream = Self {
            decoder: None,
            mapper: None,
            io: io.clone(),
            file,
            mem,
            options,
            ready: Vec::new(),
            in_flight: None,
            done: false,
        };
        match stream.open_inner(format, split, req, format_options).await {
            Ok(()) => Ok(stream),
            Err(e) if stream.ignored(&e) => {
                stream.done = true;
                Ok(stream)
            }
            Err(e) => Err(e),
        }
    }

    fn ignored(&self, e: &KrishivError) -> bool {
        let skip = (self.options.ignore_corrupt_files && is_class(e, CORRUPT))
            || (self.options.ignore_missing_files && is_class(e, MISSING));
        if skip {
            tracing::warn!(path = self.file.display_path(), error = %e, "skipped the rest of a file");
        }
        skip
    }

    async fn open_inner(
        &mut self,
        format: &dyn FileFormat,
        split: ByteRange,
        req: &ScanRequest,
        o: &FormatOptions,
    ) -> Result<()> {
        let path = self.file.display_path();
        let meta = self
            .io
            .read_metadata(format, &self.file, self.options.needs, o)
            .await?;
        let file_schema = if format.capabilities(o).self_describing {
            meta.schema()
        } else {
            std::sync::Arc::clone(&req.table_schema)
        };
        let mapper = SchemaMapper::new(req, file_schema, &path)?;
        let file_req = mapper.file_request(req);
        let units = format
            .plan_units(meta.as_ref(), &split, &file_req)
            .map_err(|e| e.into_error(&path))?;
        let decoder = format
            .open(&self.io, meta, units, &file_req, o)
            .await
            .map_err(|e| e.into_error(&path))?;
        self.decoder = Some(decoder);
        self.mapper = Some(mapper);
        Ok(())
    }

    /// The next batch in the table schema, or `None` at the end of the split.
    ///
    /// # Errors
    /// Decode, I/O, mapping and memory errors (corrupt files end the stream
    /// instead when ignored).
    pub async fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.done {
            return Ok(None);
        }
        match self.next_inner().await {
            Ok(Some(b)) => Ok(Some(b)),
            Ok(None) => {
                self.finish();
                Ok(None)
            }
            Err(e) if self.ignored(&e) => {
                self.finish();
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    fn finish(&mut self) {
        self.done = true;
        self.decoder = None;
        self.ready.clear();
        self.in_flight = None;
        let held = self.mem.size();
        self.mem.shrink(held);
    }

    async fn next_inner(&mut self) -> Result<Option<RecordBatch>> {
        let path = self.file.display_path();
        loop {
            let step = self
                .decoder
                .as_mut()
                .ok_or_else(|| internal("scan stream without a decoder"))?
                .step()
                .map_err(|e| e.into_error(&path))?;
            match step {
                Step::Done => return Ok(None),
                Step::Batch(batch) => {
                    self.account();
                    self.prefetch();
                    let mapper = self
                        .mapper
                        .as_ref()
                        .ok_or_else(|| internal("scan stream without a mapper"))?;
                    return mapper.map_batch(&batch).map(Some);
                }
                Step::NeedData(ranges) => {
                    let data = self.gather(ranges).await?;
                    self.decoder
                        .as_mut()
                        .ok_or_else(|| internal("scan stream without a decoder"))?
                        .push(data)
                        .map_err(|e| e.into_error(&path))?;
                    self.account();
                    self.prefetch();
                }
            }
        }
    }

    /// The bytes of `ranges`: from prefetched chunks where possible, the
    /// rest fetched now (after reserving it, with waiting).
    async fn gather(&mut self, ranges: Vec<Range<u64>>) -> Result<Vec<(Range<u64>, Bytes)>> {
        if let Some((_, fut, wanted)) = self.in_flight.take() {
            // The prefetch was issued for what the decoder asks next; a
            // failed prefetch is only a lost hint (the fetch below retries).
            if let Ok(bytes) = fut.await {
                for (range, bytes) in wanted.into_iter().zip(bytes) {
                    self.ready.push(Chunk {
                        range,
                        bytes,
                        served: 0,
                    });
                }
            }
            self.account();
        }
        let mut out = Vec::with_capacity(ranges.len());
        let mut missing = Vec::new();
        for r in ranges {
            match self.take_ready(&r) {
                Some(b) => out.push((r, b)),
                None => missing.push(r),
            }
        }
        if !missing.is_empty() {
            let need: u64 = missing.iter().map(|r| r.end - r.start).sum();
            let target = self.held_bytes().saturating_add(need);
            let grow = usize::try_from(target)
                .unwrap_or(usize::MAX)
                .saturating_sub(self.mem.size());
            if grow > 0 {
                self.mem.grow(grow).await?;
            }
            let fetched = self.io.get_ranges(&self.file, &missing).await?;
            out.extend(missing.into_iter().zip(fetched));
        }
        self.drop_served();
        Ok(out)
    }

    fn take_ready(&mut self, r: &Range<u64>) -> Option<Bytes> {
        let c = self
            .ready
            .iter_mut()
            .find(|c| c.range.start <= r.start && r.end <= c.range.end)?;
        let a = usize::try_from(r.start - c.range.start).ok()?;
        let b = usize::try_from(r.end - c.range.start).ok()?;
        c.served += r.end - r.start;
        Some(c.bytes.slice(a..b))
    }

    fn drop_served(&mut self) {
        self.ready
            .retain(|c| c.served < c.range.end - c.range.start);
    }

    /// Prefetched bytes held or in flight.
    fn prefetched(&self) -> u64 {
        let ready: u64 = self.ready.iter().map(|c| c.range.end - c.range.start).sum();
        ready + self.in_flight.as_ref().map_or(0, |(n, ..)| *n)
    }

    fn held_bytes(&self) -> u64 {
        let buffered = self.decoder.as_ref().map_or(0, |d| d.buffered_bytes()) as u64;
        buffered + self.prefetched()
    }

    /// Shrinks the reservation to what is held.
    fn account(&mut self) {
        let held = usize::try_from(self.held_bytes()).unwrap_or(usize::MAX);
        if self.mem.size() > held {
            let extra = self.mem.size() - held;
            self.mem.shrink(extra);
        }
    }

    /// Starts a prefetch of the decoder's upcoming ranges if none is in
    /// flight and the budget and the pool allow it.
    fn prefetch(&mut self) {
        if self.in_flight.is_some() {
            return;
        }
        let budget = self
            .options
            .prefetch_bytes
            .saturating_sub(self.prefetched());
        let Some(d) = self.decoder.as_ref().filter(|_| budget > 0) else {
            return;
        };
        let ranges: Vec<Range<u64>> = d
            .upcoming(budget)
            .into_iter()
            .filter(|r| {
                !self
                    .ready
                    .iter()
                    .any(|c| c.range.start <= r.start && r.end <= c.range.end)
            })
            .collect();
        let bytes: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        if bytes == 0 || bytes > budget {
            return;
        }
        let target = usize::try_from(self.held_bytes() + bytes).unwrap_or(usize::MAX);
        let grow = target.saturating_sub(self.mem.size());
        if grow > 0 && self.mem.try_grow(grow).is_err() {
            return;
        }
        let fut = self.io.fetch(self.file.clone(), ranges.clone());
        self.in_flight = Some((bytes, fut, ranges));
    }

    /// Drops prefetched bytes (the scan's response to a spill request) and
    /// returns the bytes released.
    pub fn release_prefetch(&mut self) -> usize {
        let before = self.mem.size();
        self.ready.clear();
        self.in_flight = None;
        self.account();
        before - self.mem.size()
    }

    /// Bytes reserved now.
    #[must_use]
    pub fn reserved(&self) -> usize {
        self.mem.size()
    }

    /// The decoder's counters.
    #[must_use]
    pub fn metrics(&self) -> ScanUnitMetrics {
        self.decoder
            .as_ref()
            .map(|d| d.metrics())
            .unwrap_or_default()
    }

    /// The mapper of this file (`None` when the file was skipped).
    #[must_use]
    pub fn mapper(&self) -> Option<&SchemaMapper> {
        self.mapper.as_ref()
    }

    /// Shared, for adapters that need the scheduler.
    #[must_use]
    pub fn io(&self) -> &IoContext {
        &self.io
    }

    /// The file being read.
    #[must_use]
    pub fn file(&self) -> &FileRef {
        &self.file
    }
}
