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
//! **Memory (ADR 0007).** Prefetched bytes are reserved in the stream's
//! [`Reservation`] with [`Reservation::try_grow`] only, and are the first
//! thing dropped ([`ScanStream::release_prefetch`]). What the decoder holds
//! (its buffered bytes, and the unit being decoded:
//! [`Decoder::working_bytes`], or before the unit's reader starts its
//! decoding estimate, [`Decoder::next_decode_bytes`]) is, with the query's
//! [`ScanGate`], the stream's share of the gate: grown before a unit's
//! bytes are fetched (the gate may make the stream wait for other streams'
//! units to end; it never aborts the query) and shrunk as the decoder
//! releases them. Without a gate it is the stream's reservation too, where
//! bytes the decoder needs to progress (one unit's column chunks, in
//! dependency order) are reserved with waiting ([`Reservation::grow`],
//! which may arbitrate) and the unit being decoded is not counted.
//! Dropping the stream returns everything.
//!
//! **Cancellation.** Every await is drop-safe: dropping a pending
//! [`ScanStream::next_batch`] cancels its requests; the decoder never saw a
//! partial push, so the next call asks again.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

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
use crate::scan_gate::ScanGate;
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

/// The prefetched chunks of a file, keyed by start offset (column chunks do
/// not overlap), with their total kept up to date: the scan asks after every
/// batch whether each upcoming range is held, so a lookup must not be a
/// linear scan of every held chunk (hundreds with a large prefetch budget).
#[derive(Default)]
struct Ready {
    chunks: BTreeMap<u64, Chunk>,
    bytes: u64,
}

impl Ready {
    fn insert(&mut self, chunk: Chunk) {
        self.bytes += chunk.range.end - chunk.range.start;
        if let Some(old) = self.chunks.insert(chunk.range.start, chunk) {
            self.bytes -= old.range.end - old.range.start;
        }
    }

    /// The bytes of `r` if a chunk holds all of it; a chunk is dropped once
    /// the decoder took all of its bytes.
    fn take(&mut self, r: &Range<u64>) -> Option<Bytes> {
        let (&key, c) = self.chunks.range_mut(..=r.start).next_back()?;
        if r.end > c.range.end {
            return None;
        }
        let a = usize::try_from(r.start - c.range.start).ok()?;
        let b = usize::try_from(r.end - c.range.start).ok()?;
        let bytes = c.bytes.slice(a..b);
        c.served += r.end - r.start;
        let len = c.range.end - c.range.start;
        if c.served >= len {
            self.chunks.remove(&key);
            self.bytes -= len;
        }
        Some(bytes)
    }

    fn holds(&self, r: &Range<u64>) -> bool {
        self.chunks
            .range(..=r.start)
            .next_back()
            .is_some_and(|(_, c)| r.end <= c.range.end)
    }

    fn clear(&mut self) {
        self.chunks.clear();
        self.bytes = 0;
    }
}

type Pending = (u64, BoxFuture<'static, Result<Vec<Bytes>>>, Vec<Range<u64>>);

/// Reads one split of one file (see the module docs).
pub struct ScanStream {
    decoder: Option<Box<dyn Decoder>>,
    /// The decoder's counters when it was dropped (the split ended).
    final_metrics: ScanUnitMetrics,
    final_row_filter: Option<bool>,
    mapper: Option<SchemaMapper>,
    io: IoContext,
    file: FileRef,
    mem: Reservation,
    options: ScanOptions,
    ready: Ready,
    in_flight: Option<Pending>,
    /// The last prefetch attempt found nothing to fetch within the budget
    /// and nothing it depends on (the held chunks, the decoder's position)
    /// changed since: the next attempt would find the same, so it is
    /// skipped instead of re-walking every upcoming range after each batch.
    settled: bool,
    done: bool,
    /// The query's gate and this stream's id in it (module docs).
    gate: Option<(Arc<ScanGate>, u64)>,
    /// A unit's share was asked for and the unit has not ended.
    unit_open: bool,
    /// The open unit's reader holds data (its working bytes were seen).
    reading: bool,
    /// Bytes of shares that ended over-committed, in total.
    overcommitted: u64,
}

impl Drop for ScanStream {
    fn drop(&mut self) {
        self.end_unit();
    }
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
            final_metrics: ScanUnitMetrics::default(),
            final_row_filter: None,
            mapper: None,
            io: io.clone(),
            file,
            mem,
            options,
            ready: Ready::default(),
            in_flight: None,
            settled: false,
            done: false,
            gate: None,
            unit_open: false,
            reading: false,
            overcommitted: 0,
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

    /// Shares units' working sets with the query's other streams through
    /// `gate` (module docs). Without a gate they are not accounted.
    #[must_use]
    pub fn with_gate(mut self, gate: Arc<ScanGate>) -> Self {
        let id = gate.stream_id();
        self.gate = Some((gate, id));
        self
    }

    /// Bytes of decode working sets that ended a unit over-committed
    /// (without a reservation; see [`ScanGate`]).
    #[must_use]
    pub fn overcommitted(&self) -> u64 {
        self.overcommitted
    }

    /// What the gate share covers (module docs): the decoder's buffered
    /// bytes and the open unit's working set or decoding estimate.
    fn share_target(&self) -> usize {
        let Some(d) = self.decoder.as_ref() else {
            return 0;
        };
        let unit = match d.working_bytes() {
            0 if self.unit_open => d.next_decode_bytes(),
            w => w,
        };
        d.buffered_bytes().saturating_add(unit)
    }

    /// Grows the gate share by what the decoder is about to hold: `incoming`
    /// bytes pushed into it and, for a unit that starts, its decoding
    /// estimate. A stream that holds nothing may wait here for other
    /// streams (module docs).
    async fn grow_share(&mut self, incoming: u64) -> Result<()> {
        let Some((gate, id)) = self.gate.clone() else {
            return Ok(());
        };
        let target = self
            .share_target()
            .saturating_add(usize::try_from(incoming).unwrap_or(usize::MAX));
        let held = gate.share(id);
        if target > held {
            // Over-committed bytes are counted when the share drops them
            // (`settle_share`, `end_unit`), which also covers revoked ones.
            gate.acquire(id, target - held).await?;
        }
        Ok(())
    }

    /// Ends the open unit once its reader has, and shrinks the gate share
    /// to what the decoder holds.
    fn settle_share(&mut self) {
        let working = self.decoder.as_ref().map_or(0, |d| d.working_bytes());
        if working > 0 {
            self.reading = true;
        } else if self.reading {
            self.reading = false;
            self.unit_open = false;
        }
        if let Some((gate, id)) = &self.gate {
            let target = self.share_target();
            if gate.share(*id) > target {
                let over = gate.shrink(*id, target);
                self.overcommitted = self
                    .overcommitted
                    .saturating_add(u64::try_from(over).unwrap_or(u64::MAX));
            }
        }
    }

    fn end_unit(&mut self) {
        if let Some((gate, id)) = &self.gate {
            let over = gate.shrink(*id, 0);
            self.overcommitted = self
                .overcommitted
                .saturating_add(u64::try_from(over).unwrap_or(u64::MAX));
        }
        self.unit_open = false;
        self.reading = false;
    }

    fn finish(&mut self) {
        self.done = true;
        self.end_unit();
        if let Some(d) = self.decoder.take() {
            self.final_metrics = d.metrics();
            self.final_row_filter = d.row_filter_measured();
        }
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
                    // A unit whose data was prefetched may have followed the
                    // last one without a request: its share, before the batch
                    // moves on.
                    self.grow_share(0).await?;
                    self.prefetch();
                    let mapper = self
                        .mapper
                        .as_ref()
                        .ok_or_else(|| internal("scan stream without a mapper"))?;
                    return mapper.map_batch(&batch).map(Some);
                }
                Step::NeedData(ranges) => {
                    // A unit that ended returns its share (and lets waiting
                    // streams in) before this one asks for the next.
                    self.account();
                    if !self.reading {
                        self.unit_open = true;
                    }
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
        // The decoder moved on and the held chunks change: worth a new look.
        self.settled = false;
        // Ranges come from file metadata (untrusted): an inverted one is a
        // corrupt file, never an overflow.
        if let Some(bad) = ranges.iter().find(|r| r.start > r.end) {
            return Err(crate::FormatError::Corrupt(format!(
                "the decoder asked for the inverted range {bad:?}"
            ))
            .into_error(&self.file.display_path()));
        }
        if let Some((_, fut, wanted)) = self.in_flight.take() {
            // The prefetch was issued for what the decoder asks next; a
            // failed prefetch is only a lost hint (the fetch below retries).
            if let Ok(bytes) = fut.await {
                for (range, bytes) in wanted.into_iter().zip(bytes) {
                    self.ready.insert(Chunk {
                        range,
                        bytes,
                        served: 0,
                    });
                }
            }
            self.account();
        }
        // With a gate, everything pushed into the decoder is the share's,
        // asked for before anything is fetched.
        let incoming = ranges.iter().map(|r| r.end - r.start).sum();
        self.grow_share(incoming).await?;
        let mut out = Vec::with_capacity(ranges.len());
        let mut missing = Vec::new();
        for r in ranges {
            match self.ready.take(&r) {
                Some(b) => out.push((r, b)),
                None => missing.push(r),
            }
        }
        if !missing.is_empty() {
            let need: u64 = missing.iter().map(|r| r.end - r.start).sum();
            // The read's coalescing gaps are held until its ranges are
            // copied out: reserved for the read, released by `account`.
            let gaps = self
                .io
                .request_bytes(&self.file, &missing)
                .saturating_sub(need);
            // With a gate the ranges are the share's (`grow_share`), so the
            // reservation covers only the gaps.
            let target = match self.gate {
                Some(_) => self.held_bytes().saturating_add(gaps),
                None => self.held_bytes().saturating_add(need).saturating_add(gaps),
            };
            let grow = usize::try_from(target)
                .unwrap_or(usize::MAX)
                .saturating_sub(self.mem.size());
            if grow > 0 {
                self.mem.grow(grow).await?;
            }
            let fetched = self.io.get_ranges(&self.file, &missing).await?;
            out.extend(missing.into_iter().zip(fetched));
        }
        Ok(out)
    }

    /// Prefetched bytes held or in flight.
    fn prefetched(&self) -> u64 {
        self.ready.bytes + self.in_flight.as_ref().map_or(0, |(n, ..)| *n)
    }

    /// What the reservation covers: prefetched bytes, and without a gate
    /// the decoder's buffered bytes.
    fn held_bytes(&self) -> u64 {
        let buffered = match &self.gate {
            Some(_) => 0,
            None => self.decoder.as_ref().map_or(0, |d| d.buffered_bytes()) as u64,
        };
        buffered + self.prefetched()
    }

    /// Shrinks the reservation to what is held, and returns the share of a
    /// unit that ended.
    fn account(&mut self) {
        let held = usize::try_from(self.held_bytes()).unwrap_or(usize::MAX);
        if self.mem.size() > held {
            let extra = self.mem.size() - held;
            self.mem.shrink(extra);
        }
        self.settle_share();
    }

    /// Starts a prefetch of the decoder's upcoming ranges if none is in
    /// flight and the budget and the pool allow it.
    fn prefetch(&mut self) {
        if self.in_flight.is_some() || self.settled {
            return;
        }
        let budget = self
            .options
            .prefetch_bytes
            .saturating_sub(self.prefetched());
        let Some(d) = self.decoder.as_ref().filter(|_| budget > 0) else {
            self.settled = true;
            return;
        };
        let ranges: Vec<Range<u64>> = d
            .upcoming(budget)
            .into_iter()
            .filter(|r| r.start < r.end)
            .filter(|r| !self.ready.holds(r))
            .collect();
        let bytes: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        if bytes == 0 || bytes > budget {
            self.settled = true;
            return;
        }
        // In flight, the reads hold their coalescing gaps too: reserved
        // until they complete (`prefetched` counts `span`), then only the
        // ranges stay (`ready`).
        let span = self.io.request_bytes(&self.file, &ranges).max(bytes);
        let target = usize::try_from(self.held_bytes() + span).unwrap_or(usize::MAX);
        let grow = target.saturating_sub(self.mem.size());
        if grow > 0 && self.mem.try_grow(grow).is_err() {
            return;
        }
        let fut = self.io.fetch(self.file.clone(), ranges.clone());
        self.in_flight = Some((span, fut, ranges));
    }

    /// Drops prefetched bytes and returns the bytes released.
    pub fn release_prefetch(&mut self) -> usize {
        let before = self.mem.size();
        self.ready.clear();
        self.in_flight = None;
        self.settled = false;
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
            .map_or(self.final_metrics, |d| d.metrics())
    }

    /// What this stream last measured for its row-level filter
    /// ([`Decoder::row_filter_measured`]), for the next piece of the file to
    /// start from ([`ScanRequest::row_filter_start`]).
    #[must_use]
    pub fn row_filter_measured(&self) -> Option<bool> {
        self.decoder
            .as_ref()
            .map_or(self.final_row_filter, |d| d.row_filter_measured())
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

#[cfg(test)]
mod tests {
    use super::{Bytes, Chunk, Ready};

    fn chunk(start: u64, len: u64) -> Chunk {
        let bytes: Vec<u8> = (0..len)
            .map(|i| ((start + i) % 256).to_le_bytes()[0])
            .collect();
        Chunk {
            range: start..start + len,
            bytes: Bytes::from(bytes),
            served: 0,
        }
    }

    /// Lookups find the chunk holding a range (or none), a chunk leaves once
    /// all its bytes were taken, and the held total follows every change.
    #[test]
    fn ready_chunks_are_found_taken_and_counted() {
        let mut r = Ready::default();
        r.insert(chunk(100, 50));
        r.insert(chunk(10, 20));
        assert_eq!(r.bytes, 70);
        assert!(r.holds(&(10..30)) && r.holds(&(120..130)));
        assert!(!r.holds(&(5..15)) && !r.holds(&(25..35)) && !r.holds(&(140..160)));
        assert!(!r.holds(&(0..1)) && !r.holds(&(200..210)));
        assert_eq!(r.take(&(12..14)).unwrap().as_ref(), &[12, 13]);
        assert_eq!(r.take(&(25..40)), None);
        assert_eq!(r.bytes, 70, "a partly taken chunk stays");
        assert_eq!(r.take(&(10..30)).unwrap().len(), 20);
        assert_eq!(r.bytes, 50, "a fully taken chunk leaves");
        assert!(!r.holds(&(10..12)));
        // A chunk at the same offset replaces the one held.
        r.insert(chunk(100, 10));
        assert_eq!(r.bytes, 10);
        r.clear();
        assert_eq!((r.bytes, r.chunks.len()), (0, 0));
    }
}
