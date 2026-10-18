// SPDX-License-Identifier: Apache-2.0
//! The [`FileFormat`] trait and its sans-IO [`Decoder`] / [`Encoder`].
//!
//! # Reading one split of one file
//! 1. [`IoContext::read_metadata`](crate::IoContext::read_metadata) returns
//!    the file's [`FileMetadata`] from the cache or through
//!    [`FileFormat::read_metadata`] (one speculative tail read, then exact
//!    reads only if the footer is larger).
//! 2. The framework maps the table-level [`ScanRequest`] to the file's names
//!    and types ([`SchemaMapper`](crate::SchemaMapper)).
//! 3. [`FileFormat::plan_units`] picks the units (row groups, chunks) the
//!    split owns (the **midpoint rule**, [`ByteRange::owns`]) and prunes them
//!    with statistics: pruning stage 1, synchronous, no I/O.
//! 4. [`FileFormat::open`] returns a [`Decoder`], a state machine the
//!    framework drives with [`Decoder::step`]: it asks for byte ranges
//!    ([`Step::NeedData`], in dependency order: predicate columns, then
//!    dictionary pages and page indexes for pruning stages 2–3, then the
//!    surviving projection), is pushed the bytes, and returns batches in the
//!    file-level output schema ([`ScanRequest::output_schema`]).
//!
//! A step does about one batch of work and never blocks or awaits; every
//! await is in the framework, which is drop-safe (cancellation).
//!
//! Libraries that only offer an async "read at" interface fit through
//! [`AsyncDecoder`]: their reads go to a [`PushReadAt`] that turns them into
//! [`Step::NeedData`], so their I/O still goes through the scheduler.

use std::any::Any;
use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use bytes::Bytes;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;

use crate::error::{FormatError, FormatResult, internal};
use crate::io::{ByteRange, FileRef, IoContext};
use crate::options::{FormatOptions, OptionSchema};
use crate::request::ScanRequest;
use crate::stats::FileStatistics;

/// One file format (Parquet, CSV, Vortex, ...), implemented once.
///
/// Every method that can fail returns a [`FormatError`]; the framework adds
/// the file's redacted path. A format reads bytes only through the
/// [`IoContext`] (never `object_store`), so caching, budgets, counters,
/// retries and version checks apply to every byte.
pub trait FileFormat: Send + Sync + 'static {
    /// The registry key and Spark `format(...)` name, e.g. `parquet`.
    fn name(&self) -> &'static str;

    /// The options this format understands.
    fn options(&self) -> &'static OptionSchema;

    /// What this format can do with these options.
    fn capabilities(&self, o: &FormatOptions) -> Capabilities;

    /// The suffix of files it writes, after Spark's `-c000`: e.g.
    /// `.snappy.parquet`, `.csv.gz`.
    fn file_suffix(&self, o: &FormatOptions) -> String;

    /// The file-native schema of `files` (one file unless the format samples
    /// several, as CSV/JSON inference does). Merging schemas across files
    /// (`mergeSchema`) is the framework's. The default reads the first
    /// file's metadata through the cache.
    fn infer_schema<'a>(
        &'a self,
        io: &'a IoContext,
        files: &'a [FileRef],
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<SchemaRef>> {
        Box::pin(async move {
            let Some(first) = files.first() else {
                return Err(FormatError::Other(internal(
                    "schema inference without files",
                )));
            };
            let m = io
                .read_metadata(self, first, MetadataNeeds::default(), o)
                .await?;
            Ok(m.schema())
        })
    }

    /// Reads the file's metadata (footer, plus the page index or bloom
    /// filters when `needs` asks). Called through
    /// [`IoContext::read_metadata`](crate::IoContext::read_metadata), which
    /// caches the result. Use [`IoContext::get_tail`] for the footer.
    /// Resolve decryption keys here with
    /// [`FormatOptions::resolve_secret`], and drop them before returning.
    fn read_metadata<'a>(
        &'a self,
        io: &'a IoContext,
        file: &'a FileRef,
        needs: MetadataNeeds,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Arc<dyn FileMetadata>>>;

    /// The units of `split` to read, pruned with statistics (stage 1):
    /// synchronous, no I/O. `req` is file-level. A unit belongs to the split
    /// holding its byte midpoint, so any split boundaries read each unit
    /// exactly once.
    ///
    /// # Errors
    /// [`FormatError`] if `m` is not this format's metadata.
    fn plan_units(
        &self,
        m: &dyn FileMetadata,
        split: &ByteRange,
        req: &ScanRequest,
    ) -> FormatResult<UnitPlan>;

    /// A decoder for `units` (from [`Self::plan_units`]). `req` is
    /// file-level. Async only so that column decryption keys can be
    /// resolved here (and dropped with the decoder); data bytes are fetched
    /// by the decoder's [`Step::NeedData`], never here.
    fn open<'a>(
        &'a self,
        io: &'a IoContext,
        m: Arc<dyn FileMetadata>,
        units: UnitPlan,
        req: &'a ScanRequest,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Box<dyn Decoder>>>;

    /// An encoder for one file of `schema`. Formats that cannot write return
    /// `KRISHIV_UNSUPPORTED_FEATURE`. Async only for key resolution, as
    /// [`Self::open`].
    fn writer<'a>(
        &'a self,
        io: &'a IoContext,
        schema: SchemaRef,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Box<dyn Encoder>>>;

    /// Checks, when a write is planned, that files of `schema` can be
    /// written with `o` (a known codec, a supported timestamp type), so a
    /// bad setting fails before any task runs. [`Self::writer`] may still
    /// fail later on the data itself. Default: no checks.
    ///
    /// # Errors
    /// The error the write would fail with.
    fn check_write(&self, schema: &SchemaRef, o: &FormatOptions) -> FormatResult<()> {
        let _ = (schema, o);
        Ok(())
    }
}

/// What [`FileFormat::read_metadata`] must load beyond the footer. A cached
/// entry serves a request it [covers](Self::covers); otherwise the metadata
/// is read again with both sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub struct MetadataNeeds {
    /// The page index (column and offset indexes), for page pruning.
    pub page_index: bool,
    /// Bloom filters.
    pub bloom: bool,
}

impl MetadataNeeds {
    /// What to load besides the footer.
    #[must_use]
    pub fn new(page_index: bool, bloom: bool) -> Self {
        Self { page_index, bloom }
    }

    /// Whether metadata loaded with `self` serves a request for `other`.
    #[must_use]
    pub fn covers(&self, other: &Self) -> bool {
        (self.page_index || !other.page_index) && (self.bloom || !other.bloom)
    }

    /// Both sets.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            page_index: self.page_index || other.page_index,
            bloom: self.bloom || other.bloom,
        }
    }
}

/// A file's decoded metadata, cached by the framework. Formats downcast
/// their own type from `Arc<dyn FileMetadata>` (trait upcasting to
/// `Arc<dyn Any + Send + Sync>`).
pub trait FileMetadata: Any + Send + Sync + fmt::Debug {
    /// The file-native schema.
    fn schema(&self) -> SchemaRef;
    /// Whole-file statistics.
    fn statistics(&self) -> FileStatistics;
    /// Heap bytes held, for the cache's byte bound.
    fn heap_size(&self) -> usize;
    /// What was loaded besides the footer.
    fn needs(&self) -> MetadataNeeds {
        MetadataNeeds::default()
    }
}

/// Whether files of a format can be read in byte ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Splittable {
    /// Never (`multiLine` JSON/CSV, whole-file formats).
    No,
    /// Always (Parquet: units are found from the footer).
    Yes,
    /// By the file's compression suffix, as Hadoop codecs decide: none or
    /// `.bz2` yes; `.gz`, `.deflate`, `.lz4`, `.snappy`, `.zst` no.
    ByCodec,
}

/// What a format can do (for the planner and the framework).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent capability flags the planner checks one by one"
)]
#[non_exhaustive]
pub struct Capabilities {
    /// Whether files can be read in byte-range splits.
    pub splittable: Splittable,
    /// Whether files carry their own schema (Parquet, Avro, Vortex). When
    /// not (CSV, JSON, text), [`FileMetadata::schema`] is not used: the
    /// framework treats the table's data schema as the file's, and the
    /// decoder parses into the file-level request's types.
    pub self_describing: bool,
    /// Whether metadata carries statistics (pruning, the optimizer).
    pub statistics: bool,
    /// Whether the decoder produces `_metadata.row_index`.
    pub row_index: bool,
    /// Whether the decoder honours [`ScanRequest::selection`]. The planner
    /// pushes a selection only into formats that do.
    pub selection: bool,
    /// Whether page-level pruning (stage 3) is implemented.
    pub page_pruning: bool,
    /// Whether page checksums are verified (`pageChecksumVerification`).
    pub page_checksums: bool,
    /// Whether [`FileFormat::writer`] is implemented.
    pub write: bool,
}

/// Compression suffixes whose Hadoop codec cannot split (observed with
/// Spark 4.2's text source).
const UNSPLITTABLE_SUFFIXES: &[&str] = &[".gz", ".deflate", ".lz4", ".snappy", ".zst"];

impl Capabilities {
    /// Capabilities with `splittable` and nothing else; set the other
    /// fields on the result.
    #[must_use]
    pub fn new(splittable: Splittable) -> Self {
        Self {
            splittable,
            self_describing: false,
            statistics: false,
            row_index: false,
            selection: false,
            page_pruning: false,
            page_checksums: false,
            write: false,
        }
    }

    /// Whether the file at `path` can be split.
    #[must_use]
    pub fn is_splittable(&self, path: &str) -> bool {
        match self.splittable {
            Splittable::No => false,
            Splittable::Yes => true,
            Splittable::ByCodec => !UNSPLITTABLE_SUFFIXES.iter().any(|s| path.ends_with(s)),
        }
    }
}

/// One unit of a file a decoder reads (a row group, a stripe, a chunk).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanUnit {
    /// The format's index of the unit.
    pub index: usize,
    /// Position of the unit's first row in the file (for `row_index` and
    /// [`RowSelection`](crate::RowSelection)).
    pub first_row: u64,
    /// Rows in the unit.
    pub rows: u64,
    /// The unit's bytes in the file (its midpoint decides its split).
    pub bytes: Range<u64>,
}

/// The units a decoder reads, after stage-1 pruning.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UnitPlan {
    /// Units to read, in file order.
    pub units: Vec<ScanUnit>,
    /// Units the split owns (before pruning).
    pub owned: usize,
    /// Owned units skipped by statistics.
    pub pruned_by_statistics: usize,
}

/// What a decoder needs or produced.
#[derive(Debug)]
#[non_exhaustive]
pub enum Step {
    /// Push these byte ranges (absolute file offsets), then step again. A
    /// step without the push returns the same ranges.
    NeedData(Vec<Range<u64>>),
    /// One batch in the file-level output schema.
    Batch(RecordBatch),
    /// No more batches.
    Done,
}

/// Counters of one decoder (`EXPLAIN ANALYZE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ScanUnitMetrics {
    /// Units the decoder was opened with.
    pub units: u64,
    /// Units skipped by dynamic filters or dictionaries (stage 2).
    pub units_pruned: u64,
    /// Pages skipped by the page index (stage 3).
    pub pages_pruned: u64,
    /// Rows decoded.
    pub rows: u64,
    /// Bytes pushed into the decoder.
    pub bytes_pushed: u64,
}

/// A sans-IO decoding state machine (see the module docs).
pub trait Decoder: Send {
    /// Does about one batch of work. Never blocks, never awaits.
    ///
    /// # Errors
    /// [`FormatError::Corrupt`] for undecodable data.
    fn step(&mut self) -> FormatResult<Step>;

    /// Hands over bytes for ranges asked by [`Step::NeedData`] (or listed by
    /// [`Self::upcoming`]); each pair is a range and its bytes.
    ///
    /// # Errors
    /// [`FormatError`] if the bytes cannot be used.
    fn push(&mut self, data: Vec<(Range<u64>, Bytes)>) -> FormatResult<()>;

    /// Ranges the decoder will ask for next, up to `max_bytes`, for
    /// prefetching; never needed for progress. Default: none.
    fn upcoming(&self, max_bytes: u64) -> Vec<Range<u64>> {
        let _ = max_bytes;
        Vec::new()
    }

    /// Bytes pushed and not yet released (the framework reserves them).
    fn buffered_bytes(&self) -> usize;

    /// Counters so far.
    fn metrics(&self) -> ScanUnitMetrics;
}

/// What an encoder produced for one file.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WrittenFile {
    /// The last bytes of the file (footer).
    pub tail: Bytes,
    /// The statistics a reader of the file reports.
    pub statistics: FileStatistics,
}

impl WrittenFile {
    /// The file's last bytes and its statistics.
    #[must_use]
    pub fn new(tail: Bytes, statistics: FileStatistics) -> Self {
        Self { tail, statistics }
    }
}

/// A sans-IO encoder for one file: batches in, bytes out. The framework
/// uploads the bytes (multipart) and reserves [`Self::buffered_bytes`].
pub trait Encoder: Send {
    /// Encodes one batch. Never blocks.
    ///
    /// # Errors
    /// [`FormatError`] if the batch cannot be encoded.
    fn write(&mut self, batch: &RecordBatch) -> FormatResult<()>;

    /// Encodes buffered rows now (ends the current row group or chunk early)
    /// so that their memory can be released; called under memory pressure.
    /// Default: nothing buffered beyond [`Self::take_bytes`].
    ///
    /// # Errors
    /// [`FormatError`] if encoding fails.
    fn flush(&mut self) -> FormatResult<()> {
        Ok(())
    }

    /// Encoded bytes ready to upload, in file order.
    fn take_bytes(&mut self) -> Option<Bytes>;

    /// Ends the file: the remaining bytes and its statistics.
    ///
    /// # Errors
    /// [`FormatError`] if encoding fails.
    fn finish(self: Box<Self>) -> FormatResult<WrittenFile>;

    /// Bytes held in memory (rows not yet encoded plus encoded bytes not yet
    /// taken).
    fn buffered_bytes(&self) -> usize;

    /// Bytes of the file produced so far (taken or not), for rollover at a
    /// target file size.
    fn written_bytes(&self) -> u64;
}

/// Random-access reads for libraries with an async interface. The only
/// implementations are over the scheduler: [`IoContext::reader`] (for
/// metadata, which may await) and [`PushReadAt`] (inside a decoder).
pub trait ReadAt: Send + Sync {
    /// The file size.
    fn size(&self) -> u64;
    /// The bytes of `range`.
    fn read_at(&self, range: Range<u64>) -> BoxFuture<'_, FormatResult<Bytes>>;
}

#[derive(Debug, Default)]
struct PushState {
    /// Ranges asked for and not yet pushed, in request order.
    wanted: Vec<Range<u64>>,
    /// Pushed bytes not yet read in full.
    data: Vec<Pushed>,
}

/// One pushed range and the parts of it no read has taken yet: it is
/// released once every byte was read (by one read or several).
#[derive(Debug)]
struct Pushed {
    range: Range<u64>,
    bytes: Bytes,
    unread: Vec<Range<u64>>,
}

impl Pushed {
    fn new(range: Range<u64>, bytes: Bytes) -> Self {
        let unread = vec![range.clone()];
        Self {
            range,
            bytes,
            unread,
        }
    }

    /// Marks `read` as read.
    fn consume(&mut self, read: &Range<u64>) {
        self.unread = std::mem::take(&mut self.unread)
            .into_iter()
            .flat_map(|u| {
                let left = u.start..read.start.min(u.end);
                let right = read.end.max(u.start)..u.end;
                [left, right]
            })
            .filter(|r| r.start < r.end)
            .collect();
    }
}

/// A [`ReadAt`] fed by pushes: a read of bytes not pushed yet stays pending
/// and is reported by [`AsyncDecoder::step`] as [`Step::NeedData`].
#[derive(Debug)]
pub struct PushReadAt {
    size: u64,
    state: Mutex<PushState>,
}

impl PushReadAt {
    /// A reader of a file of `size` bytes.
    #[must_use]
    pub fn new(size: u64) -> Arc<Self> {
        Arc::new(Self {
            size,
            state: Mutex::new(PushState::default()),
        })
    }

    fn state(&self) -> std::sync::MutexGuard<'_, PushState> {
        // Poisoning needs a panic while holding the lock; the critical
        // sections below cannot panic, so recover the data.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn poll_read(&self, range: &Range<u64>) -> Poll<FormatResult<Bytes>> {
        if range.end > self.size || range.start > range.end {
            return Poll::Ready(Err(FormatError::Corrupt(format!(
                "read of {range:?} beyond the end of a {}-byte file",
                self.size
            ))));
        }
        let mut st = self.state();
        let hit = st
            .data
            .iter()
            .position(|p| p.range.start <= range.start && range.end <= p.range.end);
        if let Some(i) = hit {
            let p = &mut st.data[i];
            let slice = usize::try_from(range.start - p.range.start)
                .ok()
                .zip(usize::try_from(range.end - p.range.start).ok())
                .map(|(a, b)| p.bytes.slice(a..b));
            p.consume(range);
            if p.unread.is_empty() {
                st.data.swap_remove(i);
            }
            return Poll::Ready(
                slice.ok_or_else(|| FormatError::Corrupt("range overflows memory".to_owned())),
            );
        }
        if !st.wanted.contains(range) {
            st.wanted.push(range.clone());
        }
        Poll::Pending
    }
}

impl ReadAt for PushReadAt {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_at(&self, range: Range<u64>) -> BoxFuture<'_, FormatResult<Bytes>> {
        Box::pin(std::future::poll_fn(move |_| self.poll_read(&range)))
    }
}

/// A [`Decoder`] over an async batch stream whose only awaits are reads of a
/// [`PushReadAt`] (a library with an async "read at" interface). Each step
/// polls the stream once; reads it is blocked on become
/// [`Step::NeedData`]. The stream must not await anything else (timers,
/// spawned tasks): a stream pending without a read is an error.
pub struct AsyncDecoder {
    stream: BoxStream<'static, FormatResult<RecordBatch>>,
    reader: Arc<PushReadAt>,
    done: bool,
    metrics: ScanUnitMetrics,
}

impl fmt::Debug for AsyncDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncDecoder")
            .field("reader", &self.reader)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl AsyncDecoder {
    /// Drives `stream`, whose reads go to `reader`.
    #[must_use]
    pub fn new(
        stream: BoxStream<'static, FormatResult<RecordBatch>>,
        reader: Arc<PushReadAt>,
    ) -> Self {
        Self {
            stream,
            reader,
            done: false,
            metrics: ScanUnitMetrics::default(),
        }
    }
}

impl Decoder for AsyncDecoder {
    fn step(&mut self) -> FormatResult<Step> {
        if self.done {
            return Ok(Step::Done);
        }
        {
            let st = self.reader.state();
            if !st.wanted.is_empty() {
                return Ok(Step::NeedData(st.wanted.clone()));
            }
        }
        let mut cx = Context::from_waker(Waker::noop());
        match self.stream.poll_next_unpin(&mut cx) {
            Poll::Ready(Some(Ok(batch))) => {
                self.metrics.rows += batch.num_rows() as u64;
                Ok(Step::Batch(batch))
            }
            Poll::Ready(Some(Err(e))) => Err(e),
            Poll::Ready(None) => {
                self.done = true;
                Ok(Step::Done)
            }
            Poll::Pending => {
                let wanted = self.reader.state().wanted.clone();
                if wanted.is_empty() {
                    Err(FormatError::Other(internal(
                        "an async decoder awaited something other than a read",
                    )))
                } else {
                    Ok(Step::NeedData(wanted))
                }
            }
        }
    }

    fn push(&mut self, data: Vec<(Range<u64>, Bytes)>) -> FormatResult<()> {
        // Reads slice pushed bytes by range: a push must hold exactly its
        // range's bytes, or a later read would slice out of bounds.
        if let Some((r, b)) = data
            .iter()
            .find(|(r, b)| r.start > r.end || b.len() as u64 != r.end - r.start)
        {
            return Err(FormatError::Other(internal(format!(
                "pushed {} bytes for the range {r:?}",
                b.len()
            ))));
        }
        let mut st = self.reader.state();
        for (r, b) in data {
            self.metrics.bytes_pushed += b.len() as u64;
            st.wanted
                .retain(|w| !(r.start <= w.start && w.end <= r.end));
            st.data.push(Pushed::new(r, b));
        }
        Ok(())
    }

    fn buffered_bytes(&self) -> usize {
        self.reader.state().data.iter().map(|p| p.bytes.len()).sum()
    }

    fn metrics(&self) -> ScanUnitMetrics {
        self.metrics
    }
}
