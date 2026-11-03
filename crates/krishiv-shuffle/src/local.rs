// SPDX-License-Identifier: Apache-2.0
//! [`LocalShuffle`]: the [`Exchange`] of the tasks of one process (design
//! P2-T176 §2.3, blocking mode). Map outputs are frames of format v1
//! ([`crate::frame`]), kept in memory or, with a spill scope, appended to
//! one file per map output attempt in the scope's directory, so a stage's
//! output does not hold memory once written.
//!
//! - **Commit before read.** A writer's frames are its own until
//!   [`commit`](ExchangeWriter::commit); the first committed attempt of a
//!   map partition is the one every reader reads, and a later commit of the
//!   same map partition is dropped with its file (I10). A writer dropped
//!   without a commit removes its file.
//! - **Checks on read.** Each frame's checksum and version are checked, and
//!   its header must name the map output, reduce partition and sequence the
//!   reader expects; a reduce partition must end in an `End` frame. Damage
//!   is `KRISHIV_SPILL_FILE_CORRUPT`, never wrong rows.
//!
//! In memory, committed frames are not charged to any pool: that mode is for
//! tests and the simulation, whose stages are small. Files are not counted
//! against the spill cap.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write as _;
use std::os::unix::fs::FileExt as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_common::error::{
    InternalError, KrishivError, KrishivSpillFileCorrupt, KrishivSpillIoError,
};
use krishiv_exec::exchange::{
    Exchange, ExchangeFrame, ExchangeReader, ExchangeWriter, MapOutputId,
};
use krishiv_exec::physical::ExchangeId;
use krishiv_memory::SpillScope;

use crate::frame::{self, FrameHeader, FrameKind, HEADER_BYTES};

type Result<T> = std::result::Result<T, KrishivError>;

/// The exchanges of one process's tasks (see the module docs).
#[derive(Debug)]
pub struct LocalShuffle {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    scope: Option<SpillScope>,
    next_file: AtomicU64,
    /// Committed map outputs, by exchange and stage attempt, then map
    /// partition: the first commit of each.
    committed: Mutex<HashMap<(ExchangeId, u32), Committed>>,
}

/// The committed outputs of one exchange at one stage attempt, by map
/// partition.
type Committed = BTreeMap<u32, Arc<MapOutput>>;

/// Where one frame is.
#[derive(Debug)]
enum Location {
    Memory(Vec<u8>),
    File { offset: u64, len: u64 },
}

/// One committed (or still written) map output.
#[derive(Debug)]
struct MapOutput {
    id: MapOutputId,
    /// The frames of each reduce partition, ending in `End` once committed.
    frames: Vec<Vec<Location>>,
    /// The file the frames on disk are in.
    file: Option<OutputFile>,
}

#[derive(Debug)]
struct OutputFile {
    path: PathBuf,
    file: File,
}

impl Drop for OutputFile {
    fn drop(&mut self) {
        // Best effort: the scope's directory goes with the scope anyway.
        let _ = std::fs::remove_file(&self.path);
    }
}

impl LocalShuffle {
    /// A shuffle keeping its frames in memory.
    #[must_use]
    pub fn in_memory() -> Self {
        Self::with(None)
    }

    /// A shuffle writing its frames to files in `scope`'s directory.
    #[must_use]
    pub fn on_disk(scope: SpillScope) -> Self {
        Self::with(Some(scope))
    }

    fn with(scope: Option<SpillScope>) -> Self {
        Self {
            inner: Arc::new(Inner {
                scope,
                next_file: AtomicU64::new(0),
                committed: Mutex::default(),
            }),
        }
    }

    /// The map partitions of `exchange` at `stage_attempt` with a committed
    /// output, and the attempt each reader reads.
    #[must_use]
    pub fn committed(&self, exchange: ExchangeId, stage_attempt: u32) -> BTreeMap<u32, u32> {
        self.inner
            .committed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&(exchange, stage_attempt))
            .map(|maps| maps.iter().map(|(m, o)| (*m, o.id.attempt)).collect())
            .unwrap_or_default()
    }
}

impl Exchange for LocalShuffle {
    fn writer(
        &self,
        output: MapOutputId,
        partitions: u32,
        schema: SchemaRef,
    ) -> Result<Box<dyn ExchangeWriter>> {
        if partitions == 0 {
            return Err(internal(format!(
                "exchange {} written to no partitions",
                output.exchange.0
            )));
        }
        let file = match &self.inner.scope {
            Some(scope) => {
                let n = self.inner.next_file.fetch_add(1, Ordering::Relaxed);
                let path = scope.dir().join(format!(
                    "x{}-s{}-m{}-a{}-{n}.frames",
                    output.exchange.0, output.stage_attempt, output.map, output.attempt
                ));
                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .map_err(|e| io("create", &path, &e))?;
                Some(OutputFile { path, file })
            }
            None => None,
        };
        Ok(Box::new(LocalWriter {
            shuffle: Arc::clone(&self.inner),
            schema,
            output: MapOutput {
                id: output,
                frames: (0..partitions).map(|_| Vec::new()).collect(),
                file,
            },
            offset: 0,
        }))
    }

    fn reader(
        &self,
        exchange: ExchangeId,
        stage_attempt: u32,
        map: u32,
        reduce: u32,
    ) -> Result<Box<dyn ExchangeReader>> {
        let output = self
            .inner
            .committed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&(exchange, stage_attempt))
            .and_then(|maps| maps.get(&map))
            .cloned()
            .ok_or_else(|| {
                internal(format!(
                    "exchange {} has no committed output of map partition {map} at stage attempt {stage_attempt}",
                    exchange.0
                ))
            })?;
        if output.frames.get(reduce as usize).is_none() {
            return Err(internal(format!(
                "reduce partition {reduce} of exchange {} is out of range",
                exchange.0
            )));
        }
        Ok(Box::new(LocalReader {
            output,
            reduce,
            next: 0,
        }))
    }
}

struct LocalWriter {
    shuffle: Arc<Inner>,
    schema: SchemaRef,
    output: MapOutput,
    /// The file's length so far.
    offset: u64,
}

impl LocalWriter {
    fn append(&mut self, reduce: u32, kind: FrameKind, body: &[u8]) -> Result<()> {
        let id = self.output.id;
        let frames = self
            .output
            .frames
            .get_mut(reduce as usize)
            .ok_or_else(|| internal(format!("reduce partition {reduce} is out of range")))?;
        let header = FrameHeader {
            kind,
            id: id.exchange.0,
            stage_attempt: id.stage_attempt,
            map: id.map,
            attempt: id.attempt,
            sequence: u32::try_from(frames.len())
                .map_err(|_| internal("too many frames in one partition".to_owned()))?,
            reduce,
        };
        let bytes = frame::encode(&header, body).map_err(|e| internal(e.to_string()))?;
        match &mut self.output.file {
            None => frames.push(Location::Memory(bytes)),
            Some(f) => {
                f.file
                    .write_all(&bytes)
                    .map_err(|e| io("write", &f.path, &e))?;
                let len = bytes.len() as u64;
                frames.push(Location::File {
                    offset: self.offset,
                    len,
                });
                self.offset += len;
            }
        }
        Ok(())
    }
}

impl ExchangeWriter for LocalWriter {
    fn write(&mut self, reduce: u32, batch: &RecordBatch) -> Result<()> {
        if batch.schema().fields() != self.schema.fields() {
            return Err(internal(format!(
                "exchange {} written with {:?} where it has {:?}",
                self.output.id.exchange.0,
                batch.schema().fields(),
                self.schema.fields()
            )));
        }
        let body = frame::batch_body(batch).map_err(|e| internal(e.to_string()))?;
        self.append(reduce, FrameKind::Data, &body)
    }

    fn commit(mut self: Box<Self>) -> Result<()> {
        for reduce in 0..self.output.frames.len() {
            let reduce = u32::try_from(reduce)
                .map_err(|_| internal(format!("reduce partition {reduce}")))?;
            self.append(reduce, FrameKind::End, &[])?;
        }
        let LocalWriter {
            shuffle, output, ..
        } = *self;
        let key = (output.id.exchange, output.id.stage_attempt);
        let map = output.id.map;
        // A later attempt of a committed map partition is dropped here,
        // with its file.
        shuffle
            .committed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(key)
            .or_default()
            .entry(map)
            .or_insert_with(|| Arc::new(output));
        Ok(())
    }
}

struct LocalReader {
    output: Arc<MapOutput>,
    reduce: u32,
    next: usize,
}

impl LocalReader {
    fn frame(&mut self) -> Result<ExchangeFrame> {
        let o = &self.output;
        let frames = o
            .frames
            .get(self.reduce as usize)
            .ok_or_else(|| internal(format!("reduce partition {} is out of range", self.reduce)))?;
        let at = self.next;
        let location = frames.get(at).ok_or_else(|| {
            corrupt(
                o,
                self.reduce,
                "the partition ended without its end frame".to_owned(),
            )
        })?;
        let read;
        let bytes: &[u8] = match location {
            Location::Memory(b) => b,
            Location::File { offset, len } => {
                let f = o
                    .file
                    .as_ref()
                    .ok_or_else(|| internal("a frame on disk without its file".to_owned()))?;
                let mut buf = vec![
                    0_u8;
                    usize::try_from(*len)
                        .map_err(|_| internal(format!("a frame of {len} bytes")))?
                ];
                f.file
                    .read_exact_at(&mut buf, *offset)
                    .map_err(|e| io("read", &f.path, &e))?;
                read = buf;
                &read
            }
        };
        if bytes.len() < HEADER_BYTES {
            return Err(corrupt(
                o,
                self.reduce,
                format!("a frame of {} bytes", bytes.len()),
            ));
        }
        let (header, body) =
            frame::decode(bytes).map_err(|e| corrupt(o, self.reduce, e.to_string()))?;
        let want = (
            o.id.exchange.0,
            o.id.stage_attempt,
            o.id.map,
            o.id.attempt,
            self.reduce,
            u32::try_from(at).unwrap_or(u32::MAX),
        );
        let got = (
            header.id,
            header.stage_attempt,
            header.map,
            header.attempt,
            header.reduce,
            header.sequence,
        );
        if got != want {
            return Err(corrupt(
                o,
                self.reduce,
                format!("frame {got:?} where {want:?} was expected"),
            ));
        }
        self.next += 1;
        match header.kind {
            FrameKind::Data => Ok(ExchangeFrame::Data(
                frame::body_batch(body).map_err(|e| corrupt(o, self.reduce, e.to_string()))?,
            )),
            FrameKind::End => Ok(ExchangeFrame::End),
            FrameKind::Value => Err(corrupt(
                o,
                self.reduce,
                "a value frame among exchange rows".to_owned(),
            )),
        }
    }
}

impl ExchangeReader for LocalReader {
    fn next(&mut self) -> BoxFuture<'_, Result<ExchangeFrame>> {
        let frame = self.frame();
        Box::pin(std::future::ready(frame))
    }
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

fn io(operation: &str, path: &std::path::Path, e: &std::io::Error) -> KrishivError {
    KrishivSpillIoError {
        operation: operation.to_owned(),
        path: path.display().to_string(),
        reason: e.to_string(),
    }
    .into()
}

fn corrupt(o: &MapOutput, reduce: u32, reason: String) -> KrishivError {
    let path = o.file.as_ref().map_or_else(
        || {
            format!(
                "exchange {} map {} attempt {} reduce {reduce} (memory)",
                o.id.exchange.0, o.id.map, o.id.attempt
            )
        },
        |f| f.path.display().to_string(),
    );
    KrishivSpillFileCorrupt { path, reason }.into()
}
