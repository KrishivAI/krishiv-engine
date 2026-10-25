// SPDX-License-Identifier: Apache-2.0
//! Finished spill files and reading them back, a batch at a time.

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{Array, Int64Array};
use arrow_buffer::Buffer;
use arrow_ipc::reader::StreamDecoder;
use arrow_schema::SchemaRef;
use krishiv_types::batch::{Batch, Weights};

use super::SpillError;
use super::dir::ScopeInner;
use super::format::{
    FILE_HEADER_LEN, FOOTER_PAYLOAD_LEN, FRAME_HEADER_LEN, FrameHeader, KIND_FOOTER, KIND_PLAIN,
    KIND_WEIGHTED, WEIGHT_COLUMN, check_file_header,
};
use crate::Reservation;

#[derive(Debug)]
struct FileInner {
    scope: Arc<ScopeInner>,
    path: PathBuf,
    schema: SchemaRef,
    bytes: u64,
    rows: u64,
    batches: u64,
    /// The most a reader reserves at once: a frame plus its decoded bound.
    max_read: usize,
}

impl Drop for FileInner {
    fn drop(&mut self) {
        // A failure to delete is left to the scope and session cleanup.
        let _ = fs::remove_file(&self.path);
        self.scope.manager().uncharge(self.bytes);
    }
}

/// A finished spill file. Dropping it deletes the file and returns its bytes
/// to the spill cap; a [`SpillReader`] made from it keeps it alive until the
/// reader drops too.
#[derive(Debug)]
pub struct SpillFile {
    inner: Arc<FileInner>,
}

impl SpillFile {
    pub(crate) fn new(
        scope: Arc<ScopeInner>,
        path: PathBuf,
        schema: SchemaRef,
        bytes: u64,
        rows: u64,
        batches: u64,
        max_read: usize,
    ) -> Self {
        Self {
            inner: Arc::new(FileInner {
                scope,
                path,
                schema,
                bytes,
                rows,
                batches,
                max_read,
            }),
        }
    }

    /// The schema of the batches in the file.
    pub fn schema(&self) -> &SchemaRef {
        &self.inner.schema
    }

    /// Bytes the file occupies on disk.
    pub fn bytes(&self) -> u64 {
        self.inner.bytes
    }

    /// Rows in the file.
    pub fn num_rows(&self) -> u64 {
        self.inner.rows
    }

    /// Batches in the file.
    pub fn num_batches(&self) -> u64 {
        self.inner.batches
    }

    /// The most a [`SpillReader`] of this file reserves at once (its largest
    /// frame plus that frame's decoded bound): reserve this much before
    /// reading the file back and a read can never be refused for memory.
    pub fn max_read_reservation(&self) -> usize {
        self.inner.max_read
    }

    /// The file's path (for diagnostics; do not move or open it).
    pub fn path(&self) -> &std::path::Path {
        &self.inner.path
    }

    /// Opens the file for streaming read-back. The reader accounts its
    /// working memory in `reservation` (see [`SpillReader`]). Any number of
    /// readers may be open on one file.
    ///
    /// # Errors
    /// [`SpillError::Cancelled`] after the scope was cancelled;
    /// [`SpillError::Io`] if the file cannot be opened;
    /// [`SpillError::Corrupt`] if its header or size is wrong.
    pub fn reader(&self, reservation: Reservation) -> Result<SpillReader, SpillError> {
        self.inner.scope.check_cancelled()?;
        let path = &self.inner.path;
        let mut file = File::open(path).map_err(|e| SpillError::io("open", path, e))?;
        let len = file
            .metadata()
            .map_err(|e| SpillError::io("stat", path, e))?
            .len();
        if len != self.inner.bytes {
            return Err(SpillError::corrupt(
                path,
                format!("file is {len} bytes, {} were written", self.inner.bytes),
            ));
        }
        let mut header = [0u8; FILE_HEADER_LEN];
        read_exact(&mut file, &mut header, &self.inner, "the file header")?;
        check_file_header(&header).map_err(|reason| SpillError::corrupt(path, reason))?;
        Ok(SpillReader {
            file: Arc::clone(&self.inner),
            handle: file,
            len,
            pos: FILE_HEADER_LEN as u64,
            batches: 0,
            rows: 0,
            // Reading spilled data back has no spill to fall back on: the
            // resident-set guard never refuses its working memory.
            reservation: reservation.unguarded(),
            held: 0,
            done: false,
        })
    }
}

fn read_exact(
    handle: &mut File,
    buf: &mut [u8],
    file: &FileInner,
    what: &str,
) -> Result<(), SpillError> {
    handle.read_exact(buf).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            SpillError::corrupt(&file.path, format!("truncated in {what}"))
        } else {
            SpillError::io("read from", &file.path, e)
        }
    })
}

/// Streams the batches of a [`SpillFile`] in the order they were written.
///
/// It is an [`Iterator`] of `Result<Batch, SpillError>` and fused: after the
/// last batch or an error other than [`SpillError::Memory`] it returns `None`.
/// Each open reader holds one file descriptor (a k-way merge over many runs
/// needs a descriptor budget).
///
/// # Memory
/// One frame at a time. Before it reads anything the reader reserves the
/// frame's size plus the `decoded` bound recorded when the frame was written,
/// and it keeps that reserved until the next call or until it drops, so the
/// batch it just returned is covered while the caller processes it (at most
/// about twice the batch's real size, since an uncompressed batch shares the
/// frame buffer). A caller that keeps a batch longer must reserve it itself.
/// A frame the pool cannot hold is a retryable [`SpillError::Memory`].
#[derive(Debug)]
pub struct SpillReader {
    file: Arc<FileInner>,
    handle: File,
    len: u64,
    pos: u64,
    batches: u64,
    rows: u64,
    reservation: Reservation,
    held: usize,
    done: bool,
}

impl SpillReader {
    fn corrupt(&self, reason: impl Into<String>) -> SpillError {
        SpillError::corrupt(&self.file.path, reason)
    }

    fn release(&mut self) {
        self.reservation.shrink(std::mem::take(&mut self.held));
    }

    fn read_batch(&mut self) -> Result<Option<Batch>, SpillError> {
        self.release();
        if self.done {
            return Ok(None);
        }
        self.file.scope.check_cancelled()?;
        let start = self.pos;
        let mut raw = [0u8; FRAME_HEADER_LEN];
        let remaining = self.len - self.pos;
        if remaining == 0 {
            return Err(self.corrupt("truncated: no footer"));
        }
        if remaining < FRAME_HEADER_LEN as u64 {
            return Err(self.corrupt("truncated in a frame header"));
        }
        read_exact(&mut self.handle, &mut raw, &self.file, "a frame header")?;
        self.pos += FRAME_HEADER_LEN as u64;
        let header = FrameHeader::parse(&raw);
        if header.len > self.len - self.pos {
            return Err(self.corrupt(format!(
                "frame of {} bytes at offset {start} runs past the end of the file",
                header.len
            )));
        }
        match header.kind {
            KIND_FOOTER => self.read_footer(&header, start),
            KIND_PLAIN | KIND_WEIGHTED => self.read_data(&header, start).map(Some),
            other => Err(self.corrupt(format!("unknown frame kind {other}"))),
        }
    }

    /// Reserves `len + decoded` bytes, then reads and verifies the payload.
    /// A refusal by the pool leaves the reader at the start of the frame, so
    /// the call can be repeated once memory is free.
    fn read_payload(&mut self, header: &FrameHeader, start: u64) -> Result<Vec<u8>, SpillError> {
        // `header.len` fits the file, so this only fails on a 32-bit target.
        let len = usize::try_from(header.len)
            .map_err(|_| self.corrupt("frame too large for this platform"))?;
        // Compression ratios stay far below this; a larger claim is damage.
        if header.decoded > header.len.saturating_mul(1 << 12).saturating_add(1 << 20) {
            return Err(self.corrupt(format!("implausible decoded size {}", header.decoded)));
        }
        let need = len.saturating_add(usize::try_from(header.decoded).unwrap_or(usize::MAX));
        if let Err(e) = self.reservation.try_grow(need) {
            self.handle
                .seek(SeekFrom::Start(start))
                .map_err(|e| SpillError::io("seek in", &self.file.path, e))?;
            self.pos = start;
            return Err(e.into());
        }
        self.held += need;
        let mut payload = vec![0u8; len];
        read_exact(&mut self.handle, &mut payload, &self.file, "a frame")?;
        self.pos += header.len;
        if header.checksum(&payload) != header.crc {
            return Err(self.corrupt(format!(
                "checksum mismatch in the frame ending at offset {}",
                self.pos
            )));
        }
        Ok(payload)
    }

    fn read_footer(
        &mut self,
        header: &FrameHeader,
        start: u64,
    ) -> Result<Option<Batch>, SpillError> {
        if header.len != FOOTER_PAYLOAD_LEN as u64 {
            return Err(self.corrupt("malformed footer"));
        }
        let payload = self.read_payload(header, start)?;
        let mut b = [0u8; 8];
        let mut r = [0u8; 8];
        b.copy_from_slice(payload.get(..8).unwrap_or_default());
        r.copy_from_slice(payload.get(8..16).unwrap_or_default());
        let (batches, rows) = (u64::from_le_bytes(b), u64::from_le_bytes(r));
        if (batches, rows) != (self.batches, self.rows)
            || (batches, rows) != (self.file.batches, self.file.rows)
        {
            return Err(self.corrupt(format!(
                "footer says {batches} batches and {rows} rows, read {} and {}, wrote {} and {}",
                self.batches, self.rows, self.file.batches, self.file.rows
            )));
        }
        if self.pos != self.len {
            return Err(self.corrupt("data after the footer"));
        }
        self.done = true;
        Ok(None)
    }

    fn read_data(&mut self, header: &FrameHeader, start: u64) -> Result<Batch, SpillError> {
        let payload = self.read_payload(header, start)?;
        let batch = self.decode(header, payload)?;
        self.batches += 1;
        self.rows += batch.num_rows() as u64;
        Ok(batch)
    }

    /// Decodes without copying: uncompressed buffers are slices of `payload`.
    fn decode(&self, header: &FrameHeader, payload: Vec<u8>) -> Result<Batch, SpillError> {
        let err = |e: arrow_schema::ArrowError| self.corrupt(format!("undecodable batch: {e}"));
        let mut buf = Buffer::from_vec(payload);
        let mut decoder = StreamDecoder::new();
        let mut decoded = None;
        while !buf.is_empty() {
            if let Some(batch) = decoder.decode(&mut buf).map_err(err)?
                && decoded.replace(batch).is_some()
            {
                return Err(self.corrupt("frame holds more than one batch"));
            }
        }
        decoder.finish().map_err(err)?;
        let record = decoded.ok_or_else(|| self.corrupt("frame holds no batch"))?;
        let (data, weights) = if header.kind == KIND_WEIGHTED {
            let n = record
                .num_columns()
                .checked_sub(1)
                .ok_or_else(|| self.corrupt("weighted frame without a weights column"))?;
            let schema = record.schema();
            let column = record
                .column(n)
                .as_any()
                .downcast_ref::<Int64Array>()
                .filter(|_| schema.field(n).name() == WEIGHT_COLUMN)
                .ok_or_else(|| self.corrupt("last column is not the Int64 weights column"))?
                .clone();
            let keep: Vec<usize> = (0..n).collect();
            (record.project(&keep).map_err(err)?, Weights::Column(column))
        } else {
            (record, Weights::Const(header.constant))
        };
        if data.schema() != self.file.schema {
            return Err(self.corrupt("batch schema differs from the file's"));
        }
        Batch::with_weights(data, weights).map_err(|e| self.corrupt(e.to_string()))
    }
}

impl Iterator for SpillReader {
    type Item = Result<Batch, SpillError>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.read_batch().transpose();
        // A refusal by the memory pool is not final: the reader is back at the
        // start of the frame and the call can be repeated.
        if matches!(item, Some(Err(ref e)) if !matches!(e, SpillError::Memory(_))) {
            self.done = true;
            self.release();
        }
        item
    }
}
