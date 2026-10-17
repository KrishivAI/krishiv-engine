// SPDX-License-Identifier: Apache-2.0
//! Finished spill files and reading them back, a batch at a time.

use std::fs::{self, File};
use std::io::{self, Cursor, Read};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{Array, Int64Array};
use arrow_ipc::reader::StreamReader;
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
    ) -> Self {
        Self {
            inner: Arc::new(FileInner {
                scope,
                path,
                schema,
                bytes,
                rows,
                batches,
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
            reservation,
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
/// It is an [`Iterator`] of `Result<Batch, SpillError>` and fused: after an
/// error or the last batch it returns `None`.
///
/// # Memory
/// One frame at a time: the reader reserves the frame's size before reading
/// it, then the decoded batch's size, and keeps the decoded batch's bytes
/// reserved until the next call or until it drops, so the batch it just
/// returned is covered while the caller processes it. A caller that keeps a
/// batch longer must reserve it itself. A frame the pool cannot hold is
/// [`SpillError::Memory`]; the reader never reads it.
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
                "frame of {} bytes at offset {} runs past the end of the file",
                header.len,
                self.pos - FRAME_HEADER_LEN as u64
            )));
        }
        match header.kind {
            KIND_FOOTER => self.read_footer(&header),
            KIND_PLAIN | KIND_WEIGHTED => self.read_data(&header).map(Some),
            other => Err(self.corrupt(format!("unknown frame kind {other}"))),
        }
    }

    /// Reads a frame's payload into memory the reservation covers.
    fn read_payload(&mut self, header: &FrameHeader) -> Result<(Vec<u8>, usize), SpillError> {
        // `header.len` fits the file, so this only fails on a 32-bit target.
        let len = usize::try_from(header.len)
            .map_err(|_| self.corrupt("frame too large for this platform"))?;
        self.reservation.try_grow(len)?;
        self.held += len;
        let mut payload = vec![0u8; len];
        read_exact(&mut self.handle, &mut payload, &self.file, "a frame")?;
        self.pos += header.len;
        if header.checksum(&payload) != header.crc {
            return Err(self.corrupt(format!(
                "checksum mismatch in the frame ending at offset {}",
                self.pos
            )));
        }
        Ok((payload, len))
    }

    fn read_footer(&mut self, header: &FrameHeader) -> Result<Option<Batch>, SpillError> {
        if header.len != FOOTER_PAYLOAD_LEN as u64 {
            return Err(self.corrupt("malformed footer"));
        }
        let (payload, _) = self.read_payload(header)?;
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

    fn read_data(&mut self, header: &FrameHeader) -> Result<Batch, SpillError> {
        let (payload, frame_len) = self.read_payload(header)?;
        let batch = self.decode(header, &payload)?;
        drop(payload);
        // The frame buffer is gone; the decoded batch takes its place.
        let decoded = batch.data().get_array_memory_size()
            + match batch.weights() {
                Weights::Const(_) => 0,
                Weights::Column(c) => c.get_array_memory_size(),
            };
        self.reservation.try_grow(decoded)?;
        self.held += decoded;
        self.reservation.shrink(frame_len);
        self.held -= frame_len;
        self.batches += 1;
        self.rows += batch.num_rows() as u64;
        Ok(batch)
    }

    fn decode(&self, header: &FrameHeader, payload: &[u8]) -> Result<Batch, SpillError> {
        let err = |e: arrow_schema::ArrowError| self.corrupt(format!("undecodable batch: {e}"));
        let mut cursor = Cursor::new(payload);
        let mut stream = StreamReader::try_new(&mut cursor, None).map_err(err)?;
        let record = match stream.next() {
            Some(Ok(record)) => record,
            Some(Err(e)) => return Err(err(e)),
            None => return Err(self.corrupt("frame holds no batch")),
        };
        if stream.next().is_some() {
            return Err(self.corrupt("frame holds more than one batch"));
        }
        drop(stream);
        if cursor.position() != payload.len() as u64 {
            return Err(self.corrupt("undecodable batch: data after the end of the IPC stream"));
        }
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
        if matches!(item, Some(Err(_))) {
            self.done = true;
            self.release();
        }
        item
    }
}
