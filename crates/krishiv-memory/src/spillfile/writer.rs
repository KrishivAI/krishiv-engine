// SPDX-License-Identifier: Apache-2.0
//! Writing a spill file: atomic (temp file, then rename), capped, cancellable.

use std::fs::{self, File};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use krishiv_types::batch::{Batch, Weights};

use super::SpillError;
use super::dir::{ScopeInner, create_new_file};
use super::format::{
    Crc32, FRAME_HEADER_LEN, FrameHeader, KIND_FOOTER, KIND_PLAIN, KIND_WEIGHTED, WEIGHT_COLUMN,
    file_header,
};
use super::reader::SpillFile;
use crate::Reservation;

/// Size of the write buffer in front of the file; part of the scratch.
const BUF_BYTES: usize = 64 << 10;

/// Writes [`Batch`]es to one spill file.
///
/// # Atomicity and durability
/// Batches go to `<n>.spill.tmp`; [`finish`](Self::finish) writes the footer
/// and renames it to `<n>.spill`. A `.spill` file is therefore always
/// complete, and a writer that is dropped, fails or is cancelled leaves
/// nothing behind. Nothing is `fsync`ed, on purpose: spill files are scratch
/// data that is only meaningful to the process that wrote them. After a
/// crash they are deleted at the next start, never read, so durability would
/// cost latency for no benefit. Errors that surface only at `close` are not
/// observable on a local file system, which is why a network one is refused.
///
/// # Accounting
/// The IPC bytes stream through a checksum into a 64 KiB buffer and the
/// file; no frame is ever held in memory. Each write charges the spill cap as
/// the bytes go out; the charge is returned when the finished [`SpillFile`]
/// is dropped, or at once when the writer fails or is dropped unfinished.
/// Arrow encodes one batch at a time, so a write needs scratch memory of up
/// to about twice the batch's size. The writer reserves it from the
/// [`Reservation`] it was given and returns it after the write. Pass a
/// *dedicated scratch* reservation: whatever it already holds is not treated
/// as free, so the reservation that holds the state being spilled would be
/// charged on top of that state.
///
/// Hitting the cap in the middle of a file is an error for the whole query: the
/// file is discarded, not truncated.
///
/// The I/O is blocking: call it from `spawn_blocking`. Each writer holds one
/// file descriptor until it finishes or drops.
#[derive(Debug)]
pub struct SpillWriter {
    scope: Arc<ScopeInner>,
    tmp: PathBuf,
    dest: PathBuf,
    file: Option<BufWriter<File>>,
    schema: SchemaRef,
    weighted_schema: SchemaRef,
    reservation: Reservation,
    charged: u64,
    batches: u64,
    rows: u64,
    committed: bool,
    failed: bool,
}

impl SpillWriter {
    pub(crate) fn create(
        scope: Arc<ScopeInner>,
        tmp: PathBuf,
        dest: PathBuf,
        schema: SchemaRef,
        reservation: Reservation,
    ) -> Result<Self, SpillError> {
        let file = create_new_file(&tmp).map_err(|e| SpillError::io("create", &tmp, e))?;
        let file = BufWriter::with_capacity(BUF_BYTES, file);
        let mut fields: Vec<Arc<Field>> = schema.fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new(WEIGHT_COLUMN, DataType::Int64, false)));
        let weighted_schema =
            Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
        let mut w = Self {
            scope,
            tmp,
            dest,
            file: Some(file),
            schema,
            weighted_schema,
            reservation,
            charged: 0,
            batches: 0,
            rows: 0,
            committed: false,
            failed: false,
        };
        // A failure here drops `w`, which removes the temp file.
        w.put_raw(&file_header())?;
        Ok(w)
    }

    /// Batches written so far.
    pub fn num_batches(&self) -> u64 {
        self.batches
    }

    /// Rows written so far.
    pub fn num_rows(&self) -> u64 {
        self.rows
    }

    /// Bytes written so far, charged to the spill cap.
    pub fn bytes_written(&self) -> u64 {
        self.charged
    }

    /// Appends one batch, weights included.
    ///
    /// A [`SpillError::Memory`] refusal is not fatal: nothing was written, so
    /// the driver can wait for memory and call again with the same batch.
    /// After any other error the writer is finished: its temp file is already
    /// gone and its bytes are already returned to the cap.
    ///
    /// # Errors
    /// [`SpillError::Cancelled`] if the scope was cancelled;
    /// [`SpillError::Memory`] if the scratch memory is refused;
    /// [`SpillError::CapExceeded`] if the frame does not fit under the cap;
    /// [`SpillError::Io`] if the file system fails; [`SpillError::Internal`]
    /// if the batch's schema differs from the writer's.
    pub fn write(&mut self, batch: &Batch) -> Result<(), SpillError> {
        self.guarded(|w| w.scope.check_cancelled())?;
        let decoded = decoded_bytes(batch);
        let scratch = usize::try_from(decoded)
            .unwrap_or(usize::MAX)
            .saturating_mul(2)
            .saturating_add(BUF_BYTES);
        self.reservation.try_grow(scratch)?;
        let result = self.guarded(|w| w.write_inner(batch, decoded));
        self.reservation.shrink(scratch);
        result
    }

    /// Writes the footer and publishes the file under its final name.
    ///
    /// # Errors
    /// As [`write`](Self::write), and [`SpillError::Io`] if the rename fails.
    pub fn finish(mut self) -> Result<SpillFile, SpillError> {
        self.guarded(Self::finish_inner)?;
        // The file now belongs to the `SpillFile`; our `Drop` must not undo it.
        self.committed = true;
        let bytes = std::mem::take(&mut self.charged);
        Ok(SpillFile::new(
            Arc::clone(&self.scope),
            self.dest.clone(),
            Arc::clone(&self.schema),
            bytes,
            self.rows,
            self.batches,
        ))
    }

    fn guarded<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, SpillError>,
    ) -> Result<T, SpillError> {
        if self.failed {
            return Err(Self::used_after_error());
        }
        let result = f(self).map_err(|e| match e {
            // A cancel removes the directory under the writer, which can
            // surface as a file system error; it is a cancel.
            SpillError::Io { .. } if self.scope.check_cancelled().is_err() => SpillError::Cancelled,
            other => other,
        });
        if result.is_err() {
            self.failed = true;
            self.abort();
        }
        result
    }

    fn used_after_error() -> SpillError {
        SpillError::Internal("spill writer used after an earlier error".to_owned())
    }

    fn write_inner(&mut self, batch: &Batch, decoded: u64) -> Result<(), SpillError> {
        self.scope.check_cancelled()?;
        if batch.schema() != self.schema {
            return Err(SpillError::Internal(format!(
                "spill writer for schema {:?} got a batch with schema {:?}",
                self.schema,
                batch.schema()
            )));
        }
        let (record, kind, constant) = match batch.weights() {
            Weights::Const(w) => (batch.data().clone(), KIND_PLAIN, *w),
            Weights::Column(column) => {
                let mut columns: Vec<ArrayRef> = batch.data().columns().to_vec();
                columns.push(Arc::new(column.clone()));
                let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
                let record = RecordBatch::try_new_with_options(
                    Arc::clone(&self.weighted_schema),
                    columns,
                    &options,
                )
                .map_err(|e| encode_error(&self.tmp, &e))?;
                (record, KIND_WEIGHTED, 0)
            }
        };
        let options = self
            .scope
            .manager()
            .write_options()
            .map_err(|e| encode_error(&self.tmp, &e))?;
        self.frame(kind, constant, decoded, |sink| {
            let mut stream = StreamWriter::try_new_with_options(sink, &record.schema(), options)?;
            stream.write(&record)?;
            stream.finish()
        })?;
        self.batches += 1;
        self.rows += batch.num_rows() as u64;
        Ok(())
    }

    fn finish_inner(&mut self) -> Result<(), SpillError> {
        self.scope.check_cancelled()?;
        let (batches, rows) = (self.batches, self.rows);
        self.frame(KIND_FOOTER, 0, 0, |sink| {
            sink.write_all(&batches.to_le_bytes())?;
            Ok(sink.write_all(&rows.to_le_bytes())?)
        })?;
        // Flush and close before renaming so no descriptor outlives the writer.
        if let Some(file) = self.file.take() {
            let file = file
                .into_inner()
                .map_err(|e| SpillError::io("write to", &self.tmp, e.into_error()))?;
            drop(file);
        }
        fs::rename(&self.tmp, &self.dest).map_err(|e| SpillError::io("rename", &self.tmp, e))
    }

    /// Writes one frame: a placeholder header, the payload streamed by `body`
    /// (checksummed and charged to the cap as it goes), then the real header.
    fn frame(
        &mut self,
        kind: u8,
        constant: i64,
        decoded: u64,
        body: impl FnOnce(&mut Sink<'_>) -> Result<(), ArrowError>,
    ) -> Result<(), SpillError> {
        let Some(out) = self.file.as_mut() else {
            return Err(SpillError::Internal("spill writer is closed".to_owned()));
        };
        let start = self.charged;
        let mut sink = Sink::new(out, &self.scope, &self.tmp, &mut self.charged);
        let placeholder = sink.put(&[0u8; FRAME_HEADER_LEN], false);
        let result = placeholder
            .map_err(ArrowError::from)
            .and_then(|()| body(&mut sink));
        let (crc, len, error) = (sink.crc, sink.len, sink.error.take());
        drop(sink);
        if let Err(e) = result {
            return Err(error.unwrap_or_else(|| encode_error(&self.tmp, &e)));
        }
        let header = FrameHeader {
            kind,
            constant,
            len,
            decoded,
            crc: 0,
        }
        .seal(crc);
        let patched = out
            .seek(SeekFrom::Start(start))
            .and_then(|_| out.write_all(&header))
            .and_then(|()| out.seek(SeekFrom::End(0)));
        patched
            .map(drop)
            .map_err(|e| SpillError::io("write to", &self.tmp, e))
    }

    /// Writes bytes outside any frame, charged to the cap.
    fn put_raw(&mut self, bytes: &[u8]) -> Result<(), SpillError> {
        let Some(out) = self.file.as_mut() else {
            return Err(SpillError::Internal("spill writer is closed".to_owned()));
        };
        let mut sink = Sink::new(out, &self.scope, &self.tmp, &mut self.charged);
        sink.put(bytes, false).map_err(|e| {
            sink.error
                .take()
                .unwrap_or_else(|| SpillError::io("write to", &self.tmp, e))
        })
    }

    /// Deletes the temp file and returns the charged bytes. Idempotent.
    fn abort(&mut self) {
        drop(self.file.take());
        let _ = fs::remove_file(&self.tmp);
        self.scope
            .manager()
            .uncharge(std::mem::take(&mut self.charged));
    }
}

impl Drop for SpillWriter {
    fn drop(&mut self) {
        if !self.committed {
            self.abort();
        }
    }
}

fn encode_error(tmp: &Path, e: &ArrowError) -> SpillError {
    SpillError::io("encode a batch for", tmp, io::Error::other(e.to_string()))
}

/// An upper bound of the memory the decoded batch needs (the bytes of the
/// rows written, weights included): what the reader reserves beyond the frame.
fn decoded_bytes(batch: &Batch) -> u64 {
    let columns: usize = batch
        .data()
        .columns()
        .iter()
        .map(|c| {
            c.to_data()
                .get_slice_memory_size()
                .unwrap_or_else(|_| c.get_array_memory_size())
        })
        .sum();
    let weights = match batch.weights() {
        Weights::Const(_) => 0,
        Weights::Column(_) => batch.num_rows() * 8,
    };
    (columns + weights) as u64
}

/// Streams frame payload to the file: charges the cap, checks for cancel,
/// counts and checksums the bytes. Failures are kept in `error` because they
/// pass through Arrow's `io::Error`-shaped interface.
struct Sink<'a> {
    out: &'a mut BufWriter<File>,
    scope: &'a ScopeInner,
    tmp: &'a Path,
    charged: &'a mut u64,
    crc: Crc32,
    len: u64,
    error: Option<SpillError>,
}

impl<'a> Sink<'a> {
    fn new(
        out: &'a mut BufWriter<File>,
        scope: &'a ScopeInner,
        tmp: &'a Path,
        charged: &'a mut u64,
    ) -> Self {
        Self {
            out,
            scope,
            tmp,
            charged,
            crc: Crc32::new(),
            len: 0,
            error: None,
        }
    }

    fn put(&mut self, buf: &[u8], hashed: bool) -> io::Result<()> {
        let n = buf.len() as u64;
        let admitted = self
            .scope
            .check_cancelled()
            .and_then(|()| self.scope.manager().try_charge(n));
        if let Err(e) = admitted {
            self.error = Some(e);
            return Err(io::Error::other("spill write refused"));
        }
        *self.charged += n;
        if let Err(e) = self.out.write_all(buf) {
            let kind = e.kind();
            self.error = Some(SpillError::io("write to", self.tmp, e));
            return Err(kind.into());
        }
        if hashed {
            self.crc.update(buf);
            self.len += n;
        }
        Ok(())
    }
}

impl Write for Sink<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.put(buf, true)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
