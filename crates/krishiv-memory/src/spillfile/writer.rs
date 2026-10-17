// SPDX-License-Identifier: Apache-2.0
//! Writing a spill file: atomic (temp file, then rename), capped, cancellable.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_types::batch::{Batch, Weights};

use super::SpillError;
use super::dir::{ScopeInner, create_new_file};
use super::format::{
    FOOTER_PAYLOAD_LEN, FRAME_HEADER_LEN, KIND_FOOTER, KIND_PLAIN, KIND_WEIGHTED, WEIGHT_COLUMN,
    file_header, seal_frame,
};
use super::reader::SpillFile;
use crate::Reservation;

/// Writes [`Batch`]es to one spill file.
///
/// # Atomicity and durability
/// Batches go to `<n>.spill.tmp`; [`finish`](Self::finish) writes the footer
/// and renames it to `<n>.spill`. A `.spill` file is therefore always
/// complete, and a writer that is dropped, fails or is cancelled leaves
/// nothing behind. Nothing is `fsync`ed, on purpose: spill files are scratch
/// data that is only meaningful to the process that wrote them. After a
/// crash they are deleted at the next start, never read, so durability would
/// cost latency for no benefit.
///
/// # Accounting
/// Every frame is charged to the spill cap before it is written; the charge
/// is returned when the finished [`SpillFile`] is dropped, or at once when
/// the writer fails or is dropped unfinished. Encoding a batch needs scratch
/// memory of about the batch's size; the writer reserves it from the
/// [`Reservation`] it was given, using bytes the reservation already holds
/// first (so an operator that pre-reserves scratch can always spill), and
/// gives it back after each write.
/// The I/O is blocking: call it from `spawn_blocking`.
#[derive(Debug)]
pub struct SpillWriter {
    scope: Arc<ScopeInner>,
    tmp: PathBuf,
    dest: PathBuf,
    file: Option<File>,
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
        w.append(&file_header())?;
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
        let grown = ensure_scratch(&mut self.reservation, scratch_estimate(batch))?;
        let result = self.guarded(|w| w.write_inner(batch));
        self.reservation.shrink(grown);
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

    fn write_inner(&mut self, batch: &Batch) -> Result<(), SpillError> {
        self.scope.check_cancelled()?;
        if batch.schema() != self.schema {
            return Err(SpillError::Internal(format!(
                "spill writer for schema {:?} got a batch with schema {:?}",
                self.schema,
                batch.schema()
            )));
        }
        let frame = self.encode(batch)?;
        self.append(&frame)?;
        self.batches += 1;
        self.rows += batch.num_rows() as u64;
        Ok(())
    }

    fn encode(&self, batch: &Batch) -> Result<Vec<u8>, SpillError> {
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
                .map_err(|e| self.encode_error(&e))?;
                (record, KIND_WEIGHTED, 0)
            }
        };
        let mut frame = vec![0u8; FRAME_HEADER_LEN];
        let options = self
            .scope
            .manager()
            .write_options()
            .map_err(|e| self.encode_error(&e))?;
        let mut stream = StreamWriter::try_new_with_options(&mut frame, &record.schema(), options)
            .map_err(|e| self.encode_error(&e))?;
        stream.write(&record).map_err(|e| self.encode_error(&e))?;
        stream.finish().map_err(|e| self.encode_error(&e))?;
        drop(stream);
        seal_frame(&mut frame, kind, constant);
        Ok(frame)
    }

    fn encode_error(&self, e: &arrow_schema::ArrowError) -> SpillError {
        SpillError::io(
            "encode a batch for",
            &self.tmp,
            io::Error::other(e.to_string()),
        )
    }

    fn finish_inner(&mut self) -> Result<(), SpillError> {
        self.scope.check_cancelled()?;
        let mut frame = vec![0u8; FRAME_HEADER_LEN];
        frame.extend_from_slice(&self.batches.to_le_bytes());
        frame.extend_from_slice(&self.rows.to_le_bytes());
        debug_assert_eq!(frame.len(), FRAME_HEADER_LEN + FOOTER_PAYLOAD_LEN);
        seal_frame(&mut frame, KIND_FOOTER, 0);
        self.append(&frame)?;
        // Close before renaming so no descriptor outlives the writer.
        drop(self.file.take());
        fs::rename(&self.tmp, &self.dest).map_err(|e| SpillError::io("rename", &self.tmp, e))
    }

    /// Charges `bytes` to the cap, then writes them.
    fn append(&mut self, bytes: &[u8]) -> Result<(), SpillError> {
        self.scope.check_cancelled()?;
        let len = bytes.len() as u64;
        self.scope.manager().try_charge(len)?;
        self.charged += len;
        let Some(file) = self.file.as_mut() else {
            return Err(SpillError::Internal("spill writer is closed".to_owned()));
        };
        file.write_all(bytes)
            .map_err(|e| SpillError::io("write to", &self.tmp, e))
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

/// Upper bound of the scratch needed to encode `batch`: its memory size, plus
/// weights, plus IPC framing and padding.
fn scratch_estimate(batch: &Batch) -> usize {
    let mut bytes = batch.data().get_array_memory_size();
    if matches!(batch.weights(), Weights::Column(_)) {
        bytes = bytes.saturating_add(batch.num_rows().saturating_mul(8));
    }
    bytes.saturating_add(bytes / 8).saturating_add(4096)
}

/// Grows `reservation` to at least `needed` bytes; returns the growth, which
/// the caller shrinks again.
fn ensure_scratch(reservation: &mut Reservation, needed: usize) -> Result<usize, SpillError> {
    let have = reservation.size();
    if have >= needed {
        return Ok(0);
    }
    let grow = needed - have;
    reservation.try_grow(grow)?;
    Ok(grow)
}
