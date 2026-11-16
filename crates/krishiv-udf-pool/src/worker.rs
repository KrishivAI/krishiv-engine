// SPDX-License-Identifier: Apache-2.0
//! One worker process: its start, its handshake, and the `udf` and `batch` exchanges of wire v1.
//!
//! A [`Worker`] is used by one call at a time (the pool hands it out exclusively). Every method blocks on the
//! worker, so it runs only on the pool's own threads, never on a compute worker.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Read};
use std::os::fd::{AsFd, OwnedFd};
use std::process::{Child, ExitStatus};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_buffer::Buffer;
use arrow_ipc::reader::StreamDecoder;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::DataType;
use krishiv_common::error::{
    InternalError, KrishivError, KrishivPythonWorkerExited, OperationCanceled, PythonException,
};
use krishiv_udf_sandbox::{EXIT_SANDBOX_FAILED, SandboxError};
use rustix::process::{Pid, PidfdFlags, Signal};
use serde_json::{Value, json};

use crate::pool::CallWatch;
use crate::wire::{self, Message};

/// How long a new worker may take to import its packages and greet the server.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

/// The most of a worker's standard error kept for an error message.
const STDERR_TAIL: usize = 8 * 1024;

/// A started worker process, before its handshake.
pub(crate) struct Started {
    pub(crate) child: Child,
    pub(crate) control: OwnedFd,
    pub(crate) private_dir: tempfile::TempDir,
    /// Started through the sandbox trampoline, whose reserved exit status means it could not enter the sandbox.
    pub(crate) sandboxed: bool,
}

/// A worker that greeted the server.
pub(crate) struct Worker {
    child: Child,
    pid: Pid,
    /// Signals go through this pidfd, never the pid: a pidfd never names a reused pid, so a late kill (a
    /// cancel racing the worker's own exit) cannot reach another process.
    pidfd: Arc<OwnedFd>,
    control: OwnedFd,
    /// Removed when the worker is dropped (after the process is gone).
    _private_dir: tempfile::TempDir,
    /// The payload ids this worker has loaded.
    loaded: HashSet<u64>,
    stderr: Arc<Mutex<String>>,
    /// Disconnects when the stderr thread has read to end of file.
    stderr_done: Option<std::sync::mpsc::Receiver<()>>,
    /// False once an exchange failed in a way that leaves the worker unusable.
    healthy: bool,
    /// Started through the trampoline (`Started::sandboxed`).
    sandboxed: bool,
    /// True once the worker greeted the server: the trampoline is gone, so an exit status is the worker's own.
    greeted: bool,
}

impl std::fmt::Debug for Worker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Worker")
            .field("pid", &self.pid)
            .field("loaded", &self.loaded.len())
            .field("healthy", &self.healthy)
            .finish_non_exhaustive()
    }
}

/// What a worker is asked to load: Spark's `command` bytes and how to call them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonPayload {
    /// Spark's `PythonUDF.command`: cloudpickled `(func, return_type)`.
    pub command: Arc<[u8]>,
    /// Spark's eval type (100, 101, 200 or 250 for the scalar kinds).
    pub eval_type: i32,
    /// The client's Python `major.minor`.
    pub python_ver: String,
}

impl Worker {
    /// Waits for the worker's greeting and checks its wire version. `watch` gets the worker's pidfd before
    /// the wait, so a call cancelled meanwhile kills the starting worker (review 2026-11-13 n3): the wait
    /// then ends at once, and the worker is reaped when this returns.
    pub(crate) fn handshake(started: Started, watch: &dyn CallWatch) -> Result<Self, KrishivError> {
        let Started {
            mut child,
            control,
            private_dir,
            sandboxed,
        } = started;
        let pid = Pid::from_child(&child);
        // The child is not reaped yet, so `pid` is still its own.
        let pidfd = match rustix::process::pidfd_open(pid, PidfdFlags::empty()) {
            Ok(fd) => Arc::new(fd),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(InternalError {
                    message: format!("cannot open a pidfd for a Python UDF worker: {e}"),
                }
                .into());
            }
        };
        let stderr = Arc::new(Mutex::new(String::new()));
        let stderr_done = child
            .stderr
            .take()
            .and_then(|pipe| drain_stderr(pipe, pid, Arc::clone(&stderr)));
        let mut worker = Worker {
            child,
            pid,
            pidfd,
            control,
            _private_dir: private_dir,
            loaded: HashSet::new(),
            stderr,
            stderr_done,
            healthy: false,
            sandboxed,
            greeted: false,
        };
        if !watch.starting(&worker.pidfd) {
            // Dropping the worker kills and reaps it.
            return Err(OperationCanceled {}.into());
        }
        let greeting = rustix::net::sockopt::set_socket_timeout(
            worker.control.as_fd(),
            rustix::net::sockopt::Timeout::Recv,
            Some(STARTUP_TIMEOUT),
        )
        .map_err(std::io::Error::from)
        .and_then(|()| wire::recv(worker.control.as_fd()));
        let msg = match greeting {
            Ok(Some(msg)) => msg,
            Ok(None) => return Err(worker.exited("before it started")),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                return Err(worker.exited(&format!(
                    "it did not start within {} s",
                    STARTUP_TIMEOUT.as_secs()
                )));
            }
            Err(e) => return Err(worker.protocol(&e.to_string())),
        };
        match msg.kind() {
            "hello" if msg.u64_field("wire") == Some(wire::WIRE_VERSION) => {}
            "hello" => {
                return Err(worker.protocol(&format!(
                    "it speaks wire version {:?}, the server {}",
                    msg.body.get("wire"),
                    wire::WIRE_VERSION
                )));
            }
            "error" => return Err(worker.python_error(&msg)),
            other => return Err(worker.protocol(&format!("unexpected greeting {other:?}"))),
        }
        rustix::net::sockopt::set_socket_timeout(
            worker.control.as_fd(),
            rustix::net::sockopt::Timeout::Recv,
            None,
        )
        .map_err(|e| worker.server_failure(&e.to_string()))?;
        tracing::debug!(
            pid = pid.as_raw_nonzero().get(),
            python = msg.str_field("python"),
            pyspark = msg.str_field("pyspark"),
            "Python UDF worker started"
        );
        worker.healthy = true;
        worker.greeted = true;
        Ok(worker)
    }

    /// The worker's pidfd, for a cancel to kill it from another thread.
    pub(crate) fn pidfd(&self) -> Arc<OwnedFd> {
        Arc::clone(&self.pidfd)
    }

    /// Whether the worker can take another call.
    pub(crate) fn healthy(&self) -> bool {
        self.healthy
    }

    /// Loads payload `id` unless this worker already has it.
    pub(crate) fn ensure_loaded(
        &mut self,
        id: u64,
        payload: &PythonPayload,
    ) -> Result<(), KrishivError> {
        if self.loaded.contains(&id) {
            return Ok(());
        }
        let len = payload.command.len();
        let file = wire::sealed_memfd("krishiv-udf-command", &payload.command)
            .map_err(|e| self.server_failure(&format!("cannot pass the UDF: {e}")))?;
        let header = json!({
            "t": "udf",
            "id": id,
            "eval_type": payload.eval_type,
            "python_ver": payload.python_ver,
            "len": len,
        });
        let reply = self.exchange(&header, &file)?;
        match reply.kind() {
            "ready" => {
                self.loaded.insert(id);
                Ok(())
            }
            "error" => Err(self.python_error(&reply)),
            other => Err(self.protocol(&format!("unexpected answer {other:?} to a UDF"))),
        }
    }

    /// Runs loaded payload `id` over `args` and returns its result column, checked against `return_type` and
    /// the row count. `max_result` caps the result's size in bytes.
    pub(crate) fn run_batch(
        &mut self,
        id: u64,
        args: &RecordBatch,
        conf: &BTreeMap<String, String>,
        return_type: &DataType,
        max_result: u64,
    ) -> Result<ArrayRef, KrishivError> {
        let (file, len) = write_input(args)
            .map_err(|e| self.server_failure(&format!("cannot pass a batch: {e}")))?;
        let mut conf = conf.clone();
        conf.insert(
            "krishiv.python.udf.rows".to_owned(),
            args.num_rows().to_string(),
        );
        let header = json!({"t": "batch", "udf": id, "len": len, "conf": conf});
        let mut reply = self.exchange(&header, &file)?;
        match reply.kind() {
            "result" => {}
            // A Python error leaves the worker usable: it answered and waits for the next message.
            "error" => return Err(self.python_error(&reply)),
            other => return Err(self.protocol(&format!("unexpected answer {other:?} to a batch"))),
        }
        let (Some(len), Some(fd), true) = (
            reply.u64_field("len"),
            reply.fds.pop(),
            reply.fds.is_empty(),
        ) else {
            return Err(self.protocol("a result without exactly one descriptor and a length"));
        };
        let bytes =
            wire::read_untrusted(fd, len, max_result).map_err(|e| self.protocol(&e.to_string()))?;
        decode_result(bytes, return_type, args.num_rows()).inspect_err(|_| self.healthy = false)
    }

    /// Sends `header` with `file` attached and waits for the answer.
    fn exchange(&mut self, header: &Value, file: &File) -> Result<Message, KrishivError> {
        if let Err(e) = wire::send(self.control.as_fd(), header, &[file.as_fd()]) {
            return Err(if e.kind() == std::io::ErrorKind::BrokenPipe {
                self.exited("while it was sent work")
            } else {
                self.server_failure(&e.to_string())
            });
        }
        match wire::recv(self.control.as_fd()) {
            Ok(Some(msg)) => Ok(msg),
            Ok(None) => Err(self.exited("while it ran a UDF")),
            Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset => {
                Err(self.exited("while it ran a UDF"))
            }
            Err(e) => Err(self.protocol(&e.to_string())),
        }
    }

    /// `PYTHON_EXCEPTION` from an `error` message: a Python error the worker reported.
    fn python_error(&mut self, msg: &Message) -> KrishivError {
        if msg.str_field("kind") != "python" {
            return self.protocol(&format!(
                "the worker reported: {}",
                msg.str_field("message")
            ));
        }
        PythonException {
            msg: "An exception was thrown from the Python worker".to_owned(),
            traceback: msg.str_field("message").to_owned(),
        }
        .into()
    }

    /// The worker broke the protocol (code in it wrote to its control socket, or sent a result it then
    /// changed): `KRISHIV_PYTHON_WORKER_EXITED`, since it is killed and not reused. Not `INTERNAL_ERROR`, which
    /// would read as an engine bug when user code caused it.
    fn protocol(&mut self, what: &str) -> KrishivError {
        self.healthy = false;
        self.kill();
        KrishivPythonWorkerExited {
            reason: format!("it broke the worker protocol: {what}"),
        }
        .into()
    }

    /// The server could not talk to the worker (a memfd or socket call failed on the server's side):
    /// `INTERNAL_ERROR`; the worker is killed and not reused.
    fn server_failure(&mut self, what: &str) -> KrishivError {
        self.healthy = false;
        self.kill();
        InternalError {
            message: format!("Python UDF worker {}: {what}", self.pid.as_raw_nonzero()),
        }
        .into()
    }

    /// The worker is gone (end of stream on its control socket): the error says how it ended.
    fn exited(&mut self, when: &str) -> KrishivError {
        self.healthy = false;
        // It may be closing its socket and still running: make sure it is gone before it is waited for.
        self.kill();
        let status = self.child.wait();
        // The process is gone, but its last output may still be in the pipe: let the stderr thread read it
        // to end of file first (bounded, in case something else still holds the pipe).
        if let Some(done) = self.stderr_done.take() {
            let _ = done.recv_timeout(Duration::from_secs(1));
        }
        let tail = self.stderr_tail();
        // Only the trampoline exits with the reserved status, and only before the worker greets the server:
        // after that (or without the sandbox) the same code is the UDF's own `os._exit(125)`.
        if self.sandboxed
            && !self.greeted
            && let Ok(status) = status
            && status.code() == Some(i32::from(EXIT_SANDBOX_FAILED))
        {
            return KrishivError::from(&SandboxError::Unavailable {
                reason: if tail.is_empty() {
                    "the trampoline could not enter the sandbox".to_owned()
                } else {
                    tail
                },
            });
        }
        let how = status.map_or_else(|e| format!("cannot wait for it: {e}"), describe);
        let mut reason = format!("it ended {when} ({how})");
        if !tail.is_empty() {
            reason.push_str("; its last output: ");
            reason.push_str(&tail);
        }
        KrishivPythonWorkerExited { reason }.into()
    }

    fn stderr_tail(&self) -> String {
        self.stderr
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .trim()
            .to_owned()
    }

    /// Sends `SIGKILL` through the pidfd (the worker may already be gone, or reaped: then nothing happens).
    pub(crate) fn kill(&self) {
        kill_via(&self.pidfd);
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Closing the control socket ends an idle worker; a busy one is killed. Either way it is reaped here.
        // Signals go through the pidfd, so one sent after this (a late cancel) reaches no other process.
        self.kill();
        let _ = self.child.wait();
    }
}

/// Sends `SIGKILL` to the process of `pidfd`; after that process exited (or was reaped) the call fails
/// harmlessly, since a pidfd never names a reused pid.
pub(crate) fn kill_via(pidfd: &OwnedFd) {
    let _ = rustix::process::pidfd_send_signal(pidfd, Signal::KILL);
}

fn describe(status: ExitStatus) -> String {
    use std::os::unix::process::ExitStatusExt;
    match (status.code(), status.signal()) {
        (Some(code), _) => format!("exit code {code}"),
        (None, Some(signal)) => format!("signal {signal}"),
        (None, None) => status.to_string(),
    }
}

/// Logs a worker's standard error and keeps its last [`STDERR_TAIL`] bytes, on a thread of its own that ends
/// when the worker does. The receiver disconnects when the thread has read to end of file.
fn drain_stderr(
    mut pipe: impl Read + Send + 'static,
    pid: Pid,
    tail: Arc<Mutex<String>>,
) -> Option<std::sync::mpsc::Receiver<()>> {
    let (done, finished) = std::sync::mpsc::channel::<()>();
    let spawned = std::thread::Builder::new()
        .name("krishiv-udf-stderr".to_owned())
        .spawn(move || {
            let _done = done;
            let mut buf = [0_u8; 4096];
            loop {
                let n = match pipe.read(&mut buf) {
                    Ok(0) | Err(_) => return,
                    Ok(n) => n,
                };
                let text = String::from_utf8_lossy(buf.get(..n).unwrap_or_default());
                tracing::debug!(
                    pid = pid.as_raw_nonzero().get(),
                    "Python UDF worker: {}",
                    text.trim_end()
                );
                let mut kept = tail.lock().unwrap_or_else(PoisonError::into_inner);
                kept.push_str(&text);
                if kept.len() > STDERR_TAIL {
                    let mut cut = kept.len() - STDERR_TAIL;
                    while !kept.is_char_boundary(cut) {
                        cut += 1;
                    }
                    kept.drain(..cut);
                }
            }
        });
    match spawned {
        Ok(_) => Some(finished),
        Err(e) => {
            tracing::warn!("cannot start the thread that reads a Python UDF worker's output: {e}");
            None
        }
    }
}

/// `args` as an Arrow IPC stream in a sealed memfd, and its length.
fn write_input(args: &RecordBatch) -> std::io::Result<(File, u64)> {
    let file = wire::memfd("krishiv-udf-input")?;
    {
        let mut writer = StreamWriter::try_new(BufWriter::new(&file), &args.schema())
            .map_err(std::io::Error::other)?;
        writer.write(args).map_err(std::io::Error::other)?;
        writer.finish().map_err(std::io::Error::other)?;
        let mut inner = writer.into_inner().map_err(std::io::Error::other)?;
        std::io::Write::flush(&mut inner)?;
    }
    let len = file.metadata()?.len();
    wire::seal(&file)?;
    Ok((file, len))
}

/// Decodes a worker's result stream as the pool does: the framing checked first ([`check_frames`]), then
/// arrow-ipc's validating decoder, then one column of `return_type`, one batch and `rows` rows. Every failure
/// is `PYTHON_EXCEPTION` naming what was wrong. Public for the `udf_result_ipc` fuzz target only, so hidden
/// from the documentation; not a supported interface.
///
/// The decoder slices the column's buffers out of `bytes` rather than copying them (review 2026-11-13 n1):
/// the pool reads a result into one 64-byte-aligned buffer ([`wire::read_untrusted`]), so it is copied once,
/// from the memfd. A buffer arrow-ipc finds misaligned (only from a caller that passes unaligned bytes) is
/// copied into an aligned one.
///
/// # Errors
/// Anything but a well-formed, uncompressed stream of that shape.
#[doc(hidden)]
pub fn decode_result(
    bytes: impl Into<Buffer>,
    return_type: &DataType,
    rows: usize,
) -> Result<ArrayRef, KrishivError> {
    let invalid = |detail: String| -> KrishivError {
        PythonException {
            msg: "The Python worker returned an invalid result".to_owned(),
            traceback: detail,
        }
        .into()
    };
    let mut bytes: Buffer = bytes.into();
    let first = check_frames(bytes.as_slice()).map_err(invalid)?;
    // The schema is checked from the first message's header before any batch is decoded (review 2026-11-14 m5).
    if let Some(header) = bytes.as_slice().get(8..first) {
        let message = arrow_ipc::root_as_message(header)
            .map_err(|e| invalid(format!("an invalid message header: {e}")))?;
        let Some(schema) = message.header_as_schema() else {
            return Err(invalid(
                "the stream does not start with a schema".to_owned(),
            ));
        };
        let schema = arrow_ipc::convert::try_fb_to_schema(schema)
            .map_err(|e| invalid(format!("an unreadable schema: {e}")))?;
        let [field] = schema.fields().iter().collect::<Vec<_>>()[..] else {
            return Err(invalid(format!(
                "{} columns, expected one",
                schema.fields().len()
            )));
        };
        if field.data_type() != return_type {
            return Err(invalid(format!(
                "a column of type {}, expected {return_type}",
                field.data_type()
            )));
        }
    }
    let mut decoder = StreamDecoder::new();
    let mut batch = None;
    // `check_frames` saw a whole stream, so the decoder sees every message whole and keeps body slices.
    while !bytes.is_empty() {
        let decoded = decoder
            .decode(&mut bytes)
            .map_err(|e| invalid(format!("unreadable Arrow stream: {e}")))?;
        if let Some(decoded) = decoded
            && batch.replace(decoded).is_some()
        {
            return Err(invalid("more than one batch".to_owned()));
        }
    }
    decoder
        .finish()
        .map_err(|e| invalid(format!("unreadable Arrow stream: {e}")))?;
    let Some(batch) = batch else {
        return Err(invalid("no batch".to_owned()));
    };
    if batch.num_rows() != rows {
        return Err(invalid(format!(
            "{} rows, expected {rows}",
            batch.num_rows()
        )));
    }
    Ok(Arc::clone(batch.column(0)))
}

/// Walks the stream's messages before arrow-ipc decodes them: each has the continuation marker and a
/// verified flatbuffer header (`root_as_message`), is a schema, record batch or dictionary batch, has a body
/// inside the stream, and, for a batch, no compression and buffers inside its body.
///
/// Compression is refused because arrow-ipc 60 allocates a compressed buffer's declared uncompressed length
/// before reading it: a worker could declare 2^45 bytes in a 16 KiB stream and abort the server (review
/// 2026-11-13 M1). The server never asks for a compressed result, and the worker writes without compression.
///
/// Returns where the first message's header ends (it starts at byte 8), so the schema can be checked before
/// any batch is decoded; 0 for an empty stream.
fn check_frames(bytes: &[u8]) -> Result<usize, String> {
    const CONTINUATION: u32 = 0xFFFF_FFFF;
    let word = |at: usize| -> Option<u32> {
        Some(u32::from_le_bytes(
            bytes.get(at..at.checked_add(4)?)?.try_into().ok()?,
        ))
    };
    let mut at = 0_usize;
    let mut first = None;
    while at < bytes.len() {
        if word(at) != Some(CONTINUATION) {
            return Err(format!("no continuation marker at byte {at}"));
        }
        let len = word(at + 4).ok_or_else(|| format!("a truncated message at byte {at}"))?;
        at += 8;
        if len == 0 {
            // End of stream: nothing may follow.
            return if at == bytes.len() {
                Ok(first.unwrap_or(0))
            } else {
                Err(format!(
                    "{} bytes after the end of the stream",
                    bytes.len() - at
                ))
            };
        }
        let len = usize::try_from(len).map_err(|e| e.to_string())?;
        let meta_end = at
            .checked_add(len)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| format!("a message header past the end at byte {at}"))?;
        let meta = bytes.get(at..meta_end).unwrap_or_default();
        let message = arrow_ipc::root_as_message(meta)
            .map_err(|e| format!("an invalid message header at byte {at}: {e}"))?;
        let body_len = usize::try_from(message.bodyLength())
            .map_err(|_| format!("a negative body length at byte {at}"))?;
        let body_end = meta_end
            .checked_add(body_len)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| format!("a message body past the end at byte {at}"))?;
        let batch = match message.header_type() {
            arrow_ipc::MessageHeader::Schema => None,
            arrow_ipc::MessageHeader::RecordBatch => message.header_as_record_batch(),
            arrow_ipc::MessageHeader::DictionaryBatch => {
                message.header_as_dictionary_batch().and_then(|d| d.data())
            }
            other => return Err(format!("an unexpected {other:?} message at byte {at}")),
        };
        if let Some(batch) = batch {
            if batch.compression().is_some() {
                return Err("a compressed batch (results must be uncompressed)".to_owned());
            }
            for buffer in batch.buffers().into_iter().flatten() {
                let end = u64::try_from(buffer.offset())
                    .ok()
                    .zip(u64::try_from(buffer.length()).ok())
                    .and_then(|(o, l)| o.checked_add(l));
                if end.is_none_or(|end| end > body_len as u64) {
                    return Err(format!("a buffer outside its message body at byte {at}"));
                }
            }
        }
        first.get_or_insert(meta_end);
        at = body_end;
    }
    Ok(first.unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, StringViewArray};
    use arrow_schema::{Field, Schema};
    use krishiv_common::error::ErrorClass;

    fn stream(batches: &[RecordBatch]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut w = StreamWriter::try_new(&mut out, &batches[0].schema()).expect("writer");
        for b in batches {
            w.write(b).expect("write");
        }
        w.finish().expect("finish");
        drop(w);
        out
    }

    fn longs(values: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("_0", DataType::Int64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values))]).expect("batch")
    }

    #[test]
    fn a_valid_result_decodes() {
        let col = decode_result(stream(&[longs(vec![1, 2])]), &DataType::Int64, 2).expect("ok");
        assert_eq!(col.len(), 2);
    }

    #[test]
    fn a_result_is_decoded_without_a_copy() {
        // Review 2026-11-13 n1: the column's values are a slice of the buffer the result was read into.
        let bytes = stream(&[longs((0..1024).collect())]);
        let mut aligned = arrow_buffer::MutableBuffer::new(bytes.len());
        aligned.extend_from_slice(&bytes);
        let buffer = Buffer::from(aligned);
        let start = buffer.as_ptr().addr();
        let range = start..start + buffer.len();
        let col = decode_result(buffer, &DataType::Int64, 1024).expect("ok");
        let data = col.to_data();
        let values = data.buffers()[0].as_ptr().addr();
        assert!(range.contains(&values), "the values were copied");
    }

    #[test]
    fn an_empty_stream_has_no_batch() {
        let schema_only = {
            let batch = longs(vec![]);
            let mut out = Vec::new();
            let mut w = StreamWriter::try_new(&mut out, &batch.schema()).expect("writer");
            w.finish().expect("finish");
            drop(w);
            out
        };
        let err = decode_result(schema_only, &DataType::Int64, 0).expect_err("no batch");
        assert!(err.to_string().contains("no batch"), "{err}");
        let err = decode_result(Vec::<u8>::new(), &DataType::Int64, 0).expect_err("nothing");
        assert!(err.to_string().contains("no batch"), "{err}");
    }

    #[test]
    fn wrong_results_are_refused() {
        let class = |r: Result<ArrayRef, KrishivError>| r.expect_err("refused").error_class();
        assert_eq!(
            class(decode_result(
                stream(&[longs(vec![1])]),
                &DataType::Utf8View,
                1
            )),
            ErrorClass::PythonException
        );
        assert!(decode_result(stream(&[longs(vec![1])]), &DataType::Int64, 2).is_err());
        assert!(
            decode_result(
                stream(&[longs(vec![1]), longs(vec![2])]),
                &DataType::Int64,
                1
            )
            .is_err()
        );
        assert!(decode_result(b"garbage".to_vec(), &DataType::Int64, 1).is_err());
        let two = RecordBatch::try_from_iter([
            ("a", Arc::new(Int64Array::from(vec![1])) as ArrayRef),
            ("b", Arc::new(StringViewArray::from(vec!["x"])) as ArrayRef),
        ])
        .expect("batch");
        assert!(decode_result(stream(&[two]), &DataType::Int64, 1).is_err());
    }

    /// An LZ4 stream whose first compressed buffer declares `declared` uncompressed bytes.
    fn compressed(declared: u64) -> Vec<u8> {
        use arrow_ipc::CompressionType;
        use arrow_ipc::writer::IpcWriteOptions;
        let batch = longs((0..2048).collect());
        let options = IpcWriteOptions::default()
            .try_with_compression(Some(CompressionType::LZ4_FRAME))
            .expect("lz4");
        let mut out = Vec::new();
        let mut w =
            StreamWriter::try_new_with_options(&mut out, &batch.schema(), options).expect("writer");
        w.write(&batch).expect("write");
        w.finish().expect("finish");
        drop(w);
        // The batch message's body starts after its header; its first non-empty compressed buffer begins with
        // the 8-byte uncompressed length. Patch every occurrence of the true length (2048 * 8).
        let true_len = (2048_u64 * 8).to_le_bytes();
        let mut patched = 0;
        for i in 0..out.len().saturating_sub(8) {
            if out[i..i + 8] == true_len {
                out[i..i + 8].copy_from_slice(&declared.to_le_bytes());
                patched += 1;
            }
        }
        assert!(patched > 0, "found the length prefix");
        out
    }

    #[test]
    fn a_compressed_result_is_refused_before_any_allocation() {
        // Review 2026-11-13 M1: arrow-ipc would allocate 2^45 bytes and abort the process.
        for declared in [2048 * 8, 1_u64 << 45, 1_u64 << 62] {
            let err =
                decode_result(compressed(declared), &DataType::Int64, 2048).expect_err("refused");
            assert_eq!(err.error_class(), ErrorClass::PythonException);
            assert!(err.to_string().contains("compressed"), "{err}");
        }
    }

    #[test]
    fn framing_errors_are_refused() {
        let good = stream(&[longs(vec![1])]);
        let first = check_frames(&good).expect("framed");
        let header = arrow_ipc::root_as_message(&good[8..first]).expect("the first header");
        assert!(header.header_as_schema().is_some());
        let batch_first = stream(&[longs(vec![1])])
            .split_off(first + usize::try_from(header.bodyLength()).expect("small"));
        let err = decode_result(batch_first, &DataType::Int64, 1).expect_err("no schema");
        assert!(
            err.to_string().contains("does not start with a schema"),
            "{err}"
        );
        let mut no_marker = good.clone();
        no_marker[0] = 0;
        assert!(
            check_frames(&no_marker)
                .unwrap_err()
                .contains("continuation")
        );
        let mut trailing = good.clone();
        trailing.extend_from_slice(&[0; 3]);
        assert!(
            check_frames(&trailing)
                .unwrap_err()
                .contains("after the end")
        );
        let mut huge_header = good.clone();
        huge_header[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(check_frames(&huge_header).is_err());
        for cut in [3, 9, good.len() - 9] {
            assert!(check_frames(&good[..cut]).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn a_truncated_stream_is_refused() {
        let mut bytes = stream(&[longs((0..1000).collect())]);
        bytes.truncate(bytes.len() / 2);
        assert!(decode_result(bytes, &DataType::Int64, 1000).is_err());
    }

    #[test]
    fn input_goes_into_a_sealed_memfd() {
        let (file, len) = write_input(&longs(vec![7, 8])).expect("write");
        assert!(len > 0);
        assert!(std::os::unix::fs::FileExt::write_at(&file, b"x", 0).is_err());
    }
}
