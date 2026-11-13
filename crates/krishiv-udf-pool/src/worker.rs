// SPDX-License-Identifier: Apache-2.0
//! One worker process: its start, its handshake, and the `udf` and `batch` exchanges of wire v1.
//!
//! A [`Worker`] is used by one call at a time (the pool hands it out exclusively). Every method blocks on the
//! worker, so it runs only on the pool's own threads, never on a compute worker.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Cursor, Read};
use std::os::fd::{AsFd, OwnedFd};
use std::process::{Child, ExitStatus};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::DataType;
use krishiv_common::error::{
    InternalError, KrishivError, KrishivPythonWorkerExited, PythonException,
};
use krishiv_udf_sandbox::{EXIT_SANDBOX_FAILED, SandboxError};
use rustix::process::{Pid, Signal};
use serde_json::{Value, json};

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
}

/// A worker that greeted the server.
pub(crate) struct Worker {
    child: Child,
    pid: Pid,
    control: OwnedFd,
    /// Removed when the worker is dropped (after the process is gone).
    _private_dir: tempfile::TempDir,
    /// The payload ids this worker has loaded.
    loaded: HashSet<u64>,
    stderr: Arc<Mutex<String>>,
    /// False once an exchange failed in a way that leaves the worker unusable.
    healthy: bool,
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
    /// Waits for the worker's greeting and checks its wire version.
    pub(crate) fn handshake(started: Started) -> Result<Self, KrishivError> {
        let Started {
            mut child,
            control,
            private_dir,
        } = started;
        let pid = Pid::from_child(&child);
        let stderr = Arc::new(Mutex::new(String::new()));
        if let Some(pipe) = child.stderr.take() {
            drain_stderr(pipe, pid, Arc::clone(&stderr));
        }
        let mut worker = Worker {
            child,
            pid,
            control,
            _private_dir: private_dir,
            loaded: HashSet::new(),
            stderr,
            healthy: false,
        };
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
        .map_err(|e| worker.protocol(&e.to_string()))?;
        tracing::debug!(
            pid = pid.as_raw_nonzero().get(),
            python = msg.str_field("python"),
            pyspark = msg.str_field("pyspark"),
            "Python UDF worker started"
        );
        worker.healthy = true;
        Ok(worker)
    }

    /// The process id.
    pub(crate) fn pid(&self) -> Pid {
        self.pid
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
            .map_err(|e| self.protocol(&format!("cannot pass the UDF: {e}")))?;
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
        let (file, len) =
            write_input(args).map_err(|e| self.protocol(&format!("cannot pass a batch: {e}")))?;
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
                self.protocol(&e.to_string())
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

    /// The worker broke the protocol (or the server could not talk to it): it is killed and not reused.
    fn protocol(&mut self, what: &str) -> KrishivError {
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
        let tail = self.stderr_tail();
        if let Ok(status) = status
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

    /// Sends `SIGKILL` (the worker may already be gone).
    pub(crate) fn kill(&self) {
        kill_pid(self.pid);
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Closing the control socket ends an idle worker; a busy one is killed. Either way it is reaped
        // here, so its pid is never reused while the pool might still signal it.
        self.kill();
        let _ = self.child.wait();
    }
}

/// Sends `SIGKILL` to `pid`, a worker the pool has not reaped yet (so the pid is still its).
pub(crate) fn kill_pid(pid: Pid) {
    let _ = rustix::process::kill_process(pid, Signal::KILL);
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
/// when the worker does.
fn drain_stderr(mut pipe: impl Read + Send + 'static, pid: Pid, tail: Arc<Mutex<String>>) {
    let spawned = std::thread::Builder::new()
        .name("krishiv-udf-stderr".to_owned())
        .spawn(move || {
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
    if let Err(e) = spawned {
        tracing::warn!("cannot start the thread that reads a Python UDF worker's output: {e}");
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

/// The worker's result stream, read with validation: one column of `return_type`, `rows` rows, one batch.
pub(crate) fn decode_result(
    bytes: Vec<u8>,
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
    let mut reader = StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|e| invalid(format!("unreadable Arrow stream: {e}")))?;
    let schema = reader.schema();
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
    let batch = match reader.next() {
        Some(Ok(batch)) => batch,
        Some(Err(e)) => return Err(invalid(format!("unreadable Arrow batch: {e}"))),
        None => return Err(invalid("no batch".to_owned())),
    };
    if reader.next().is_some() {
        return Err(invalid("more than one batch".to_owned()));
    }
    if batch.num_rows() != rows {
        return Err(invalid(format!(
            "{} rows, expected {rows}",
            batch.num_rows()
        )));
    }
    Ok(Arc::clone(batch.column(0)))
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
