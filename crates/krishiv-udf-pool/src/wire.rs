// SPDX-License-Identifier: Apache-2.0
//! Wire `krishiv.udf.wire.v1` (design `P3-T014-connect-udfs-sandbox.md` §3.3; the worker side is
//! `python/krishiv/_udf/pool_worker.py`, whose module doc lists the messages).
//!
//! Control messages are JSON objects, one per `SOCK_SEQPACKET` message, with descriptors attached as
//! `SCM_RIGHTS`. Data travels in memfds: the server writes an input stream into a memfd, seals it and passes
//! it; the worker passes back a memfd it filled. The server never maps a memfd: it writes and reads with plain
//! system calls, so no memory is shared with the worker and nothing here needs `unsafe`.

use std::fs::File;
use std::io::{self, IoSlice, IoSliceMut, Write};
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::fs::FileExt;

use arrow_buffer::{Buffer, MutableBuffer};
use rustix::fs::{MemfdFlags, SealFlags};
use rustix::net::{
    RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags, SendAncillaryBuffer,
    SendAncillaryMessage, SendFlags,
};
use serde_json::Value;

/// The wire version both sides speak; a worker greeting with another is refused.
pub(crate) const WIRE_VERSION: u64 = 1;

/// The largest control message the server reads; the worker sends at most 128 KiB.
const MAX_MESSAGE: usize = 256 * 1024;

/// The largest control message the worker reads (`MAX_MESSAGE` in `pool_worker.py`): the server sends no
/// larger one, since the worker would see it cut.
const WORKER_MAX_MESSAGE: usize = 128 * 1024;

/// The most descriptors one message carries (the worker sends one, with a result).
const MAX_FDS: usize = 4;

/// One control message: the JSON object and the descriptors that came with it.
#[derive(Debug)]
pub(crate) struct Message {
    pub(crate) body: Value,
    pub(crate) fds: Vec<OwnedFd>,
}

impl Message {
    /// The message's kind (`t`).
    pub(crate) fn kind(&self) -> &str {
        self.body.get("t").and_then(Value::as_str).unwrap_or("")
    }

    /// The string field `key`, or `""`.
    pub(crate) fn str_field(&self, key: &str) -> &str {
        self.body.get(key).and_then(Value::as_str).unwrap_or("")
    }

    /// The unsigned integer field `key`.
    pub(crate) fn u64_field(&self, key: &str) -> Option<u64> {
        self.body.get(key).and_then(Value::as_u64)
    }
}

/// Sends `body` with `fds` attached.
pub(crate) fn send(socket: BorrowedFd<'_>, body: &Value, fds: &[BorrowedFd<'_>]) -> io::Result<()> {
    let bytes = serde_json::to_vec(body).map_err(io::Error::other)?;
    if bytes.len() > WORKER_MAX_MESSAGE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "a control message of {} bytes is over the worker's limit of {WORKER_MAX_MESSAGE}",
                bytes.len()
            ),
        ));
    }
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_FDS))];
    let mut control = SendAncillaryBuffer::new(&mut space);
    if !fds.is_empty() && !control.push(SendAncillaryMessage::ScmRights(fds)) {
        return Err(io::Error::other("too many descriptors for one message"));
    }
    let sent = rustix::net::sendmsg(
        socket,
        &[IoSlice::new(&bytes)],
        &mut control,
        SendFlags::NOSIGNAL,
    )?;
    if sent != bytes.len() {
        return Err(io::Error::other("a control message was sent in part"));
    }
    Ok(())
}

/// Receives one message; `None` at end of stream (the worker closed its end or exited).
///
/// # Errors
/// A truncated message or descriptor list (`MSG_TRUNC`, `MSG_CTRUNC`), or a body that is not a JSON object,
/// is an error: the worker is then not trusted further.
pub(crate) fn recv(socket: BorrowedFd<'_>) -> io::Result<Option<Message>> {
    let mut buf = vec![0_u8; MAX_MESSAGE];
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_FDS))];
    let mut control = RecvAncillaryBuffer::new(&mut space);
    let got = rustix::net::recvmsg(
        socket,
        &mut [IoSliceMut::new(&mut buf)],
        &mut control,
        RecvFlags::CMSG_CLOEXEC,
    )?;
    // Take every descriptor first, so none leaks on the error paths below.
    let mut fds = Vec::new();
    for msg in control.drain() {
        if let RecvAncillaryMessage::ScmRights(rights) = msg {
            fds.extend(rights);
        }
    }
    if got.flags.contains(ReturnFlags::TRUNC) || got.flags.contains(ReturnFlags::CTRUNC) {
        return Err(io::Error::other(
            "a worker message or its descriptors were truncated",
        ));
    }
    if got.bytes == 0 {
        return Ok(None);
    }
    let body: Value = serde_json::from_slice(buf.get(..got.bytes).unwrap_or_default())
        .map_err(|e| io::Error::other(format!("a worker message is not JSON: {e}")))?;
    if !body.is_object() {
        return Err(io::Error::other("a worker message is not a JSON object"));
    }
    Ok(Some(Message { body, fds }))
}

/// A new memfd (close-on-exec, sealable) as a file.
pub(crate) fn memfd(name: &str) -> io::Result<File> {
    let fd = rustix::fs::memfd_create(name, MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING)?;
    Ok(File::from(fd))
}

/// Writes `bytes` to a new memfd and seals it against every change, for a worker to map read-only.
pub(crate) fn sealed_memfd(name: &str, bytes: &[u8]) -> io::Result<File> {
    let mut file = memfd(name)?;
    file.write_all(bytes)?;
    seal(&file)?;
    Ok(file)
}

/// Seals `file` (a memfd) against writes, growth and shrinking, and against further seal changes.
pub(crate) fn seal(file: &File) -> io::Result<()> {
    rustix::fs::fcntl_add_seals(
        file.as_fd(),
        SealFlags::WRITE | SealFlags::GROW | SealFlags::SHRINK | SealFlags::SEAL,
    )?;
    Ok(())
}

/// Reads the first `len` bytes of a descriptor a worker sent, refusing more than `cap` bytes or a file
/// shorter than `len`. Reads with `pread` into memory the server owns: the worker may still change the file
/// while it is read, which can only change the bytes (they are validated afterwards), never the server's
/// memory. The buffer is Arrow's (64-byte aligned), so the result's columns are decoded as slices of it,
/// with no second copy.
pub(crate) fn read_untrusted(fd: OwnedFd, len: u64, cap: u64) -> io::Result<Buffer> {
    if len > cap {
        return Err(io::Error::other(format!(
            "a worker result of {len} bytes is over the limit of {cap} bytes"
        )));
    }
    let size = rustix::fs::fstat(&fd)?.st_size;
    if u64::try_from(size).unwrap_or(0) < len {
        return Err(io::Error::other(format!(
            "a worker result file holds {size} bytes, fewer than the {len} it announced"
        )));
    }
    let n = usize::try_from(len).map_err(io::Error::other)?;
    let mut out = MutableBuffer::new(0);
    out.try_reserve(n)
        .map_err(|e| io::Error::new(io::ErrorKind::OutOfMemory, e.to_string()))?;
    out.resize(n, 0);
    File::from(fd).read_exact_at(out.as_slice_mut(), 0)?;
    Ok(out.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustix::net::{AddressFamily, SocketFlags, SocketType};
    use serde_json::json;

    fn pair() -> (OwnedFd, OwnedFd) {
        rustix::net::socketpair(
            AddressFamily::UNIX,
            SocketType::SEQPACKET,
            SocketFlags::CLOEXEC,
            None,
        )
        .expect("socketpair")
    }

    #[test]
    fn a_message_carries_its_descriptor() {
        let (a, b) = pair();
        let file = sealed_memfd("t", b"hello").expect("memfd");
        send(a.as_fd(), &json!({"t": "x", "len": 5}), &[file.as_fd()]).expect("send");
        let mut msg = recv(b.as_fd()).expect("recv").expect("a message");
        assert_eq!(msg.kind(), "x");
        assert_eq!(msg.u64_field("len"), Some(5));
        let fd = msg.fds.pop().expect("one fd");
        let read = read_untrusted(fd, 5, 5).expect("read");
        assert_eq!(read.as_slice(), b"hello");
        assert_eq!(read.as_ptr().addr() % 64, 0, "aligned for Arrow");
    }

    #[test]
    fn messages_over_the_workers_limit_are_not_sent() {
        let (a, _b) = pair();
        let big = "x".repeat(WORKER_MAX_MESSAGE);
        let err =
            send(a.as_fd(), &json!({"t": "batch", "conf": {"k": big}}), &[]).expect_err("too big");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn sealed_memfds_refuse_writes() {
        let file = sealed_memfd("t", b"abc").expect("memfd");
        assert!(file.write_at(b"z", 0).is_err());
        assert!(file.set_len(1).is_err());
    }

    #[test]
    fn reads_are_capped_and_checked() {
        let file = sealed_memfd("t", b"abc").expect("memfd");
        let fd = OwnedFd::from(file.try_clone().expect("clone"));
        assert!(read_untrusted(fd, 4, 10).is_err(), "longer than the file");
        let fd = OwnedFd::from(file);
        assert!(read_untrusted(fd, 3, 2).is_err(), "over the cap");
    }

    #[test]
    fn end_of_stream_and_bad_messages() {
        let (a, b) = pair();
        rustix::net::send(a.as_fd(), b"not json", SendFlags::empty()).expect("send");
        rustix::net::send(a.as_fd(), b"[1]", SendFlags::empty()).expect("send");
        drop(a);
        assert!(recv(b.as_fd()).is_err());
        assert!(recv(b.as_fd()).is_err());
        assert!(recv(b.as_fd()).expect("eof").is_none());
    }
}
