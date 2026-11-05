// SPDX-License-Identifier: Apache-2.0
//! The messages hosts exchange over simulated TCP (design P2-T176 §2.6):
//! bytes only, so nothing live crosses a host. Each message is a 4-byte
//! little-endian length, a 1-byte kind and its payload. One request per
//! connection: `Run` and `Release` are answered by `Done` or `Failed`;
//! `Fetch` by the partition's batches as `Frame`s (Arrow IPC bodies, design
//! §2.4) and then `End`, or by `Failed`. Every request names its job (one
//! submitted query): exchange ids are numbered per plan, so each job has a
//! shuffle of its own on every worker.

use krishiv_common::error::{InternalError, KrishivError};
use tokio::io::{AsyncRead, AsyncReadExt as _, AsyncWrite, AsyncWriteExt as _};

/// The largest message a host accepts.
const MAX_MESSAGE: u32 = 1 << 28;

/// What a coordinator or a reduce task asks a worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    /// Run a map task: a `krishiv.plan.v1` fragment and its task descriptor
    /// ([`crate::task`]).
    Run {
        /// The job.
        job: u64,
        /// The fragment's plan bytes.
        fragment: Vec<u8>,
        /// The task descriptor's bytes.
        task: Vec<u8>,
    },
    /// Read one reduce partition of the first committed attempt of one map
    /// output the worker holds.
    Fetch {
        /// The job.
        job: u64,
        /// The exchange.
        exchange: u64,
        /// The attempt of the stage that wrote it.
        stage_attempt: u32,
        /// The map partition.
        map: u32,
        /// The reduce partition.
        reduce: u32,
    },
    /// Drop a finished job's map outputs.
    Release {
        /// The job.
        job: u64,
    },
}

/// A worker's answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reply {
    /// The task ran to its end and committed its output.
    Done,
    /// The request failed: the error's class and message.
    Failed(String),
    /// One batch of a fetched partition, as an Arrow IPC body.
    Frame(Vec<u8>),
    /// The fetched partition has no more batches.
    End,
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

fn io(e: &std::io::Error) -> KrishivError {
    internal(format!("simulated network: {e}"))
}

async fn send(
    out: &mut (impl AsyncWrite + Unpin),
    kind: u8,
    parts: &[&[u8]],
) -> Result<(), KrishivError> {
    let len: usize = 1 + parts.iter().map(|p| p.len()).sum::<usize>();
    let len = u32::try_from(len)
        .ok()
        .filter(|l| *l <= MAX_MESSAGE)
        .ok_or_else(|| internal(format!("a message of {len} bytes")))?;
    out.write_all(&len.to_le_bytes())
        .await
        .map_err(|e| io(&e))?;
    out.write_all(&[kind]).await.map_err(|e| io(&e))?;
    for p in parts {
        out.write_all(p).await.map_err(|e| io(&e))?;
    }
    out.flush().await.map_err(|e| io(&e))
}

async fn receive(input: &mut (impl AsyncRead + Unpin)) -> Result<(u8, Vec<u8>), KrishivError> {
    let mut len = [0; 4];
    input.read_exact(&mut len).await.map_err(|e| io(&e))?;
    let len = u32::from_le_bytes(len);
    if len == 0 || len > MAX_MESSAGE {
        return Err(internal(format!("a message of {len} bytes")));
    }
    let mut body = vec![0; len as usize];
    input.read_exact(&mut body).await.map_err(|e| io(&e))?;
    let rest = body.split_off(1);
    Ok((body[0], rest))
}

fn u32_at(bytes: &[u8], at: usize) -> Result<u32, KrishivError> {
    bytes
        .get(at..at + 4)
        .and_then(|b| b.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| internal("a truncated message".to_owned()))
}

fn u64_at(bytes: &[u8], at: usize) -> Result<u64, KrishivError> {
    bytes
        .get(at..at + 8)
        .and_then(|b| b.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| internal("a truncated message".to_owned()))
}

impl Request {
    /// Writes the request to `out`.
    ///
    /// # Errors
    /// A failed write or an oversized message.
    pub async fn write(&self, out: &mut (impl AsyncWrite + Unpin)) -> Result<(), KrishivError> {
        match self {
            Self::Run {
                job,
                fragment,
                task,
            } => {
                let len = u32::try_from(fragment.len())
                    .map_err(|_| internal("an oversized fragment".to_owned()))?;
                send(
                    out,
                    1,
                    &[&job.to_le_bytes(), &len.to_le_bytes(), fragment, task],
                )
                .await
            }
            Self::Release { job } => send(out, 3, &[&job.to_le_bytes()]).await,
            Self::Fetch {
                job,
                exchange,
                stage_attempt,
                map,
                reduce,
            } => {
                let body = [
                    &job.to_le_bytes()[..],
                    &exchange.to_le_bytes(),
                    &stage_attempt.to_le_bytes(),
                    &map.to_le_bytes(),
                    &reduce.to_le_bytes(),
                ]
                .concat();
                send(out, 2, &[&body]).await
            }
        }
    }

    /// Reads one request from `input`.
    ///
    /// # Errors
    /// A failed read or a malformed message.
    pub async fn read(input: &mut (impl AsyncRead + Unpin)) -> Result<Self, KrishivError> {
        let (kind, body) = receive(input).await?;
        match kind {
            1 => {
                let len = u32_at(&body, 8)? as usize;
                let fragment = body
                    .get(12..12 + len)
                    .ok_or_else(|| internal("a truncated fragment".to_owned()))?
                    .to_vec();
                Ok(Self::Run {
                    job: u64_at(&body, 0)?,
                    fragment,
                    task: body.get(12 + len..).unwrap_or_default().to_vec(),
                })
            }
            2 if body.len() == 28 => Ok(Self::Fetch {
                job: u64_at(&body, 0)?,
                exchange: u64_at(&body, 8)?,
                stage_attempt: u32_at(&body, 16)?,
                map: u32_at(&body, 20)?,
                reduce: u32_at(&body, 24)?,
            }),
            3 if body.len() == 8 => Ok(Self::Release {
                job: u64_at(&body, 0)?,
            }),
            other => Err(internal(format!("a request of kind {other}"))),
        }
    }
}

impl Reply {
    /// Writes the reply to `out`.
    ///
    /// # Errors
    /// A failed write or an oversized message.
    pub async fn write(&self, out: &mut (impl AsyncWrite + Unpin)) -> Result<(), KrishivError> {
        match self {
            Self::Done => send(out, 1, &[]).await,
            Self::Failed(message) => send(out, 2, &[message.as_bytes()]).await,
            Self::Frame(body) => send(out, 3, &[body]).await,
            Self::End => send(out, 4, &[]).await,
        }
    }

    /// Reads one reply from `input`.
    ///
    /// # Errors
    /// A failed read or a malformed message.
    pub async fn read(input: &mut (impl AsyncRead + Unpin)) -> Result<Self, KrishivError> {
        let (kind, body) = receive(input).await?;
        match kind {
            1 if body.is_empty() => Ok(Self::Done),
            2 => Ok(Self::Failed(String::from_utf8_lossy(&body).into_owned())),
            3 => Ok(Self::Frame(body)),
            4 if body.is_empty() => Ok(Self::End),
            other => Err(internal(format!("a reply of kind {other}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn messages_round_trip() {
        let requests = [
            Request::Run {
                job: 4,
                fragment: vec![1, 2, 3],
                task: vec![9],
            },
            Request::Run {
                job: 0,
                fragment: Vec::new(),
                task: Vec::new(),
            },
            Request::Release { job: u64::MAX },
            Request::Fetch {
                job: 5,
                exchange: u64::MAX,
                stage_attempt: 1,
                map: 2,
                reduce: 3,
            },
        ];
        let replies = [
            Reply::Done,
            Reply::Failed("INTERNAL_ERROR: x".to_owned()),
            Reply::Frame(vec![7; 10]),
            Reply::End,
        ];
        let mut buf = Vec::new();
        for r in &requests {
            r.write(&mut buf).await.unwrap();
        }
        for r in &replies {
            r.write(&mut buf).await.unwrap();
        }
        let mut input = buf.as_slice();
        for r in &requests {
            assert_eq!(&Request::read(&mut input).await.unwrap(), r);
        }
        for r in &replies {
            assert_eq!(&Reply::read(&mut input).await.unwrap(), r);
        }
        assert!(input.is_empty());
    }

    #[tokio::test]
    async fn a_malformed_message_is_refused() {
        let mut input: &[u8] = &[0, 0, 0, 0];
        assert!(Request::read(&mut input).await.is_err());
        let mut input: &[u8] = &[2, 0, 0, 0, 9, 0];
        assert!(Reply::read(&mut input).await.is_err());
    }
}
