// SPDX-License-Identifier: Apache-2.0
//! [`HostExchange`]: the [`Exchange`] of one simulated host (design P2-T176
//! §2.3, §2.6). Map outputs go to the host's own [`LocalShuffle`]; a reader
//! of map partition `m` reads from worker `m mod W`, which ran it (design
//! §2.5): from its own shuffle when that is this host, else over simulated
//! TCP with a [`Request::Fetch`]. Commit before read and the first
//! committed attempt per map partition are the owning shuffle's.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::exchange::{
    Exchange, ExchangeFrame, ExchangeReader, ExchangeWriter, MapOutputId,
};
use krishiv_exec::physical::ExchangeId;
use krishiv_shuffle::LocalShuffle;
use krishiv_shuffle::frame::body_batch;
use turmoil::net::TcpStream;

use crate::wire::{Reply, Request};

type Result<T> = std::result::Result<T, KrishivError>;

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// A seeded map-task failure (design §2.7): the task attempt `attempt` of
/// map partition `map` fails after `after` writes, in every exchange it
/// writes. The coordinator reruns it as the next attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fault {
    /// The map partition killed.
    pub map: u32,
    /// The attempt killed.
    pub attempt: u32,
    /// Writes that succeed before the failure.
    pub after: u32,
}

/// The worker hosts of a simulated cluster: names, in partition order, and
/// the port each serves on.
#[derive(Debug, Clone)]
pub struct Workers {
    /// The host names; map partition `m` runs on `names[m mod len]`.
    pub names: Arc<[String]>,
    /// The port every worker serves on.
    pub port: u16,
}

impl Workers {
    /// The host that runs map partition `map`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a cluster without workers.
    pub fn owner(&self, map: u32) -> Result<usize> {
        let len = u32::try_from(self.names.len())
            .ok()
            .filter(|n| *n > 0)
            .ok_or_else(|| internal("a cluster without workers".to_owned()))?;
        Ok((map % len) as usize)
    }
}

/// One host's exchange of one job (see the module docs).
#[derive(Debug)]
pub struct HostExchange {
    /// The job whose map outputs it holds and reads.
    job: u64,
    /// This host's worker index; `None` on the coordinator, which only reads.
    host: Option<usize>,
    workers: Workers,
    local: LocalShuffle,
    fault: Option<Fault>,
}

impl HostExchange {
    /// The exchange of `job` on worker `host` of `workers`, injecting
    /// `fault`.
    #[must_use]
    pub fn worker(job: u64, host: usize, workers: Workers, fault: Option<Fault>) -> Self {
        Self {
            job,
            host: Some(host),
            workers,
            local: LocalShuffle::in_memory(),
            fault,
        }
    }

    /// The coordinator's exchange of `job`: it reads every map output
    /// remotely.
    #[must_use]
    pub fn coordinator(job: u64, workers: Workers) -> Self {
        Self {
            job,
            host: None,
            workers,
            local: LocalShuffle::in_memory(),
            fault: None,
        }
    }
}

impl Exchange for HostExchange {
    fn writer(
        &self,
        output: MapOutputId,
        partitions: u32,
        schema: SchemaRef,
    ) -> Result<Box<dyn ExchangeWriter>> {
        if self.host.is_none() {
            return Err(internal("the coordinator writes no map output".to_owned()));
        }
        let inner = self.local.writer(output, partitions, schema)?;
        Ok(match self.fault {
            Some(f) if f.map == output.map && f.attempt == output.attempt => Box::new(Faulty {
                inner,
                left: f.after,
                output,
            }),
            _ => inner,
        })
    }

    fn reader(
        &self,
        exchange: ExchangeId,
        stage_attempt: u32,
        map: u32,
        reduce: u32,
    ) -> Result<Box<dyn ExchangeReader>> {
        let owner = self.workers.owner(map)?;
        if self.host == Some(owner) {
            return self.local.reader(exchange, stage_attempt, map, reduce);
        }
        let name = self
            .workers
            .names
            .get(owner)
            .cloned()
            .ok_or_else(|| internal(format!("no worker {owner}")))?;
        Ok(Box::new(Remote {
            host: name,
            port: self.workers.port,
            request: Request::Fetch {
                job: self.job,
                exchange: exchange.0,
                stage_attempt,
                map,
                reduce,
            },
            stream: None,
            ended: false,
        }))
    }
}

/// A writer that fails after a number of writes (a killed map task).
struct Faulty {
    inner: Box<dyn ExchangeWriter>,
    left: u32,
    output: MapOutputId,
}

impl ExchangeWriter for Faulty {
    fn write(&mut self, reduce: u32, batch: &RecordBatch) -> Result<()> {
        if self.left == 0 {
            return Err(internal(format!(
                "simulated fault: map {} attempt {} of exchange {} killed",
                self.output.map, self.output.attempt, self.output.exchange.0
            )));
        }
        self.left -= 1;
        self.inner.write(reduce, batch)
    }

    fn commit(self: Box<Self>) -> Result<()> {
        if self.left == 0 {
            return Err(internal(format!(
                "simulated fault: map {} attempt {} killed before its commit",
                self.output.map, self.output.attempt
            )));
        }
        self.inner.commit()
    }
}

/// One reduce partition of one map output on another host.
struct Remote {
    host: String,
    port: u16,
    request: Request,
    stream: Option<TcpStream>,
    ended: bool,
}

impl ExchangeReader for Remote {
    fn next(&mut self) -> BoxFuture<'_, Result<ExchangeFrame>> {
        Box::pin(async move {
            if self.ended {
                return Ok(ExchangeFrame::End);
            }
            let stream = match &mut self.stream {
                Some(s) => s,
                None => {
                    let mut s = TcpStream::connect((self.host.as_str(), self.port))
                        .await
                        .map_err(|e| internal(format!("connect to {}: {e}", self.host)))?;
                    self.request.write(&mut s).await?;
                    self.stream.insert(s)
                }
            };
            match Reply::read(stream).await? {
                Reply::Frame(body) => Ok(ExchangeFrame::Data(
                    body_batch(&body).map_err(|e| internal(format!("a fetched frame: {e}")))?,
                )),
                Reply::End => {
                    self.ended = true;
                    self.stream = None;
                    Ok(ExchangeFrame::End)
                }
                Reply::Failed(message) => Err(internal(format!("{}: {message}", self.host))),
                Reply::Done => Err(internal(format!("{}: a fetch answered Done", self.host))),
            }
        })
    }
}
