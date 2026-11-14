// SPDX-License-Identifier: Apache-2.0
//! A blocking client of the Hive metastore's `ThriftHiveMetastore` service: the four reads a read-only catalog
//! needs (`get_all_databases`, `get_database`, `get_all_tables`, `get_table`). Method and argument ids are from
//! `hive_metastore.thrift` (apache/hive master and `rel/release-4.0.0`, identical for these calls).
//!
//! Every call is bounded as a whole ([`limits`](super::limits)): one deadline (`krishiv.catalog.requestTimeout`) that
//! covers connect, write and every read however slowly they arrive, a budget of bytes read, a budget of decoded
//! elements and a longest string. A call runs on the caller's thread: callers on a runtime use `spawn_blocking`. A
//! metastore that needs SASL (Kerberos) is not supported and fails at the first call, not silently. The default
//! port and the buffered binary transport are the common deployment and are settings here, not assumptions; they
//! are checked against a real metastore in the Docker acceptance. (The framed transport is not offered: it is
//! unverified, and a framing layer is one more place a size can hide.)

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use thrift::protocol::{
    TBinaryInputProtocol, TBinaryOutputProtocol, TFieldIdentifier, TInputProtocol,
    TMessageIdentifier, TMessageType, TOutputProtocol, TStructIdentifier, TType,
};
use thrift::transport::{TBufferedReadTransport, TBufferedWriteTransport};
use thrift::{ProtocolErrorKind, TConfiguration, new_protocol_error};

use super::limits::{Budget, Limit};
use super::types::{Database, Table, read_fields, read_strings};

/// How to reach a metastore and how much to trust what it sends.
#[derive(Debug, Clone)]
pub struct HmsConfig {
    /// `host:port`.
    pub address: String,
    /// Longest one call may take, connect to last byte (`krishiv.catalog.requestTimeout`).
    pub request_timeout: Duration,
    /// Most connections open at once; a call waits for one within its deadline.
    pub max_connections: usize,
    /// Most bytes one call may read (`krishiv.catalog.maxMetadataBytes`).
    pub max_message_bytes: usize,
    /// Most decoded elements (list items and map entries) one call may produce.
    pub max_elements: usize,
    /// Longest single string accepted: it is allocated before its bytes arrive.
    pub max_string_bytes: usize,
    /// Deepest nesting accepted.
    pub max_depth: usize,
}

impl HmsConfig {
    /// A buffered binary connection to `address` with the design's defaults: 30 s, 256 MiB, a million elements,
    /// 16 MiB strings, depth 64.
    #[must_use]
    pub fn new(address: impl Into<String>) -> Self {
        Self {
            address: address.into(),
            request_timeout: Duration::from_secs(30),
            max_connections: 4,
            max_message_bytes: 256 << 20,
            max_elements: 1_000_000,
            max_string_bytes: 16 << 20,
            max_depth: 64,
        }
    }
}

/// What a call to the metastore can fail with. A name that does not exist is not an error here: it is `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HmsError {
    /// The metastore raised `MetaException`.
    Meta(String),
    /// A limit of [`HmsConfig`] was exceeded: nothing was loaded.
    Limit(Limit),
    /// The connection, the protocol or the server failed (including a timeout or a server that wants SASL).
    Failed(String),
}

impl From<thrift::Error> for HmsError {
    fn from(e: thrift::Error) -> Self {
        match &e {
            thrift::Error::Protocol(p) if p.kind == ProtocolErrorKind::SizeLimit => {
                Self::Limit(Limit::Size)
            }
            thrift::Error::Protocol(p) if p.kind == ProtocolErrorKind::DepthLimit => {
                Self::Limit(Limit::Depth)
            }
            _ => Self::Failed(describe(&e)),
        }
    }
}

/// A Thrift error with its kind and message: `Display` of the crate's errors keeps only a generic phrase.
fn describe(e: &thrift::Error) -> String {
    match e {
        thrift::Error::Transport(t) => format!("transport error ({:?}): {}", t.kind, t.message),
        thrift::Error::Protocol(p) => format!("protocol error ({:?}): {}", p.kind, p.message),
        thrift::Error::Application(a) => format!("server error ({:?}): {}", a.kind, a.message),
        thrift::Error::User(u) => format!("user error: {u}"),
    }
}

type Result<T> = std::result::Result<T, HmsError>;

/// The read half of the socket: every read spends the call's deadline and byte budget.
struct GuardedRead {
    stream: TcpStream,
    budget: Arc<Budget>,
}

impl Read for GuardedRead {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let Some(remaining) = self.budget.remaining() else {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "the call took too long",
            ));
        };
        let allowed = self.budget.bytes_allowed();
        if allowed == 0 {
            return Err(io::Error::other("the reply is over the byte limit"));
        }
        self.stream.set_read_timeout(Some(remaining))?;
        let cap = buf.len().min(allowed);
        let n = self.stream.read(&mut buf[..cap])?;
        self.budget.spend_bytes(n);
        Ok(n)
    }
}

/// The write half: every write spends the deadline.
struct GuardedWrite {
    stream: TcpStream,
    budget: Arc<Budget>,
}

impl Write for GuardedWrite {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let Some(remaining) = self.budget.remaining() else {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "the call took too long",
            ));
        };
        self.stream.set_write_timeout(Some(remaining))?;
        self.stream.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}

/// One connection to a metastore.
pub struct HmsClient {
    input: Box<dyn TInputProtocol + Send>,
    output: Box<dyn TOutputProtocol + Send>,
    budget: Arc<Budget>,
    /// The deadline the next call must meet, set by [`HmsClient::within`]; otherwise a fresh one.
    deadline: Option<Instant>,
    sequence: i32,
}

impl std::fmt::Debug for HmsClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HmsClient")
            .field("sequence", &self.sequence)
            .finish_non_exhaustive()
    }
}

/// A reply: the success value, or an exception the service declared (by field id).
enum Reply<T> {
    Success(T),
    Declared { id: i16, message: String },
    Nothing,
}

impl HmsClient {
    /// Connects, within the call deadline of `config`.
    ///
    /// # Errors
    /// [`HmsError::Failed`] when the address does not resolve or cannot be reached in time.
    pub fn connect(config: &HmsConfig) -> Result<Self> {
        Self::connect_until(config, Instant::now() + config.request_timeout)
    }

    /// Connects, within `deadline` (set before the caller waited for anything).
    ///
    /// # Errors
    /// [`HmsError::Failed`] when the address does not resolve or cannot be reached in time.
    pub fn connect_until(config: &HmsConfig, deadline: Instant) -> Result<Self> {
        let budget = Arc::new(Budget::new(
            config.request_timeout,
            config.max_message_bytes,
            config.max_elements,
        ));
        budget.begin_until(deadline);
        let addresses = resolve_within(config.address.clone(), deadline, resolver_gate(), |a| {
            a.to_socket_addrs().map(Iterator::collect)
        })?;
        let stream = connect_any(&addresses, deadline)?;
        stream
            .set_nodelay(true)
            .map_err(|e| HmsError::Failed(e.to_string()))?;
        let clone = |s: &TcpStream| {
            s.try_clone()
                .map_err(|e| HmsError::Failed(format!("cannot use the metastore connection: {e}")))
        };
        let (read, write) = (clone(&stream)?, stream);
        let limits = TConfiguration::builder()
            .max_message_size(Some(config.max_message_bytes))
            .max_frame_size(Some(config.max_message_bytes))
            .max_string_size(Some(config.max_string_bytes))
            .max_container_size(Some(config.max_elements))
            .max_recursion_depth(Some(config.max_depth))
            .build()?;
        let input = TBinaryInputProtocol::with_config(
            TBufferedReadTransport::new(GuardedRead {
                stream: read,
                budget: Arc::clone(&budget),
            }),
            true,
            limits.clone(),
        );
        let output = TBinaryOutputProtocol::with_config(
            TBufferedWriteTransport::new(GuardedWrite {
                stream: write,
                budget: Arc::clone(&budget),
            }),
            true,
            limits,
        );
        Ok(Self {
            input: Box::new(input),
            output: Box::new(output),
            budget,
            deadline: None,
            sequence: 0,
        })
    }

    /// `get_all_databases() -> list<string>`.
    ///
    /// # Errors
    /// [`HmsError`].
    pub fn get_all_databases(&mut self) -> Result<Vec<String>> {
        self.send("get_all_databases", |_| Ok(()))?;
        let budget = Arc::clone(&self.budget);
        let reply = self.receive("get_all_databases", |p, t| {
            (t == TType::List)
                .then(|| read_strings(p, &budget))
                .transpose()
        })?;
        match reply {
            Reply::Success(names) => Ok(names),
            Reply::Declared { message, .. } => Err(HmsError::Meta(message)),
            Reply::Nothing => Err(no_result("get_all_databases")),
        }
    }

    /// `get_database(1: string name) -> Database`; `None` when it does not exist (`NoSuchObjectException`).
    ///
    /// # Errors
    /// [`HmsError`].
    pub fn get_database(&mut self, name: &str) -> Result<Option<Database>> {
        self.send("get_database", |o| string_arg(o, "name", 1, name))?;
        let budget = Arc::clone(&self.budget);
        let reply = self.receive("get_database", |p, t| {
            (t == TType::Struct)
                .then(|| Database::read(p, &budget))
                .transpose()
        })?;
        match reply {
            Reply::Success(db) => Ok(Some(db)),
            // throws (1: NoSuchObjectException o1, 2: MetaException o2)
            Reply::Declared { id: 1, .. } => Ok(None),
            Reply::Declared { message, .. } => Err(HmsError::Meta(message)),
            Reply::Nothing => Err(no_result("get_database")),
        }
    }

    /// `get_all_tables(1: string db_name) -> list<string>`.
    ///
    /// # Errors
    /// [`HmsError`].
    pub fn get_all_tables(&mut self, db_name: &str) -> Result<Vec<String>> {
        self.send("get_all_tables", |o| string_arg(o, "db_name", 1, db_name))?;
        let budget = Arc::clone(&self.budget);
        let reply = self.receive("get_all_tables", |p, t| {
            (t == TType::List)
                .then(|| read_strings(p, &budget))
                .transpose()
        })?;
        match reply {
            Reply::Success(names) => Ok(names),
            Reply::Declared { message, .. } => Err(HmsError::Meta(message)),
            Reply::Nothing => Err(no_result("get_all_tables")),
        }
    }

    /// `get_table(1: string dbname, 2: string tbl_name) -> Table`; `None` when it does not exist.
    ///
    /// # Errors
    /// [`HmsError`].
    pub fn get_table(&mut self, db_name: &str, table_name: &str) -> Result<Option<Table>> {
        self.send("get_table", |o| {
            string_arg(o, "dbname", 1, db_name)?;
            string_arg(o, "tbl_name", 2, table_name)
        })?;
        let budget = Arc::clone(&self.budget);
        let reply = self.receive("get_table", |p, t| {
            (t == TType::Struct)
                .then(|| Table::read(p, &budget))
                .transpose()
        })?;
        match reply {
            Reply::Success(table) => Ok(Some(table)),
            // throws (1: MetaException o1, 2: NoSuchObjectException o2)
            Reply::Declared { id: 2, .. } => Ok(None),
            Reply::Declared { message, .. } => Err(HmsError::Meta(message)),
            Reply::Nothing => Err(no_result("get_table")),
        }
    }

    /// The next call must end by `deadline`.
    pub fn within(&mut self, deadline: Instant) {
        self.deadline = Some(deadline);
    }

    /// What the call ran into, named: a limit the budget remembers beats the protocol error it came out as.
    fn classify(&self, e: HmsError) -> HmsError {
        self.budget.tripped().map_or(e, HmsError::Limit)
    }

    fn send(
        &mut self,
        method: &str,
        args: impl FnOnce(&mut dyn TOutputProtocol) -> thrift::Result<()>,
    ) -> Result<()> {
        match self.deadline.take() {
            Some(deadline) => self.budget.begin_until(deadline),
            None => self.budget.begin(),
        }
        self.sequence = self.sequence.wrapping_add(1);
        let sequence = self.sequence;
        let o = &mut *self.output;
        let sent = (|| -> thrift::Result<()> {
            o.write_message_begin(&TMessageIdentifier::new(
                method,
                TMessageType::Call,
                sequence,
            ))?;
            o.write_struct_begin(&TStructIdentifier::new(format!("{method}_args")))?;
            args(o)?;
            o.write_field_stop()?;
            o.write_struct_end()?;
            o.write_message_end()?;
            o.flush()
        })();
        sent.map_err(|e| self.classify(e.into()))
    }

    /// Reads a reply: field 0 is the success value (`success` returns `None` for a type it does not expect),
    /// fields 1 and up are the declared exceptions, each a struct with the message as field 1.
    fn receive<T>(
        &mut self,
        method: &str,
        success: impl FnOnce(&mut dyn TInputProtocol, TType) -> thrift::Result<Option<T>>,
    ) -> Result<Reply<T>> {
        let sequence = self.sequence;
        let read = (|| -> Result<Reply<T>> {
            let p = &mut *self.input;
            let head = p.read_message_begin()?;
            if head.message_type == TMessageType::Exception {
                let error = thrift::Error::read_application_error_from_in_protocol(p)?;
                p.read_message_end()?;
                return Err(HmsError::Failed(format!(
                    "the metastore refused {method}: {}",
                    describe(&thrift::Error::Application(error))
                )));
            }
            if head.message_type != TMessageType::Reply
                || head.name != method
                || head.sequence_number != sequence
            {
                return Err(new_protocol_error(
                    ProtocolErrorKind::InvalidData,
                    format!("an unexpected reply to {method}"),
                )
                .into());
            }
            let mut reply = Reply::Nothing;
            let mut success = Some(success);
            read_fields(p, |p, id, t| {
                if id == 0 {
                    if let Some(read) = success.take()
                        && let Some(value) = read(p, t)?
                    {
                        reply = Reply::Success(value);
                        return Ok(true);
                    }
                    return Ok(false);
                }
                if t == TType::Struct {
                    let mut message = String::new();
                    read_fields(p, |p, id, t| {
                        if id == 1 && t == TType::String {
                            message = p.read_string()?;
                            return Ok(true);
                        }
                        Ok(false)
                    })?;
                    reply = Reply::Declared { id, message };
                    return Ok(true);
                }
                Ok(false)
            })?;
            p.read_message_end()?;
            Ok(reply)
        })();
        read.map_err(|e| self.classify(e))
    }
}

/// The lookups in flight. A lookup that outlives its call keeps its thread until the resolver returns, so during a DNS
/// outage the threads would pile up: the gate refuses a new lookup, at once, while too many are still running.
#[derive(Debug)]
pub(crate) struct ResolverGate {
    in_flight: AtomicUsize,
    cap: usize,
}

impl ResolverGate {
    pub(crate) fn new(cap: usize) -> Arc<Self> {
        Arc::new(Self {
            in_flight: AtomicUsize::new(0),
            cap,
        })
    }

    /// A permit for one lookup, kept until the lookup finishes; `None` when `cap` are running.
    fn permit(self: &Arc<Self>) -> Option<ResolverPermit> {
        self.in_flight
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.cap).then_some(n + 1)
            })
            .ok()
            .map(|_| ResolverPermit(Arc::clone(self)))
    }
}

struct ResolverPermit(Arc<ResolverGate>);

impl Drop for ResolverPermit {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The process-wide gate: eight lookups at a time is far more than a healthy resolver ever has in flight.
fn resolver_gate() -> &'static Arc<ResolverGate> {
    static GATE: OnceLock<Arc<ResolverGate>> = OnceLock::new();
    GATE.get_or_init(|| ResolverGate::new(8))
}

/// Resolves `address` within `deadline`: a literal `ip:port` needs no lookup, and a name is looked up on a thread of
/// its own so a stalled resolver cannot hold the call past the deadline (the thread ends when the resolver returns).
fn resolve_within(
    address: String,
    deadline: Instant,
    gate: &Arc<ResolverGate>,
    resolver: impl FnOnce(&str) -> io::Result<Vec<SocketAddr>> + Send + 'static,
) -> Result<Vec<SocketAddr>> {
    if let Ok(literal) = address.parse::<SocketAddr>() {
        return Ok(vec![literal]);
    }
    let left = deadline
        .checked_duration_since(Instant::now())
        .filter(|d| !d.is_zero())
        .ok_or_else(|| {
            HmsError::Failed("out of time before resolving the metastore address".to_owned())
        })?;
    let Some(permit) = gate.permit() else {
        return Err(HmsError::Failed(
            "too many metastore address lookups are still running".to_owned(),
        ));
    };
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        // Held until the lookup returns, whether or not the call is still waiting for it.
        let _permit = permit;
        // The receiver may have given up: nothing to do about that.
        let _ = sender.send(resolver(&address));
    });
    match receiver.recv_timeout(left) {
        Ok(Ok(addresses)) if !addresses.is_empty() => Ok(addresses),
        Ok(Ok(_)) => Err(HmsError::Failed(
            "the metastore address resolves to nothing".to_owned(),
        )),
        Ok(Err(e)) => Err(HmsError::Failed(format!(
            "cannot resolve the metastore address: {e}"
        ))),
        Err(_) => Err(HmsError::Failed(
            "timed out resolving the metastore address".to_owned(),
        )),
    }
}

/// Connects to the first of `addresses` that answers, all within `deadline`: each attempt gets an equal share of
/// what is left, so a dead first address (`localhost` resolving to `::1` before `127.0.0.1`) leaves time for the next.
fn connect_any(addresses: &[SocketAddr], deadline: Instant) -> Result<TcpStream> {
    let mut last = None;
    for (i, address) in addresses.iter().enumerate() {
        let Some(left) = deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
        else {
            break;
        };
        let remaining_addresses = u32::try_from(addresses.len() - i).unwrap_or(1).max(1);
        match TcpStream::connect_timeout(address, left / remaining_addresses) {
            Ok(stream) => return Ok(stream),
            Err(e) => last = Some(e),
        }
    }
    Err(HmsError::Failed(match last {
        Some(e) => format!("cannot connect to the metastore: {e}"),
        None => "cannot connect to the metastore: out of time".to_owned(),
    }))
}

/// A reply that holds neither the result nor a declared exception is damaged: it does not say the name is absent.
fn no_result(method: &str) -> HmsError {
    HmsError::Failed(format!("the reply to {method} has no result"))
}

fn string_arg(o: &mut dyn TOutputProtocol, name: &str, id: i16, value: &str) -> thrift::Result<()> {
    o.write_field_begin(&TFieldIdentifier::new(name, TType::String, id))?;
    o.write_string(value)?;
    o.write_field_end()
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::thread;

    use thrift::protocol::{TListIdentifier, TMessageType};
    use thrift::{ApplicationError, ApplicationErrorKind};

    use super::super::test_server::fake::{
        cfg, exception, field, reply_with, serve, strings_field, struct_field, write_table,
    };
    use super::*;

    #[test]
    fn it_lists_databases_and_tables() {
        let (address, server) = serve(2, |method, seq, o| {
            reply_with(o, method, seq, |o| match method {
                "get_all_databases" => strings_field(o, 0, &["default", "web"]),
                _ => strings_field(o, 0, &["events", "users"]),
            })
        });
        let mut client = HmsClient::connect(&cfg(&address)).unwrap();
        assert_eq!(client.get_all_databases().unwrap(), ["default", "web"]);
        assert_eq!(client.get_all_tables("web").unwrap(), ["events", "users"]);
        server.join().unwrap();
    }

    #[test]
    fn it_reads_a_table_and_skips_what_it_does_not_know() {
        let (address, server) = serve(1, |method, seq, o| {
            reply_with(o, method, seq, |o| struct_field(o, 0, write_table))
        });
        let table = HmsClient::connect(&cfg(&address))
            .unwrap()
            .get_table("web", "events")
            .unwrap()
            .unwrap();
        server.join().unwrap();
        assert_eq!(
            (table.table_name.as_str(), table.db_name.as_str()),
            ("events", "web")
        );
        assert_eq!(table.owner.as_deref(), Some("etl"));
        assert_eq!(table.create_time, 1_700_000_000);
        assert_eq!(table.table_type.as_deref(), Some("EXTERNAL_TABLE"));
        assert_eq!(table.parameters["spark.sql.sources.provider"], "parquet");
        assert_eq!(table.partition_keys.len(), 1);
        assert_eq!(table.partition_keys[0].name, "dt");
        let sd = table.sd.unwrap();
        assert_eq!(sd.location.as_deref(), Some("s3a://bucket/web/events"));
        assert_eq!(sd.cols.len(), 2);
        assert_eq!(
            (
                sd.cols[0].name.as_str(),
                sd.cols[0].type_name.as_str(),
                sd.cols[0].comment.as_deref()
            ),
            ("id", "bigint", Some("the key"))
        );
        assert_eq!(sd.cols[1].type_name, "map<string,string>");
        assert!(sd.compressed);
        assert_eq!(sd.num_buckets, -1);
        let serde = sd.serde_info.unwrap();
        assert!(
            serde
                .serialization_lib
                .unwrap()
                .ends_with("ParquetHiveSerDe")
        );
        assert_eq!(serde.parameters["serialization.format"], "1");
    }

    #[test]
    fn a_missing_table_or_database_is_none_and_a_meta_exception_is_an_error() {
        let (address, server) = serve(4, |method, seq, o| {
            reply_with(o, method, seq, |o| match method {
                // get_database throws (1: NoSuchObject, 2: Meta)
                "get_database" if seq == 1 => exception(o, 1, "no such database"),
                "get_database" => exception(o, 2, "metastore is unhappy"),
                // get_table throws (1: Meta, 2: NoSuchObject)
                _ if seq == 3 => exception(o, 2, "no such table"),
                _ => exception(o, 1, "metastore is unhappy"),
            })
        });
        let mut client = HmsClient::connect(&cfg(&address)).unwrap();
        assert_eq!(client.get_database("nope").unwrap(), None);
        assert_eq!(
            client.get_database("x").unwrap_err(),
            HmsError::Meta("metastore is unhappy".to_owned())
        );
        assert_eq!(client.get_table("web", "nope").unwrap(), None);
        assert_eq!(
            client.get_table("web", "x").unwrap_err(),
            HmsError::Meta("metastore is unhappy".to_owned())
        );
        server.join().unwrap();
    }

    #[test]
    fn a_server_exception_is_a_failure_naming_the_call() {
        let (address, server) = serve(1, |method, seq, o| {
            o.write_message_begin(&TMessageIdentifier::new(
                method,
                TMessageType::Exception,
                seq,
            ))?;
            thrift::Error::write_application_error_to_out_protocol(
                &ApplicationError::new(ApplicationErrorKind::InternalError, "boom"),
                o,
            )?;
            o.write_message_end()
        });
        let e = HmsClient::connect(&cfg(&address))
            .unwrap()
            .get_all_databases()
            .unwrap_err();
        server.join().unwrap();
        let HmsError::Failed(m) = e else {
            panic!("{e:?}");
        };
        assert!(m.contains("get_all_databases") && m.contains("boom"), "{m}");
    }

    #[test]
    fn a_reply_to_another_call_is_refused() {
        let (address, server) = serve(1, |_, seq, o| {
            reply_with(o, "get_all_tables", seq, |o| strings_field(o, 0, &[]))
        });
        let e = HmsClient::connect(&cfg(&address))
            .unwrap()
            .get_all_databases()
            .unwrap_err();
        server.join().unwrap();
        assert!(matches!(e, HmsError::Failed(m) if m.contains("unexpected reply")));
    }

    #[test]
    fn a_stalled_metastore_fails_at_the_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let held = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            thread::sleep(Duration::from_secs(3));
            drop(stream);
        });
        let config = HmsConfig {
            request_timeout: Duration::from_millis(300),
            ..HmsConfig::new(&address)
        };
        let started = std::time::Instant::now();
        let e = HmsClient::connect(&config)
            .unwrap()
            .get_all_databases()
            .unwrap_err();
        assert!(matches!(e, HmsError::Failed(_)), "{e:?}");
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "it waited {:?}",
            started.elapsed()
        );
        held.join().unwrap();
    }

    #[test]
    fn a_reply_over_the_size_limit_is_refused_not_read() {
        let (address, server) = serve(1, |method, seq, o| {
            reply_with(o, method, seq, |o| {
                strings_field(o, 0, &["x".repeat(100_000).as_str()])
            })
        });
        let config = HmsConfig {
            max_message_bytes: 4096,
            ..cfg(&address)
        };
        let e = HmsClient::connect(&config)
            .unwrap()
            .get_all_databases()
            .unwrap_err();
        assert_eq!(e, HmsError::Limit(Limit::Bytes));
        drop(server);
    }

    #[test]
    fn a_reply_nested_past_the_depth_limit_is_refused() {
        let (address, server) = serve(1, |method, seq, o| {
            reply_with(o, method, seq, |o| {
                // An unknown field holding structs nested 20 deep.
                fn nest(o: &mut dyn TOutputProtocol, depth: usize) -> thrift::Result<()> {
                    struct_field(o, 50, |o| {
                        if depth == 0 {
                            Ok(())
                        } else {
                            nest(o, depth - 1)
                        }
                    })
                }
                nest(o, 20)
            })
        });
        let config = HmsConfig {
            max_depth: 5,
            ..cfg(&address)
        };
        let e = HmsClient::connect(&config)
            .unwrap()
            .get_all_databases()
            .unwrap_err();
        assert_eq!(e, HmsError::Limit(Limit::Depth));
        drop(server);
    }

    #[test]
    fn an_unreachable_metastore_is_a_failure_not_a_hang() {
        let closed = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().to_string()
        };
        let e = HmsClient::connect(&cfg(&closed)).unwrap_err();
        assert!(matches!(e, HmsError::Failed(m) if m.contains("cannot connect")));
        let e = HmsClient::connect(&cfg("not an address")).unwrap_err();
        assert!(matches!(e, HmsError::Failed(_)));
    }

    /// Empty `FieldSchema` structs: one byte each on the wire, a whole struct each once decoded.
    fn tiny_structs(o: &mut dyn TOutputProtocol, id: i16, n: i32) -> thrift::Result<()> {
        field(o, TType::List, id)?;
        o.write_list_begin(&TListIdentifier::new(TType::Struct, n))?;
        for _ in 0..n {
            o.write_struct_begin(&TStructIdentifier::new("FieldSchema"))?;
            o.write_field_stop()?;
            o.write_struct_end()?;
        }
        o.write_list_end()?;
        o.write_field_end()
    }

    /// Many lists that are each under the per-container cap but together over the call's element budget: 1,200
    /// one-byte structs against a budget of 1,000.
    #[test]
    fn many_tiny_elements_spread_over_lists_are_refused_by_the_element_budget() {
        let (address, server) = serve(1, |method, seq, o| {
            reply_with(o, method, seq, |o| {
                struct_field(o, 0, |o| {
                    struct_field(o, 7, |o| tiny_structs(o, 1, 600))?;
                    tiny_structs(o, 8, 600)
                })
            })
        });
        let config = HmsConfig {
            max_elements: 1_000,
            ..cfg(&address)
        };
        let e = HmsClient::connect(&config)
            .unwrap()
            .get_table("web", "t")
            .unwrap_err();
        assert_eq!(e, HmsError::Limit(Limit::Elements));
        drop(server);
    }

    /// One list over the per-container cap is refused by the protocol layer before a struct is read.
    #[test]
    fn a_single_list_over_the_container_cap_is_refused() {
        let (address, server) = serve(1, |method, seq, o| {
            reply_with(o, method, seq, |o| {
                struct_field(o, 0, |o| tiny_structs(o, 8, 5_000))
            })
        });
        let config = HmsConfig {
            max_elements: 1_000,
            ..cfg(&address)
        };
        let e = HmsClient::connect(&config)
            .unwrap()
            .get_table("web", "t")
            .unwrap_err();
        assert_eq!(e, HmsError::Limit(Limit::Size));
        drop(server);
    }

    /// Lists under the budget pass: the budget is a ceiling, not a tax.
    #[test]
    fn lists_under_the_budget_are_read() {
        let (address, server) = serve(1, |method, seq, o| {
            reply_with(o, method, seq, |o| {
                struct_field(o, 0, |o| {
                    struct_field(o, 7, |o| tiny_structs(o, 1, 400))?;
                    tiny_structs(o, 8, 400)
                })
            })
        });
        let config = HmsConfig {
            max_elements: 1_000,
            ..cfg(&address)
        };
        let table = HmsClient::connect(&config)
            .unwrap()
            .get_table("web", "t")
            .unwrap()
            .unwrap();
        assert_eq!(table.sd.unwrap().cols.len(), 400);
        assert_eq!(table.partition_keys.len(), 400);
        server.join().unwrap();
    }

    /// A string that announces a gigabyte and sends nothing: refused before anything is allocated or waited for.
    #[test]
    fn a_string_that_announces_more_than_the_limit_is_refused_before_its_bytes_arrive() {
        let (address, server) = serve(1, |method, seq, o| {
            o.write_message_begin(&TMessageIdentifier::new(method, TMessageType::Reply, seq))?;
            o.write_struct_begin(&TStructIdentifier::new("r"))?;
            field(o, TType::List, 0)?;
            o.write_list_begin(&TListIdentifier::new(TType::String, 1))?;
            o.write_i32(1 << 30)?;
            o.flush()?;
            thread::sleep(Duration::from_secs(3));
            Ok(())
        });
        let started = std::time::Instant::now();
        let e = HmsClient::connect(&cfg(&address))
            .unwrap()
            .get_all_databases()
            .unwrap_err();
        assert_eq!(e, HmsError::Limit(Limit::Size));
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "it waited {:?}",
            started.elapsed()
        );
        drop(server);
    }

    /// The timeout is for the whole call: a server that sends a valid reply one byte at a time never lets a single
    /// read time out, but cannot hold the call past the deadline.
    #[test]
    fn a_server_that_trickles_its_reply_cannot_hold_the_call_past_the_deadline() {
        use std::io::Write;

        struct Trickle(std::net::TcpStream);
        impl Write for Trickle {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                for byte in buf {
                    if self.0.write_all(&[*byte]).is_err() {
                        break;
                    }
                    self.0.flush().ok();
                    thread::sleep(Duration::from_millis(40));
                }
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.0.flush()
            }
        }

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let read = stream.try_clone().unwrap();
            let mut input = TBinaryInputProtocol::new(TBufferedReadTransport::new(read), true);
            let head = input.read_message_begin().unwrap();
            read_fields(&mut input, |_, _, _| Ok(false)).unwrap();
            let mut output =
                TBinaryOutputProtocol::new(TBufferedWriteTransport::new(Trickle(stream)), true);
            // Errors are expected once the client has given up.
            let _ = reply_with(&mut output, &head.name, head.sequence_number, |o| {
                strings_field(o, 0, &["default", "web", "more", "names"])
            });
            let _ = output.flush();
        });
        let config = HmsConfig {
            request_timeout: Duration::from_millis(500),
            ..HmsConfig::new(&address)
        };
        let started = std::time::Instant::now();
        let e = HmsClient::connect(&config)
            .unwrap()
            .get_all_databases()
            .unwrap_err();
        let waited = started.elapsed();
        assert!(matches!(e, HmsError::Failed(_)), "{e:?}");
        assert!(
            waited < Duration::from_millis(1_800),
            "it waited {waited:?}"
        );
        server.join().unwrap();
    }

    #[test]
    fn a_list_of_the_wrong_element_type_is_refused() {
        let (address, server) = serve(1, |method, seq, o| {
            reply_with(o, method, seq, |o| {
                field(o, TType::List, 0)?;
                o.write_list_begin(&TListIdentifier::new(TType::I32, 1))?;
                o.write_i32(7)?;
                o.write_list_end()?;
                o.write_field_end()
            })
        });
        let e = HmsClient::connect(&cfg(&address))
            .unwrap()
            .get_all_databases()
            .unwrap_err();
        assert!(matches!(e, HmsError::Failed(_)), "{e:?}");
        drop(server);
    }

    /// A reply that has neither the result nor a declared exception does not say the name is absent.
    #[test]
    fn a_reply_with_no_result_and_no_exception_is_a_failure_not_absence() {
        let (address, server) = serve(4, |method, seq, o| reply_with(o, method, seq, |_| Ok(())));
        let mut client = HmsClient::connect(&cfg(&address)).unwrap();
        for e in [
            client.get_table("web", "t").map(|_| ()).unwrap_err(),
            client.get_database("web").map(|_| ()).unwrap_err(),
            client.get_all_tables("web").map(|_| ()).unwrap_err(),
            client.get_all_databases().map(|_| ()).unwrap_err(),
        ] {
            assert!(
                matches!(&e, HmsError::Failed(m) if m.contains("has no result")),
                "{e:?}"
            );
        }
        server.join().unwrap();
    }

    /// A success field of the wrong wire type is also no result.
    #[test]
    fn a_result_of_the_wrong_type_is_a_failure_not_absence() {
        let (address, server) = serve(1, |method, seq, o| {
            reply_with(o, method, seq, |o| {
                field(o, TType::I32, 0)?;
                o.write_i32(7)?;
                o.write_field_end()
            })
        });
        let e = HmsClient::connect(&cfg(&address))
            .unwrap()
            .get_table("web", "t")
            .unwrap_err();
        assert!(
            matches!(&e, HmsError::Failed(m) if m.contains("has no result")),
            "{e:?}"
        );
        server.join().unwrap();
    }

    #[test]
    fn a_stalled_resolver_cannot_hold_the_call_past_the_deadline() {
        let started = Instant::now();
        let e = resolve_within(
            "metastore.example:9083".to_owned(),
            Instant::now() + Duration::from_millis(200),
            &ResolverGate::new(8),
            |_| {
                thread::sleep(Duration::from_secs(3));
                Ok(Vec::new())
            },
        )
        .unwrap_err();
        assert!(
            matches!(&e, HmsError::Failed(m) if m.contains("timed out resolving")),
            "{e:?}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn a_literal_address_needs_no_lookup_and_a_failed_lookup_is_named() {
        let literal = resolve_within(
            "127.0.0.1:9083".to_owned(),
            Instant::now() + Duration::from_secs(1),
            &ResolverGate::new(8),
            |_| panic!("a literal address is not looked up"),
        )
        .unwrap();
        assert_eq!(literal, ["127.0.0.1:9083".parse::<SocketAddr>().unwrap()]);
        let e = resolve_within(
            "nope.invalid:1".to_owned(),
            Instant::now() + Duration::from_secs(1),
            &ResolverGate::new(8),
            |_| Err(io::Error::other("no such host")),
        )
        .unwrap_err();
        assert!(
            matches!(&e, HmsError::Failed(m) if m.contains("cannot resolve")),
            "{e:?}"
        );
        let none = resolve_within(
            "x.invalid:1".to_owned(),
            Instant::now() + Duration::from_secs(1),
            &ResolverGate::new(8),
            |_| Ok(Vec::new()),
        )
        .unwrap_err();
        assert!(
            matches!(&none, HmsError::Failed(m) if m.contains("resolves to nothing")),
            "{none:?}"
        );
    }

    /// The first address is dead, the second answers: a hostname that resolves to both still connects.
    #[test]
    fn a_dead_first_address_leaves_time_for_the_next() {
        let (live, server) = serve(1, |method, seq, o| {
            reply_with(o, method, seq, |o| strings_field(o, 0, &["default"]))
        });
        let dead = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap()
        };
        let addresses = [dead, live.parse::<SocketAddr>().unwrap()];
        let stream = connect_any(&addresses, Instant::now() + Duration::from_secs(5)).unwrap();
        drop(stream);
        let e = connect_any(&[dead], Instant::now() + Duration::from_secs(5)).unwrap_err();
        assert!(
            matches!(&e, HmsError::Failed(m) if m.contains("cannot connect")),
            "{e:?}"
        );
        let out_of_time = connect_any(&addresses, Instant::now()).unwrap_err();
        assert!(
            matches!(&out_of_time, HmsError::Failed(m) if m.contains("out of time")),
            "{out_of_time:?}"
        );
        drop(server);
    }

    /// During a DNS outage the lookups outlive their calls: only `cap` threads may be stuck at once, and a call that
    /// would need another fails at once rather than adding a thread.
    #[test]
    fn stalled_lookups_are_bounded_and_the_gate_frees_when_they_return() {
        let gate = ResolverGate::new(2);
        let release = Arc::new(std::sync::Barrier::new(3));
        let stalled = |release: Arc<std::sync::Barrier>| {
            move |_: &str| -> io::Result<Vec<SocketAddr>> {
                release.wait();
                Ok(Vec::new())
            }
        };
        for _ in 0..2 {
            let e = resolve_within(
                "a.example:1".to_owned(),
                Instant::now() + Duration::from_millis(100),
                &gate,
                stalled(Arc::clone(&release)),
            )
            .unwrap_err();
            assert!(
                matches!(&e, HmsError::Failed(m) if m.contains("timed out")),
                "{e:?}"
            );
        }
        // Both lookups are still stuck: a third is refused at once, without a thread.
        let started = Instant::now();
        let e = resolve_within(
            "b.example:1".to_owned(),
            Instant::now() + Duration::from_secs(5),
            &gate,
            |_| Ok(Vec::new()),
        )
        .unwrap_err();
        assert!(
            matches!(&e, HmsError::Failed(m) if m.contains("too many")),
            "{e:?}"
        );
        assert!(started.elapsed() < Duration::from_millis(500));
        // When the stuck lookups return, their permits come back.
        release.wait();
        let ok = (0..50).find_map(|_| {
            resolve_within(
                "c.example:1".to_owned(),
                Instant::now() + Duration::from_secs(2),
                &gate,
                |_| Ok(vec!["127.0.0.1:1".parse().unwrap()]),
            )
            .ok()
            .or_else(|| {
                thread::sleep(Duration::from_millis(20));
                None
            })
        });
        assert_eq!(ok, Some(vec!["127.0.0.1:1".parse::<SocketAddr>().unwrap()]));
    }
}
