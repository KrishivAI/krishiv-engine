// SPDX-License-Identifier: Apache-2.0
//! A blocking client of the Hive metastore's `ThriftHiveMetastore` service: the four reads a read-only catalog
//! needs (`get_all_databases`, `get_database`, `get_all_tables`, `get_table`). Method and argument ids are from
//! `hive_metastore.thrift` (apache/hive master and `rel/release-4.0.0`, identical for these calls).
//!
//! Every call has the catalog request timeout (connect, read and write), the message size and nesting limits of
//! the design's parse limits (`krishiv.catalog.maxMetadataBytes`), and runs on the caller's thread: callers on a
//! runtime use `spawn_blocking`. A metastore that needs SASL (Kerberos) is not supported and fails at the first
//! call, not silently. The default port, TBinaryProtocol and buffered transport are the common deployment and are
//! settings here, not assumptions; they are checked against a real metastore in the Docker acceptance.

use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use thrift::protocol::{
    TBinaryInputProtocol, TBinaryOutputProtocol, TFieldIdentifier, TInputProtocol,
    TMessageIdentifier, TMessageType, TOutputProtocol, TStructIdentifier, TType,
};
use thrift::transport::{
    ReadHalf, TBufferedReadTransport, TBufferedWriteTransport, TFramedReadTransport,
    TFramedWriteTransport, TIoChannel, TTcpChannel, WriteHalf,
};
use thrift::{TConfiguration, new_protocol_error};

use super::types::{Database, Table, read_fields, read_strings};

/// How to reach a metastore and how much to trust what it sends.
#[derive(Debug, Clone)]
pub struct HmsConfig {
    /// `host:port`.
    pub address: String,
    /// Framed transport instead of buffered (metastores configured with `hive.metastore.thrift.framed.transport.enabled`).
    pub framed: bool,
    /// Connect, read and write timeout of one call (`krishiv.catalog.requestTimeout`).
    pub request_timeout: Duration,
    /// Largest message accepted, on decoded bytes (`krishiv.catalog.maxMetadataBytes`).
    pub max_message_bytes: usize,
    /// Deepest nesting accepted.
    pub max_depth: usize,
}

impl HmsConfig {
    /// A buffered binary connection to `address` with the design's defaults: 30 s, 256 MiB, depth 64.
    #[must_use]
    pub fn new(address: impl Into<String>) -> Self {
        Self {
            address: address.into(),
            framed: false,
            request_timeout: Duration::from_secs(30),
            max_message_bytes: 256 << 20,
            max_depth: 64,
        }
    }
}

/// What a call to the metastore can fail with. A name that does not exist is not an error here: it is `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HmsError {
    /// The metastore raised `MetaException`.
    Meta(String),
    /// The connection, the protocol or the server failed (including a timeout, a message over the limit, or a
    /// server that wants SASL).
    Failed(String),
}

impl From<thrift::Error> for HmsError {
    fn from(e: thrift::Error) -> Self {
        Self::Failed(describe(&e))
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

/// One connection to a metastore.
pub struct HmsClient {
    input: Box<dyn TInputProtocol + Send>,
    output: Box<dyn TOutputProtocol + Send>,
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
    /// Connects, with the connect, read and write timeouts of `config`.
    ///
    /// # Errors
    /// [`HmsError::Failed`] when the address does not resolve or cannot be reached in time.
    pub fn connect(config: &HmsConfig) -> Result<Self> {
        let address = config
            .address
            .to_socket_addrs()
            .map_err(|e| HmsError::Failed(format!("cannot resolve the metastore address: {e}")))?
            .next()
            .ok_or_else(|| {
                HmsError::Failed("the metastore address resolves to nothing".to_owned())
            })?;
        let stream = TcpStream::connect_timeout(&address, config.request_timeout)
            .map_err(|e| HmsError::Failed(format!("cannot connect to the metastore: {e}")))?;
        stream
            .set_nodelay(true)
            .map_err(|e| HmsError::Failed(e.to_string()))?;
        let mut channel = TTcpChannel::with_stream(stream);
        channel.set_timeouts(Some(config.request_timeout), Some(config.request_timeout))?;
        let (read, write) = channel.split()?;
        let limits = TConfiguration::builder()
            .max_message_size(Some(config.max_message_bytes))
            .max_frame_size(Some(config.max_message_bytes))
            .max_string_size(Some(config.max_message_bytes))
            .max_recursion_depth(Some(config.max_depth))
            .build()?;
        let (input, output) = protocols(config.framed, read, write, limits);
        Ok(Self {
            input,
            output,
            sequence: 0,
        })
    }

    /// `get_all_databases() -> list<string>`.
    ///
    /// # Errors
    /// [`HmsError`].
    pub fn get_all_databases(&mut self) -> Result<Vec<String>> {
        self.send("get_all_databases", |_| Ok(()))?;
        match self.receive("get_all_databases", |p, t| {
            (t == TType::List).then(|| read_strings(p)).transpose()
        })? {
            Reply::Success(names) => Ok(names),
            Reply::Declared { message, .. } => Err(HmsError::Meta(message)),
            Reply::Nothing => Ok(Vec::new()),
        }
    }

    /// `get_database(1: string name) -> Database`; `None` when it does not exist (`NoSuchObjectException`).
    ///
    /// # Errors
    /// [`HmsError`].
    pub fn get_database(&mut self, name: &str) -> Result<Option<Database>> {
        self.send("get_database", |o| string_arg(o, "name", 1, name))?;
        match self.receive("get_database", |p, t| {
            (t == TType::Struct).then(|| Database::read(p)).transpose()
        })? {
            Reply::Success(db) => Ok(Some(db)),
            // throws (1: NoSuchObjectException o1, 2: MetaException o2)
            Reply::Declared { id: 1, .. } | Reply::Nothing => Ok(None),
            Reply::Declared { message, .. } => Err(HmsError::Meta(message)),
        }
    }

    /// `get_all_tables(1: string db_name) -> list<string>`.
    ///
    /// # Errors
    /// [`HmsError`].
    pub fn get_all_tables(&mut self, db_name: &str) -> Result<Vec<String>> {
        self.send("get_all_tables", |o| string_arg(o, "db_name", 1, db_name))?;
        match self.receive("get_all_tables", |p, t| {
            (t == TType::List).then(|| read_strings(p)).transpose()
        })? {
            Reply::Success(names) => Ok(names),
            Reply::Declared { message, .. } => Err(HmsError::Meta(message)),
            Reply::Nothing => Ok(Vec::new()),
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
        match self.receive("get_table", |p, t| {
            (t == TType::Struct).then(|| Table::read(p)).transpose()
        })? {
            Reply::Success(table) => Ok(Some(table)),
            // throws (1: MetaException o1, 2: NoSuchObjectException o2)
            Reply::Declared { id: 2, .. } | Reply::Nothing => Ok(None),
            Reply::Declared { message, .. } => Err(HmsError::Meta(message)),
        }
    }

    fn send(
        &mut self,
        method: &str,
        args: impl FnOnce(&mut dyn TOutputProtocol) -> thrift::Result<()>,
    ) -> Result<()> {
        self.sequence = self.sequence.wrapping_add(1);
        let o = &mut *self.output;
        o.write_message_begin(&TMessageIdentifier::new(
            method,
            TMessageType::Call,
            self.sequence,
        ))?;
        o.write_struct_begin(&TStructIdentifier::new(format!("{method}_args")))?;
        args(o)?;
        o.write_field_stop()?;
        o.write_struct_end()?;
        o.write_message_end()?;
        o.flush()?;
        Ok(())
    }

    /// Reads a reply: field 0 is the success value (`success` returns `None` for a type it does not expect),
    /// fields 1 and up are the declared exceptions, each a struct with the message as field 1.
    fn receive<T>(
        &mut self,
        method: &str,
        success: impl FnOnce(&mut dyn TInputProtocol, TType) -> thrift::Result<Option<T>>,
    ) -> Result<Reply<T>> {
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
            || head.sequence_number != self.sequence
        {
            return Err(new_protocol_error(
                thrift::ProtocolErrorKind::InvalidData,
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
    }
}

fn string_arg(o: &mut dyn TOutputProtocol, name: &str, id: i16, value: &str) -> thrift::Result<()> {
    o.write_field_begin(&TFieldIdentifier::new(name, TType::String, id))?;
    o.write_string(value)?;
    o.write_field_end()
}

type Protocols = (
    Box<dyn TInputProtocol + Send>,
    Box<dyn TOutputProtocol + Send>,
);

fn protocols(
    framed: bool,
    read: ReadHalf<TTcpChannel>,
    write: WriteHalf<TTcpChannel>,
    limits: TConfiguration,
) -> Protocols {
    if framed {
        (
            Box::new(TBinaryInputProtocol::with_config(
                TFramedReadTransport::new(read),
                true,
                limits.clone(),
            )),
            Box::new(TBinaryOutputProtocol::with_config(
                TFramedWriteTransport::new(write),
                true,
                limits,
            )),
        )
    } else {
        (
            Box::new(TBinaryInputProtocol::with_config(
                TBufferedReadTransport::new(read),
                true,
                limits.clone(),
            )),
            Box::new(TBinaryOutputProtocol::with_config(
                TBufferedWriteTransport::new(write),
                true,
                limits,
            )),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::thread;

    use thrift::protocol::TMessageType;
    use thrift::{ApplicationError, ApplicationErrorKind};

    use super::super::test_server::fake::{
        cfg, exception, reply_with, serve, strings_field, struct_field, write_table,
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
        assert!(matches!(e, HmsError::Failed(_)), "{e:?}");
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
        assert!(matches!(e, HmsError::Failed(_)), "{e:?}");
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
}
