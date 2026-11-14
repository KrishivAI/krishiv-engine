// SPDX-License-Identifier: Apache-2.0
//! The config entries of `krishiv server`: the Spark Connect listener, its
//! session manager and limits, and shutdown (design P3-T001 §3). All are
//! fixed when the server starts; Spark's names are aliases where Spark has
//! the same setting.

use std::time::Duration;

use super::MIB;
use crate::config::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

const fn startup(
    key: &'static str,
    value_type: ValueType,
    default: ConfigValue,
    doc: &'static str,
    spark_alias: Option<&'static str>,
) -> ConfigEntry {
    ConfigEntry {
        key,
        value_type,
        default: Some(default),
        doc,
        scope: Scope::Cluster,
        settable_by: SettableBy::Startup,
        version_added: "0.3.0",
        spark_alias,
    }
}

/// The Spark Connect port.
pub static CONNECT_PORT: ConfigEntry = startup(
    "krishiv.connect.port",
    ValueType::PositiveInt,
    ConfigValue::Int(15002),
    "The port `krishiv server` serves Spark Connect on (`sc://host:15002`). A port in use fails the start.",
    Some("spark.connect.grpc.binding.port"),
);

/// The Spark Connect bind address.
pub static CONNECT_ADDRESS: ConfigEntry = startup(
    "krishiv.connect.address",
    ValueType::String,
    ConfigValue::String(std::borrow::Cow::Borrowed("127.0.0.1")),
    "The address the Spark Connect listener binds. Loopback by default (Spark binds every \
     interface); a non-loopback address needs authentication configured, or the server refuses \
     to start.",
    Some("spark.connect.grpc.binding.address"),
);

/// The largest Spark Connect request.
pub static CONNECT_MAX_INBOUND_MESSAGE_SIZE: ConfigEntry = startup(
    "krishiv.connect.maxInboundMessageSize",
    ValueType::Size,
    ConfigValue::Size(128 * MIB),
    "The largest Spark Connect request message; a larger one is refused before it is read.",
    Some("spark.connect.grpc.maxInboundMessageSize"),
);

/// Server keepalive pings.
pub static CONNECT_KEEPALIVE_ENABLED: ConfigEntry = startup(
    "krishiv.connect.keepAlive.enabled",
    ValueType::Bool,
    ConfigValue::Bool(true),
    "Whether the server pings idle Spark Connect connections to find dead clients.",
    Some("spark.connect.grpc.keepAlive.enabled"),
);

/// Interval of server keepalive pings.
pub static CONNECT_KEEPALIVE_TIME: ConfigEntry = startup(
    "krishiv.connect.keepAlive.time",
    ValueType::Duration,
    ConfigValue::Duration(Duration::from_secs(60)),
    "How often the server pings an idle Spark Connect connection.",
    Some("spark.connect.grpc.keepAlive.time"),
);

/// Timeout of server keepalive pings.
pub static CONNECT_KEEPALIVE_TIMEOUT: ConfigEntry = startup(
    "krishiv.connect.keepAlive.timeout",
    ValueType::Duration,
    ConfigValue::Duration(Duration::from_secs(20)),
    "How long the server waits for a keepalive answer before it closes the connection.",
    Some("spark.connect.grpc.keepAlive.timeout"),
);

/// Idle timeout of a Connect session.
pub static CONNECT_SESSION_IDLE_TIMEOUT: ConfigEntry = startup(
    "krishiv.connect.session.idleTimeout",
    ValueType::Duration,
    ConfigValue::Duration(Duration::from_secs(60 * 60)),
    "A Spark Connect session no RPC used for this long is released (its id then answers \
     `INVALID_HANDLE.SESSION_CLOSED`); a session running a query is kept. `0s` never expires \
     (Spark's `spark.connect.session.manager.defaultSessionTimeout`, whose `-1` this is).",
    None,
);

/// How often idle sessions are swept.
pub static CONNECT_SESSION_MAINTENANCE_INTERVAL: ConfigEntry = startup(
    "krishiv.connect.session.maintenanceInterval",
    ValueType::Millis,
    ConfigValue::Duration(Duration::from_secs(30)),
    "How often the server looks for idle Spark Connect sessions to release.",
    Some("spark.connect.session.manager.maintenanceInterval"),
);

/// Released session ids remembered.
pub static CONNECT_SESSION_TOMBSTONES: ConfigEntry = startup(
    "krishiv.connect.session.closedTombstones",
    ValueType::Int,
    ConfigValue::Int(1000),
    "How many released Spark Connect session ids are remembered, so reusing one answers \
     `INVALID_HANDLE.SESSION_CLOSED`; the least recently used is forgotten first.",
    Some("spark.connect.session.manager.closedSessionsTombstonesSize"),
);

/// Live sessions in the server.
pub static CONNECT_MAX_SESSIONS: ConfigEntry = startup(
    "krishiv.connect.maxSessions",
    ValueType::PositiveInt,
    ConfigValue::Int(1000),
    "Most live Spark Connect sessions in the server; a new one past it is refused with \
     `KRISHIV_SESSION_LIMIT`.",
    None,
);

/// Live sessions of one principal.
pub static CONNECT_MAX_SESSIONS_PER_PRINCIPAL: ConfigEntry = startup(
    "krishiv.connect.maxSessionsPerPrincipal",
    ValueType::PositiveInt,
    ConfigValue::Int(100),
    "Most live Spark Connect sessions of one principal; a new one past it is refused with \
     `KRISHIV_SESSION_LIMIT`.",
    None,
);

/// Live operations of one session.
pub static CONNECT_MAX_OPERATIONS_PER_SESSION: ConfigEntry = startup(
    "krishiv.connect.maxOperationsPerSession",
    ValueType::PositiveInt,
    ConfigValue::Int(100),
    "Most live Spark Connect operations (`ExecutePlan` calls not yet finished or released) of \
     one session; a new one past it is refused with `KRISHIV_OPERATION_LIMIT`.",
    None,
);

/// Encoded responses an operation may hold ahead of its client.
pub static CONNECT_EXECUTE_AHEAD_BYTES: ConfigEntry = startup(
    "krishiv.connect.execute.aheadBytes",
    ValueType::Size,
    ConfigValue::Size(64 * MIB),
    "Most bytes of encoded `ExecutePlan` responses one operation holds before its client reads \
     them; past it the query pauses until the client catches up (the gRPC flow-control window \
     is the backpressure), so no result is held in full. The bytes count against the memory \
     pool.",
    None,
);

/// Open Connect connections.
pub static CONNECT_MAX_CONNECTIONS: ConfigEntry = startup(
    "krishiv.connect.maxConnections",
    ValueType::PositiveInt,
    ConfigValue::Int(1024),
    "Most open Spark Connect connections; past it the server accepts no new one until one closes.",
    None,
);

/// Deadline for a new connection's first byte.
pub static CONNECT_HANDSHAKE_TIMEOUT: ConfigEntry = startup(
    "krishiv.connect.handshakeTimeout",
    ValueType::Duration,
    ConfigValue::Duration(Duration::from_secs(10)),
    "A new Spark Connect connection that sends no first byte for this long is closed, so silent \
     sockets cannot hold `krishiv.connect.maxConnections` for longer; with TLS the whole handshake \
     must also finish within it. `0s` means no deadline for either, so a client that stalls \
     mid-handshake holds its connection slot until it disconnects.",
    None,
);

/// HTTP/2 streams per connection.
pub static CONNECT_HTTP2_MAX_CONCURRENT_STREAMS: ConfigEntry = startup(
    "krishiv.connect.http2.maxConcurrentStreams",
    ValueType::PositiveInt,
    ConfigValue::Int(100),
    "Most concurrent RPCs (HTTP/2 streams) on one Spark Connect connection; more are refused.",
    None,
);

/// The largest HTTP/2 header list.
pub static CONNECT_HTTP2_MAX_HEADER_LIST_BYTES: ConfigEntry = startup(
    "krishiv.connect.http2.maxHeaderListBytes",
    ValueType::Size,
    ConfigValue::Size(16 * 1024),
    "The largest header list of a Spark Connect request.",
    None,
);

/// Pending reset streams before GOAWAY.
pub static CONNECT_HTTP2_MAX_PENDING_ACCEPT_RESETS: ConfigEntry = startup(
    "krishiv.connect.http2.maxPendingAcceptResets",
    ValueType::PositiveInt,
    ConfigValue::Int(20),
    "Streams a client may open and reset before the server accepts them, before the connection \
     is closed (the HTTP/2 rapid-reset bound).",
    None,
);

/// Shutdown grace period.
pub static SERVER_SHUTDOWN_TIMEOUT: ConfigEntry = startup(
    "krishiv.server.shutdownTimeout",
    ValueType::Duration,
    ConfigValue::Duration(Duration::from_secs(30)),
    "On SIGTERM or SIGINT, how long running queries may finish before every session is closed \
     and its queries cancelled. New requests are refused at once.",
    None,
);

/// Core dumps of the server process.
pub static SERVER_ALLOW_CORE_DUMPS: ConfigEntry = startup(
    "krishiv.server.allowCoreDumps",
    ValueType::Bool,
    ConfigValue::Bool(false),
    "Whether `krishiv server` may write a core dump. Off by default: a dump holds query data and \
     credentials.",
    None,
);

/// `/metrics` without the admin token.
pub static SERVER_METRICS_PUBLIC: ConfigEntry = startup(
    "krishiv.server.metrics.public",
    ValueType::Bool,
    ConfigValue::Bool(false),
    "Whether the admin server's `/metrics` answers without the admin token. Only on a loopback \
     address; aggregate counters only.",
    None,
);
