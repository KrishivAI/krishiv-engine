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

/// The Spark profile of the server's sessions (ADR 0003 update 2026-11-27, issue S4M-3).
pub static CONNECT_SPARK_PROFILE: ConfigEntry = startup(
    "krishiv.connect.sparkProfile",
    ValueType::Enum(&["auto", "3.5", "4.0", "4.1", "4.2", "4.3"]),
    ConfigValue::String(std::borrow::Cow::Borrowed("auto")),
    "The Spark release whose behaviour the server's sessions follow (the default of \
     `spark.sql.ansi.enabled`, for one). `auto` gives a Spark Connect session the release its \
     client declares when the session is created (a PySpark 3.5 client gets 3.5), and sessions \
     of clients that declare none, Flight SQL sessions and console sessions the newest; a release \
     line (`3.5` to `4.3`) applies to every session.",
    None,
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

/// Sent responses a reattachable operation keeps for a client that
/// reattaches.
pub static CONNECT_EXECUTE_RETRY_BUFFER_SIZE: ConfigEntry = startup(
    "krishiv.connect.execute.retryBufferSize",
    ValueType::Size,
    ConfigValue::Size(10 * MIB),
    "Bytes of already-sent `ExecutePlan` responses a reattachable operation keeps, newest first, so \
     a client that lost its stream can reattach and resume; older ones are dropped. `0` keeps \
     none. The bytes count against the memory pool.",
    Some("spark.connect.execute.reattachable.observerRetryBufferSize"),
);

/// Longest life of one reattachable response stream.
pub static CONNECT_EXECUTE_SENDER_MAX_STREAM_DURATION: ConfigEntry = startup(
    "krishiv.connect.execute.senderMaxStreamDuration",
    ValueType::Millis,
    ConfigValue::Duration(Duration::from_secs(120)),
    "After this long a reattachable `ExecutePlan` or `ReattachExecute` stream ends without \
     `ResultComplete` and the client reattaches to continue. `0` never ends it.",
    Some("spark.connect.execute.reattachable.senderMaxStreamDuration"),
);

/// Most bytes one reattachable response stream sends.
pub static CONNECT_EXECUTE_SENDER_MAX_STREAM_SIZE: ConfigEntry = startup(
    "krishiv.connect.execute.senderMaxStreamSize",
    ValueType::Size,
    ConfigValue::Size(1024 * MIB),
    "After sending this many bytes (the response that crosses it is still sent) a reattachable \
     stream ends without `ResultComplete` and the client reattaches to continue. `0` never ends \
     it.",
    Some("spark.connect.execute.reattachable.senderMaxStreamSize"),
);

/// How long an operation may have no stream attached.
pub static CONNECT_EXECUTE_DETACHED_TIMEOUT: ConfigEntry = startup(
    "krishiv.connect.execute.detachedTimeout",
    ValueType::Millis,
    ConfigValue::Duration(Duration::from_secs(300)),
    "A reattachable operation with no stream attached for this long is cancelled, removed and \
     remembered as abandoned (reattaching then answers `INVALID_HANDLE.OPERATION_ABANDONED`).",
    Some("spark.connect.execute.manager.detachedTimeout"),
);

/// How often detached operations are looked for.
pub static CONNECT_EXECUTE_MAINTENANCE_INTERVAL: ConfigEntry = startup(
    "krishiv.connect.execute.maintenanceInterval",
    ValueType::Millis,
    ConfigValue::Duration(Duration::from_secs(30)),
    "How often the server looks for operations detached past \
     `krishiv.connect.execute.detachedTimeout`.",
    Some("spark.connect.execute.manager.maintenanceInterval"),
);

/// Abandoned operation ids remembered.
pub static CONNECT_EXECUTE_ABANDONED_TOMBSTONES: ConfigEntry = startup(
    "krishiv.connect.execute.abandonedTombstones",
    ValueType::Int,
    ConfigValue::Int(10_000),
    "How many abandoned operation ids are remembered, so reattaching to one answers \
     `INVALID_HANDLE.OPERATION_ABANDONED`; the least recently used is forgotten first.",
    Some("spark.connect.execute.manager.abandonedTombstonesSize"),
);

/// Response bytes all operations of the server hold.
pub static CONNECT_MAX_RESPONSE_BYTES: ConfigEntry = startup(
    "krishiv.connect.maxResponseBytes",
    ValueType::Size,
    ConfigValue::Size(1024 * MIB),
    "Most bytes of encoded `ExecutePlan` responses all operations of the server may hold at \
     once. Each operation reserves its worst case (`krishiv.connect.execute.aheadBytes` plus, \
     reattachable, `krishiv.connect.execute.retryBufferSize`) before its first response and \
     returns it when released, abandoned or ended; past the cap a new operation waits for one \
     to end, a running one never waits. Must be at least that worst case (checked at startup). \
     The reservation assumes responses smaller than the ahead budget: a reattachable operation \
     can hold up to two responses more (the one being sent and the one crossing the retry \
     window), each at most `spark.connect.grpc.arrow.maxBatchSize`; every held response is also \
     reserved from the engine's memory pool, which stays the hard limit.",
    None,
);

/// Bytes of artifacts one Connect session may hold.
pub static CONNECT_ARTIFACTS_MAX_BYTES: ConfigEntry = startup(
    "krishiv.connect.artifacts.maxBytes",
    ValueType::Size,
    ConfigValue::Size(1024 * MIB),
    "Most bytes of artifacts (`AddArtifacts`: Python files, archives, files and cached local \
     relations) one Spark Connect session may hold, on disk and in memory together. A chunk \
     that would pass it is refused before it is written (`KRISHIV_ARTIFACT_LIMIT`, \
     `RESOURCE_EXHAUSTED`) and the artifact it belongs to is discarded. Released with the \
     session.",
    None,
);

/// Bytes of one artifact.
pub static CONNECT_ARTIFACTS_MAX_ARTIFACT_BYTES: ConfigEntry = startup(
    "krishiv.connect.artifacts.maxArtifactBytes",
    ValueType::Size,
    ConfigValue::Size(256 * MIB),
    "Most bytes of one Spark Connect artifact (`AddArtifacts`). A larger artifact is refused \
     (`KRISHIV_ARTIFACT_LIMIT`, `RESOURCE_EXHAUSTED`) before its bytes past the cap are \
     written; what was written is removed.",
    None,
);

/// Bytes of artifacts all Connect sessions together may hold.
pub static CONNECT_ARTIFACTS_MAX_TOTAL_BYTES: ConfigEntry = startup(
    "krishiv.connect.artifacts.maxTotalBytes",
    ValueType::Size,
    ConfigValue::Size(16 * 1024 * MIB),
    "Most bytes of artifacts all Spark Connect sessions of the server may hold together, on \
     disk and in memory (each session is also capped by `krishiv.connect.artifacts.maxBytes`). \
     A chunk that would pass it is refused before it is written (`KRISHIV_ARTIFACT_LIMIT`, \
     `RESOURCE_EXHAUSTED`). Must be at least `krishiv.connect.artifacts.maxArtifactBytes` \
     (checked at startup).",
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
    "On SIGTERM or SIGINT, how long running queries may finish before the rest are cancelled \
     (`KRISHIV_QUERY_KILLED`; the drain waits up to 10 more seconds for them to stop). `/readyz` \
     answers 503 and new queries and sessions are refused at once with \
     `KRISHIV_SERVER_DRAINING`; the admin API's `Drain` uses it as its default grace period. Set \
     a Kubernetes `terminationGracePeriodSeconds` of at least this plus 15 seconds.",
    None,
);

/// Whether SQL text in a served session reads the server's environment through `${env:NAME}` and
/// `${system:NAME}`.
pub static SERVER_SQL_SUBSTITUTE_ENVIRONMENT: ConfigEntry = startup(
    "krishiv.server.sql.substituteEnvironment",
    ValueType::Bool,
    ConfigValue::Bool(false),
    "Whether `${env:NAME}` and `${system:NAME}` in the SQL text of a `krishiv server` session (Spark \
     Connect, Flight SQL, the console's editor) are replaced from the server process's environment, \
     as Spark's variable substitution does. Off by default, a recorded difference from Spark: any \
     signed-in client could otherwise read every secret the process holds (`AWS_SECRET_ACCESS_KEY`, \
     the `env:` references of `krishiv.auth.*` and catalog tokens), so a served session leaves such \
     a reference as written, like an unknown name. `${name}` of a configuration value is substituted \
     as before. Embedded sessions (`krishiv sql`, the Python and Rust APIs) always read the \
     environment. A startup setting: `SET` cannot change it.",
    None,
);

/// How often the server re-runs its readiness checks.
pub static SERVER_READINESS_INTERVAL: ConfigEntry = startup(
    "krishiv.server.readiness.interval",
    ValueType::Duration,
    ConfigValue::Duration(Duration::from_secs(10)),
    "How often `krishiv server` re-runs its readiness checks (object store, authentication). \
     `/readyz` answers from the last result; a result older than three intervals counts as \
     not ready.",
    None,
);

/// Time limit of one readiness check.
pub static SERVER_READINESS_TIMEOUT: ConfigEntry = startup(
    "krishiv.server.readiness.timeout",
    ValueType::Duration,
    ConfigValue::Duration(Duration::from_secs(2)),
    "How long one readiness check may take before it counts as failed.",
    None,
);

/// Consecutive failures before a readiness check fails `/readyz`.
pub static SERVER_READINESS_FAILURE_THRESHOLD: ConfigEntry = startup(
    "krishiv.server.readiness.failureThreshold",
    ValueType::PositiveInt,
    ConfigValue::Int(3),
    "How many consecutive failures of a readiness check make `/readyz` answer 503; one slow \
     moment is not a failure. The threshold applies after a check has passed once: a required \
     check that has never passed (authentication keys not yet fetched) fails `/readyz` on its \
     first failure. A success recovers at once.",
    None,
);

/// The readiness checks that can fail `/readyz`.
pub static SERVER_READINESS_REQUIRED: ConfigEntry = startup(
    "krishiv.server.readiness.required",
    ValueType::String,
    ConfigValue::String(std::borrow::Cow::Borrowed("")),
    "Comma-separated readiness checks that fail `/readyz` when they fail, besides the always \
     required `auth`. Only `auth` is wired today; the warehouse object store check follows. Other checks are reported by \
     the admin API only, because every replica shares them and failing them would remove the \
     whole fleet at once.",
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

/// Extra host names the admin listener accepts on a loopback address.
pub static SERVER_ALLOWED_HOSTS: ConfigEntry = startup(
    "krishiv.server.allowedHosts",
    ValueType::String,
    ConfigValue::String(std::borrow::Cow::Borrowed("")),
    "Comma-separated host names (or IP literals) the admin listener accepts in the `Host` header \
     when it is bound to a loopback address, besides `localhost`, `127.0.0.1` and `[::1]`. A \
     request with any other `Host` is refused with 421, which stops a web page from reaching the \
     admin API through DNS rebinding. Unset: only the loopback names.",
    None,
);

/// `krishiv.workload.resourceGroups.file`.
pub static WORKLOAD_RESOURCE_GROUPS_FILE: ConfigEntry = ConfigEntry {
    key: "krishiv.workload.resourceGroups.file",
    value_type: ValueType::String,
    default: None,
    doc: "Path of the resource group file (JSON in the shape of Trino's `resource-groups.json`, \
          restricted to the subset Krishiv supports): the group tree, its concurrency, queue, \
          per-query memory and scan limits, and the selectors that route a query to a group. Read when \
          the engine starts; a malformed file stops the start with \
          `KRISHIV_RESOURCE_GROUP_CONFIG_INVALID`. When unset, every query is admitted at once.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.3.0",
    spark_alias: None,
};

/// `krishiv.flightSql.enabled` (P3-T101).
pub static FLIGHT_SQL_ENABLED: ConfigEntry = startup(
    "krishiv.flightSql.enabled",
    ValueType::Bool,
    ConfigValue::Bool(true),
    "Whether `krishiv server` serves Arrow Flight SQL on its Spark Connect port, when the binary is built with the `flight-sql` feature. Off: the Flight SQL paths answer `UNIMPLEMENTED`.",
    None,
);

/// `krishiv.flightSql.handleTtl` (P3-T101).
pub static FLIGHT_SQL_HANDLE_TTL: ConfigEntry = startup(
    "krishiv.flightSql.handleTtl",
    ValueType::Duration,
    ConfigValue::Duration(Duration::from_secs(600)),
    "How long an Arrow Flight SQL statement or prepared-statement handle stays fetchable after its last use; its session's release ends it sooner.",
    None,
);

/// `krishiv.flightSql.maxHandlesPerSession` (P3-T101).
pub static FLIGHT_SQL_MAX_HANDLES_PER_SESSION: ConfigEntry = startup(
    "krishiv.flightSql.maxHandlesPerSession",
    ValueType::PositiveInt,
    ConfigValue::Int(64),
    "The most Arrow Flight SQL handles (statements and prepared statements) one session holds; past it a new one is `KRISHIV_FLIGHT_SQL_HANDLE_LIMIT`.",
    None,
);

/// `krishiv.flightSql.maxHandlesPerPrincipal` (P3-T101).
pub static FLIGHT_SQL_MAX_HANDLES_PER_PRINCIPAL: ConfigEntry = startup(
    "krishiv.flightSql.maxHandlesPerPrincipal",
    ValueType::PositiveInt,
    ConfigValue::Int(1024),
    "The most Arrow Flight SQL handles one principal holds across its sessions; past it a new one is `KRISHIV_FLIGHT_SQL_HANDLE_LIMIT`.",
    None,
);

/// `krishiv.flightSql.maxBatchRows` (P3-T101).
pub static FLIGHT_SQL_MAX_BATCH_ROWS: ConfigEntry = startup(
    "krishiv.flightSql.maxBatchRows",
    ValueType::PositiveInt,
    ConfigValue::Int(10000),
    "The most parameter rows one Arrow Flight SQL prepared update binds; more is `INVALID_ARGUMENT`.",
    None,
);

/// `krishiv.flightSql.maxSchemaTables` (P3-T101).
pub static FLIGHT_SQL_MAX_SCHEMA_TABLES: ConfigEntry = startup(
    "krishiv.flightSql.maxSchemaTables",
    ValueType::PositiveInt,
    ConfigValue::Int(1000),
    "The most tables an Arrow Flight SQL `GetTables` with `include_schema` loads schemas for; a listing over it is `INVALID_ARGUMENT` (narrow the pattern).",
    None,
);

/// `krishiv.flightSql.metadataTimeout` (P3-T101).
pub static FLIGHT_SQL_METADATA_TIMEOUT: ConfigEntry = startup(
    "krishiv.flightSql.metadataTimeout",
    ValueType::Duration,
    ConfigValue::Duration(Duration::from_secs(60)),
    "The time limit of one Arrow Flight SQL metadata listing (`GetTables` and the others); past it the call is `DEADLINE_EXCEEDED`.",
    None,
);

/// `krishiv.flightSql.provisionalTtl` (P3-T101).
pub static FLIGHT_SQL_PROVISIONAL_TTL: ConfigEntry = startup(
    "krishiv.flightSql.provisionalTtl",
    ValueType::Duration,
    ConfigValue::Duration(Duration::from_secs(60)),
    "How long an Arrow Flight SQL session made for a call without a session cookie lives after its last call, unless a later call depends on its state (`SetSessionOptions`, a prepared statement, a `SET`/`USE`-class statement), in which case it lives as a cookie session does. A fetched cookie-less ticket ends its session at once.",
    None,
);

/// `krishiv.flightSql.fetchIdleTimeout` (P3-T101).
pub static FLIGHT_SQL_FETCH_IDLE_TIMEOUT: ConfigEntry = startup(
    "krishiv.flightSql.fetchIdleTimeout",
    ValueType::Duration,
    ConfigValue::Duration(Duration::from_secs(300)),
    "How long an Arrow Flight SQL result stream may wait for its client to read the next message; past it the query is cancelled and the stream ends `DEADLINE_EXCEEDED` (`KRISHIV_FLIGHT_SQL_FETCH_IDLE`). Time the engine spends computing the next batch does not count.",
    None,
);

/// `krishiv.flightSql.maxStatementBytes` (P3-T101).
pub static FLIGHT_SQL_MAX_STATEMENT_BYTES: ConfigEntry = startup(
    "krishiv.flightSql.maxStatementBytes",
    ValueType::Size,
    ConfigValue::Size(MIB),
    "The longest SQL text an Arrow Flight SQL statement may have; a longer one is `INVALID_ARGUMENT` (`KRISHIV_FLIGHT_SQL_STATEMENT_TOO_LARGE`) before it is parsed.",
    None,
);

/// `krishiv.flightSql.maxCommandResultBytes` (P3-T101).
pub static FLIGHT_SQL_MAX_COMMAND_RESULT_BYTES: ConfigEntry = startup(
    "krishiv.flightSql.maxCommandResultBytes",
    ValueType::Size,
    ConfigValue::Size(16 * MIB),
    "The largest result an Arrow Flight SQL command (a statement that is not a query, such as `SHOW TABLES`) may store for its ticket. The command runs once, at `GetFlightInfo`; a larger result is `KRISHIV_FLIGHT_SQL_RESULT_NOT_STORED`, which says the statement ran.",
    None,
);

/// `krishiv.flightSql.maxHandleBytesPerPrincipal` (P3-T101).
pub static FLIGHT_SQL_MAX_HANDLE_BYTES_PER_PRINCIPAL: ConfigEntry = startup(
    "krishiv.flightSql.maxHandleBytesPerPrincipal",
    ValueType::Size,
    ConfigValue::Size(64 * MIB),
    "The most bytes (SQL text, bound literals, schemas and stored command results) one principal's Arrow Flight SQL handles hold; past it the principal's oldest idle statement tickets are released, and a new handle is `KRISHIV_FLIGHT_SQL_HANDLE_LIMIT` when none is idle.",
    None,
);

/// `krishiv.connect.pythonFunctions.catalog` (P3-T072c).
pub static CONNECT_PYTHON_FUNCTIONS_CATALOG: ConfigEntry = ConfigEntry {
    key: "krishiv.connect.pythonFunctions.catalog",
    value_type: ValueType::String,
    default: None,
    doc: "Path of the SQLite catalog file that keeps the Python UDFs a client persists with the \
          `krishiv.v1.PersistPythonFunction` command, per tenant: later sessions of the same tenant \
          find them, also after a restart. Made when it is not there. Needs a `krishiv` binary built with the \
          `python-function-catalog` feature; when unset, nothing is persisted and the \
          command is refused.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.3.0",
    spark_alias: None,
};
