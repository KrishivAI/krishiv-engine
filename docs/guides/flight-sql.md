---
description: "Connect JDBC and ADBC clients to the server over Arrow Flight SQL on the Spark Connect port, and the limits that apply."
---

# Arrow Flight SQL

`krishiv server` also serves [Arrow Flight SQL](https://arrow.apache.org/docs/format/FlightSql.html) on the Spark
Connect port (`krishiv.connect.port`, default `15002`): one listener, two protocols. JDBC, ADBC and any other Flight
SQL client can run SQL and receive Arrow batches without a Spark client. The binary must be built with the
`flight-sql` feature; `krishiv.flightSql.enabled` (default `true`) turns it off, and the Flight SQL paths then answer
`UNIMPLEMENTED`.

## Connect

With the Arrow Flight SQL JDBC driver:

```text
jdbc:arrow-flight-sql://host:15002?useEncryption=true&token=<token>
jdbc:arrow-flight-sql://host:15002?useEncryption=true&user=<principal>&password=<token>
```

A bearer `token` and HTTP Basic authentication (`user` and `password`, where the password is a token) both work, with
the same tokens and TLS settings as the rest of the server (see
[Securing a single-node server](single-node-server-security.md)). `useEncryption=false` sends the token in plaintext: use it only against a local server you trust. With Basic
authentication, `user` must be the principal the token is configured for; a valid token with another user is
rejected.

From Python with ADBC:

```python
import adbc_driver_flightsql.dbapi as flight

with flight.connect("grpc://localhost:15002", db_kwargs={
        "username": "<principal>",
        "password": "<token>",
        "adbc.flight.sql.rpc.with_cookie_middleware": "true",  # keep the session cookie
    }) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 AS one")
        print(cur.fetch_arrow_table())
```

## What is supported

- **Queries and updates**: statements, prepared statements with bound parameters, and updates (`INSERT`, `UPDATE`,
  `DELETE`, DDL) through the same session machinery as Spark Connect, so grants, resource groups and the query
  history apply.
- **Metadata**: catalogs, schemas, tables (with schemas on request), table types, primary and foreign keys, SQL info,
  XDBC type info, and session options.
- **Sessions**: a call with no session gets a short-lived one (`krishiv.flightSql.provisionalTtl`); a client that
  keeps its session cookie keeps its session settings. The ADBC driver does not keep cookies unless its cookie
  middleware option is on (as in the example below), so session settings changed by one statement do not reach the next one without it.

## Limits

These keep one client from holding the server's memory. Every one is a startup setting; a request over a limit is
refused with a Flight status that names the setting. At a handle limit, a new handle can instead succeed by evicting an
idle statement ticket, and the evicted ticket can then no longer be fetched.

| Setting | Default | Limits |
|---|---|---|
| `krishiv.flightSql.maxStatementBytes` | `1MB` | the SQL text of one statement |
| `krishiv.flightSql.maxHandlesPerSession` | `64` | statement and prepared-statement handles of a session |
| `krishiv.flightSql.maxHandlesPerPrincipal` | `1024` | handles of one principal |
| `krishiv.flightSql.maxHandleBytesPerPrincipal` | `64MB` | bytes those handles hold |
| `krishiv.flightSql.handleTtl` | `10min` | how long an unused handle lives |
| `krishiv.flightSql.fetchIdleTimeout` | `5min` | how long a result stream waits for its client |
| `krishiv.flightSql.maxBatchRows` | `10000` | parameter rows of one prepared update |
| `krishiv.flightSql.maxCommandResultBytes` | `16MB` | the result of a command (a statement that is not a query) |
| `krishiv.flightSql.maxSchemaTables` | `1000` | tables a `GetTables` with schemas returns |
| `krishiv.flightSql.metadataTimeout` | `1min` | one metadata listing |

The full list is in the [configuration reference](../reference/config/index.md).
