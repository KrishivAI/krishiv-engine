---
title: Arrow Flight SQL
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
jdbc:arrow-flight-sql://host:15002?useEncryption=false&token=<token>
jdbc:arrow-flight-sql://host:15002?useEncryption=false&user=<name>&password=<token>
```

A bearer `token` and HTTP Basic authentication (`user` and `password`, where the password is a token) both work, with
the same tokens and TLS settings as the rest of the server (see
[Securing a single-node server](single-node-server-security.md)). Use `useEncryption=true` when the server has TLS.

From Python with ADBC:

```python
import adbc_driver_flightsql.dbapi as flight

with flight.connect("grpc://localhost:15002", db_kwargs={"username": "me", "password": "<token>"}) as conn:
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
  keeps its session cookie keeps its session settings.

## Limits

These keep one client from holding the server's memory. Every one is a startup setting; a request over a limit fails
with a Flight status that names the setting.

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
