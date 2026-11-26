---
description: "Run a Spark Connect server inside your Python process and connect stock PySpark to it."
---

# Connect stock PySpark to an in-process server

`krishiv.start_local()` starts a Spark Connect server inside your Python process, on a free port of the loopback addresses, and returns a handle with the connection string. A stock `pyspark-client` connects to it with no JVM and no separate server process. Use it to run code written against the PySpark Connect API on Krishiv in a script, a notebook or a test.

## Start a server and connect

```python
import krishiv
from pyspark.sql import SparkSession  # pip install pyspark-client

with krishiv.start_local() as server:
    spark = SparkSession.builder.remote(server.url).create()
    spark.range(3).show()
    spark.stop()
```

`server.url` is `sc://localhost:<port>/;token=<token>`. The client sends the token with every request (`authorization: Bearer <token>`), and the server refuses a connection that does not carry it. The token is made for each `start_local()` call, so another local process that finds the port cannot use it.

The host name is `localhost` on purpose: `pyspark-client` sends a token over a plain connection only to that name, and wants TLS for any other. The server listens on `127.0.0.1` and `[::1]` (the two addresses `localhost` can mean, so that no other process can hold one of them and receive the token) and on no other address.

`spark.sql(...)` through the client works, and so does the DataFrame API (`range`, `select`, `join`, `groupBy`, `agg`).

## Which clients connect

`start_local()` is tested with `pyspark-client` 4.1 and later. A PySpark 3.5 client forces TLS when the connection string carries a token, so it fails the handshake against `start_local()`, which has no TLS. To use a 3.5 client, run `krishiv server --connect-port <port>` on loopback without a token and connect to `sc://localhost:<port>`; that combination returns the right results (`version`, `range(5).collect()`, `toPandas()`). Do not run a tokenless server on an address other machines can reach: see [Securing a single-node server](single-node-server-security.md).

## Stop it

`server.stop()` closes the listener and waits for the server's thread, so the port is free when it returns. Leaving the `with` block does the same, and so does the end of the interpreter if you forget. Stopping a stopped server does nothing.

## What it is, and what it is not

- The server's sessions start from a new Krishiv session, separate from the one `krishiv.session()` returns: tables and views you created through the embedded API are not visible to the client. Pass `krishiv.start_local({"krishiv.some.key": "value"})` to set `krishiv.*` configuration for its sessions.
- It is the same Connect server as `krishiv server`, with one fixed token and no TLS, for one process. For a server other machines use, see [Securing a single-node server](single-node-server-security.md).
- It does not implement `SparkSession.builder.remote("local")`. In Spark that starts a JVM and needs the full `pyspark` package; Krishiv has no JVM. See [known differences](../concepts/known-differences.md).
