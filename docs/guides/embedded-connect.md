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

## Stop it

`server.stop()` closes the listener and waits for the server's thread, so the port is free when it returns. Leaving the `with` block does the same, and so does the end of the interpreter if you forget. Stopping a stopped server does nothing.

## Plotting

`DataFrame.plot` (Spark 4.0+, needs `plotly` and `pandas` in the client) runs in the client against Krishiv's server. `line`, `bar`, `barh`, `scatter`, `area`, `pie` and `kde` work. The client reads `spark.sql.pyspark.plotting.max_rows` (default 1000; the Krishiv key is `krishiv.sql.pyspark.plotting.max_rows`) and shows at most that many rows: a top-n plot (`bar`, `barh`, `pie`) fetches one row more to tell whether it was cut, then trims to the first rows up to the limit; a sampled plot (`line`, `scatter`, `area`) takes a random sample of that size, not the first rows; `kde` does not apply the setting. `hist` and `box` are refused for now: `hist` needs a generator (`posexplode`) in a Connect select list with two output names, and the internal function `array_binary_search`; `box` sends a `WithRelations` plan.

## What it is, and what it is not

- The server's sessions start from a new Krishiv session, separate from the one `krishiv.session()` returns: tables and views you created through the embedded API are not visible to the client. Pass `krishiv.start_local({"krishiv.some.key": "value"})` to set `krishiv.*` configuration for its sessions.
- Python UDFs (`F.udf`, `F.pandas_udf`, `mapInPandas`) run on Linux, in sandboxed worker processes of the interpreter that called `start_local()`; that interpreter needs `pyarrow` and `pyspark-client`. Keys: `krishiv.python.worker.executables` (another interpreter), `krishiv.python.worker.memoryLimit` (address space per worker, 1 GiB; raise it, for example `"4g"`, when a worker cannot import `pyarrow`, as on free-threaded Python) and `krishiv.udf.sandbox=off`. On other platforms a Python UDF is refused with `KRISHIV_UNSUPPORTED_FEATURE`.
- It is the same Connect server as `krishiv server`, with one fixed token and no TLS, for one process. For a server other machines use, see [Securing a single-node server](single-node-server-security.md).
- It does not implement `SparkSession.builder.remote("local")`. In Spark that starts a JVM and needs the full `pyspark` package; Krishiv has no JVM. See [known differences](../concepts/known-differences.md).
