---
description: "Krishiv is a Rust-native, Spark-compatible engine for batch SQL and DataFrames, in process and with no JVM."
---

# Krishiv documentation

Krishiv is a Rust-native, Apache Spark-compatible compute engine for batch SQL and DataFrames. It runs in your
process, on one vectorized, Arrow-native engine, with no JVM. Keep the PySpark and Spark SQL you already write.

## What do you want to do?

| I want to | Go to |
|---|---|
| **Install** Krishiv | [Get started](get-started/index.md): `pip install "krishiv[pyarrow]"`, Python 3.11 or newer |
| Run a **first query** | [Quickstart](get-started/quickstart.md): SQL and DataFrames, results as Arrow |
| **Migrate from PySpark** | [Migrate a PySpark job](guides/pyspark-migration.md), then [check a project](guides/check-compat.md) and [known differences](concepts/known-differences.md) |
| Look up **SQL** | [SQL language](reference/sql/index.md) and [functions](reference/sql/functions/index.md) |
| Look up the **Python API** | [Python API](reference/python/index.md) and [Python package](reference/python-package.md) |
| **Deploy** it | [Deployment](guides/deploy.md): embedded, single-node, distributed; then [securing a server](guides/single-node-server-security.md) |
| Run it in **Docker or Kubernetes** | [The memory limit in a container](guides/deploy.md#the-memory-limit-in-a-container) (Krishiv reads the cgroup limit) |
| Work with **Iceberg and Delta** | [Iceberg and Delta tables, and catalogs](guides/lakehouse-tables.md) and [changing rows in a Delta table](guides/delta-dml.md) |
| Read **files, S3, GCS or Azure** | [Data sources](guides/data-sources.md) |
| Get a **quick answer** | [Common questions](get-started/faq.md) |

## The path for a new user

1. [Get started](get-started/index.md): install and run a first query.
2. [Guides](guides/index.md): one task each, from migrating a job to deploying a server.
3. [Reference](reference/index.md): SQL, Python and Rust APIs, every [setting](reference/config/index.md), metrics.
4. [Concepts](concepts/index.md): what is [supported](concepts/compatibility.md) against Spark, the
   [known differences](concepts/known-differences.md) and [API stability](concepts/stability.md).
5. [Releases](releases/index.md): what changed, and what to check before you upgrade.
