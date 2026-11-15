---
description: "Install Krishiv, run your first query and decide where to go next."
---

# Get started

This section is for someone who has not run Krishiv yet. The [Quickstart](quickstart.md) installs the Python package,
starts a session, runs a SQL query and a DataFrame query, and reads the result back as Arrow data. It takes a few
minutes and needs Python 3.11 or newer.

Once it runs, pick the next step by what you came to do:

- Moving an existing Spark job: [migrate a PySpark job](../guides/pyspark-migration.md), and check its functions
  against the [compatibility data](../concepts/compatibility.md).
- Trying it on sample data: the [notebooks and datasets](../guides/notebooks-and-datasets.md).
- Running it somewhere other than your Python process: [deployment](../guides/deploy.md).

## Supported platforms

- **Python:** 3.11 or newer.
- **Linux:** x86-64 and aarch64 with glibc 2.28 or newer (`manylinux_2_28`), and x86-64 on musl (`musllinux_1_2`,
  for example Alpine).
- **macOS:** arm64 (Apple silicon).
- **Windows:** there is no Windows wheel; use WSL2 and install the Linux wheel.
- **Older x86-64 CPUs** (no AVX2): install `krishiv-compat` instead of `krishiv`. It is the same package and imports
  as `krishiv`; install one of the two, not both. It covers glibc Linux x86-64 only (no musl or arm64 build).
