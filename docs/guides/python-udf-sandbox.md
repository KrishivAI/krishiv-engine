---
description: "How Python UDFs run in a single-node Krishiv server: sandboxed worker processes, what they can and cannot do, and the settings that control them."
---

# Python UDFs in the server: the sandbox

> The single-node server is not part of 0.2.0. This page describes how it runs Python UDFs.

There are two ways to run a Python UDF in Krishiv, and they are protected differently.

- **In your own process** (`@udf`, `pandas_udf` and `arrow_udf` in the Python package). The function runs in the
  process that calls it, with that process's permissions. It is not sandboxed. Use it for your own code.
- **In a single-node server** (`krishiv server`, Spark Connect clients). The function a client sends runs in a
  worker process that the server starts for it. The worker is sandboxed, so client code cannot reach the server's
  memory, its network or its files outside the allowed paths.

This page covers the second case.

## What runs in a worker

Each worker belongs to one client session and one principal. The server never runs client Python itself. A worker
starts only when a UDF needs one, is reused for the next call of the same session, and stops after
`krishiv.python.worker.idleTimeout` without use or when the session ends.

In this release, the server runs scalar Python UDFs in workers. Other UDF forms are not covered yet.

## Requirements

- Linux. The sandbox uses Landlock and seccomp, which are Linux features. On a kernel without them, a server with
  `krishiv.udf.sandbox=on` refuses the UDF instead of running it unprotected.
- Each worker interpreter (`krishiv.python.worker.executables`) needs Python 3.11 or later, `pyspark-client`
  (or `pyspark`) of the same version as your client, `pyarrow` 18.0.0 or later (also with a PySpark 3.5 or 4.0
  client, whose own minimum is lower) and `pandas`. The worker does not need the `krishiv` package.

## What the sandbox allows and blocks

| Blocked | Allowed |
|---|---|
| Network sockets. A worker has no network access. | Reading and writing in the worker's own private directory. |
| Running other programs (`exec`), except the worker's own interpreter. | Reading the Python libraries and files the worker needs to run. |
| Reading or writing files outside the allowed paths. | Threads. |
| Debugging or inspecting other processes (`ptrace`), and signalling them. | Memory, CPU and file-size limits, enforced per worker. |
| Mounting filesystems, namespaces, kernel modules and similar system calls. | |
| Forking. Blocked by default; see `krishiv.python.worker.sandbox.allowFork` below. | |

A UDF that breaks a limit gets an error, and the worker is replaced. The server keeps running.

## Errors you may see

| Error | Meaning |
|---|---|
| `PYTHON_EXCEPTION` | Your function raised an exception. The message includes the Python traceback. The worker is reused. |
| `KRISHIV_PYTHON_WORKER_EXITED` | The worker stopped unexpectedly, through a crash, a signal, a memory limit or a protocol error. The message gives the exit code or signal and the last lines of the worker's stderr. |
| `KRISHIV_UDF_SANDBOX_UNAVAILABLE` | The sandbox is on, but this system cannot enforce it. |
| `PYTHON_VERSION_MISMATCH` | The client's PySpark version does not match the worker's. |

## Settings

| Setting | Default | What it does |
|---|---|---|
| `krishiv.udf.sandbox` | `on` | Set to `off` to run workers without the OS sandbox. Use this only on a trusted machine. |
| `krishiv.python.worker.executables` | `python3` | Interpreters the workers may run. Absolute paths, or names found on the server's `PATH` at start-up. |
| `krishiv.python.worker.idleTimeout` | `1min` | How long an idle worker is kept for the next call of its session. |
| `krishiv.python.worker.maxPerSession` | `8` | Most workers one session runs at once. More calls wait. |
| `krishiv.python.worker.maxTotal` | `64` | Most workers the server runs over all sessions. |
| `krishiv.python.worker.maxPerTenant` | `32` | Most workers, and most running calls, of one tenant over all its sessions. More calls wait; one tenant cannot take every worker or call thread. Sessions without a tenant count per principal; with authentication off (one local principal) there is no cap, only `maxTotal` and `maxPerSession`. |
| `krishiv.python.worker.memoryLimit` | `1GB` | Address-space limit per worker. A UDF that allocates past it gets a `MemoryError`, or its worker ends. |
| `krishiv.python.worker.tenantMemoryLimit` | unset | Address-space limit per worker of a tenant, when lower than `memoryLimit`. |
| `krishiv.python.worker.sandbox.allowFork` | `false` | Lets a UDF fork (`multiprocessing`, `os.fork`). The other limits still apply to the forked processes. |

All of these are set at start-up. Restart the server after a change.

## Persisting a function

A client can keep a registered Python UDF with the `krishiv.v1.PersistPythonFunction` command (`drop` removes it).
The server advertises the `persistpythonfunction` capability whether or not it has a function catalog; without one
(`python-function-catalog` build feature of `krishiv` and `krishiv.connect.pythonFunctions.catalog`) the command is refused with
`KRISHIV_UNSUPPORTED_FEATURE`. Stored functions are per tenant (an untenanted principal is its own key); replacing
or dropping another principal's function needs `Manage` on the cluster.
