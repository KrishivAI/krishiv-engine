---
description: "Keep Python UDF workers apart per tenant, bound what one tenant can use, and persist a registered UDF for its tenant."
---

# Python UDFs per tenant

A server that several tenants share runs each tenant's Python UDFs in workers that no other tenant uses, and bounds
how much of the server one tenant can take. This page covers the limits and how to keep a UDF across sessions. It
adds to [Python UDFs in the server: the sandbox](python-udf-sandbox.md).

## Workers are never shared

A worker serves one tenant for its whole life: it never runs a call for another tenant, even an idle one. Sessions
that have no tenant count per principal, so two users without tenants do not share a worker either. With
authentication off (one local principal) there is a single owner.

## Bound one tenant

| Setting | Default | Meaning |
|---|---|---|
| `krishiv.python.worker.maxPerTenant` | `32` | Most workers, and most running calls, of one tenant over all its sessions. More calls wait in line, so one tenant cannot take every worker or call thread. With authentication off there is no such cap, only `maxTotal` and `maxPerSession`. |
| `krishiv.python.worker.tenantMemoryLimit` | unset | Address-space limit of each worker of a tenant, when lower than `krishiv.python.worker.memoryLimit`. |

Both are startup settings: set them on `krishiv server`.

## Keep a function for the tenant

A client can persist a registered Python UDF with the `krishiv.v1.PersistPythonFunction` command (`drop` removes it).
Later sessions of the same tenant find it at their first request, also after a restart, if the principal may run
client code.

The functions live in a SQLite catalog file named by `krishiv.connect.pythonFunctions.catalog`, which needs a
`krishiv` binary built with the `python-function-catalog` feature. Without both, the command is refused with
`KRISHIV_UNSUPPORTED_FEATURE`. The server advertises the `persistpythonfunction` capability either way.

- Only the principal that persisted a function, or an admin (`Manage` on the cluster), can replace or drop it. An
  untenanted principal's functions are its own.
- Opening a catalog file with 0.3 makes it format version 2, which an older build refuses. Keep a copy first if you
  may roll back. A version 1 file is upgraded when opened.
- `CREATE FUNCTION` and `DROP FUNCTION` in SQL do not persist Python UDFs yet.
