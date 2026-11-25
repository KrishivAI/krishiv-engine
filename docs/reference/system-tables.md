---
description: "The system catalog: tables that show a session's queries, sessions, caches, memory, metrics and resource groups, and the capabilities of the server."
---

# System tables

`system` is Krishiv's own catalog of introspection tables. Query them with SQL, for example
`SELECT * FROM system.queries`. `SHOW CATALOGS` lists it next to `spark_catalog`; Spark has no such catalog (see
[known differences](../concepts/known-differences.md)).

Stability: experimental since 0.3 for the tables and their columns. A minor release may add or rename a column.
`system.queries` and `system.sessions` are the candidates to become stable in 0.4. The exception is
[`system.capabilities`](#systemcapabilities), below.

| Table | Rows |
|---|---|
| `system.queries` | one per query: its session, principal, tags, state, text, timings and usage |
| `system.sessions` | one per client session, with its principal, client type, state and running queries |
| `system.operators` | one per operator of a query, with its metrics |
| `system.catalogs` | the catalogs the caller may see |
| `system.memory` | the memory pool and its reservations |
| `system.metrics` | the engine's metrics, as rows |
| `system.caches` | cached relations and checkpoints of the session |
| `system.resource_groups` | each resource group's limits and current load ([Resource groups](../guides/resource-groups.md)) |
| `system.partition_advice` | the partition column the query history suggests for each table |
| `system.capabilities` | what this build and server offer |

Run `DESCRIBE system.<table>` for the columns of your version.

## `system.capabilities`

Stability: stable since 0.3 for the row shape and the `surface` rows. The set of `feature` row names is experimental, because it follows the build's Cargo features.

Every row is `(kind, name, value, detail)`, all text:

- `surface` rows name an interface the server offers and the revision it speaks, for example `system_tables` and the
  admin API ([Admin API](admin-api.md)). A client reads these to find out which interfaces it can use.
- `feature` rows name a compiled-in feature (for example `cloud`) and whether it is on.

The admin API's `GetCapabilities` returns the same rows.
