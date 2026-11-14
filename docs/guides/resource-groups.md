---
description: "Route queries into resource groups with concurrency, queue, memory and scan limits; tag requests and cancel by tag."
---

# Resource groups and request tags

A single-node Krishiv server can share its engine between several clients. Resource groups let you decide how
much of the engine each kind of query may use: how many run at once, how many wait, how much memory and scan each
one gets, and which query is stopped first when memory runs out.

Resource groups are configured for the server as a whole, so the limits hold across every session on it.

## Configure the groups

Point `krishiv.workload.resourceGroups.file` at a JSON file. The shape follows Trino's `resource-groups.json`,
restricted to the subset Krishiv supports. A complete example is in
[`docs/examples/resource-groups.json`](https://github.com/KrishivAI/krishiv-engine/blob/main/docs/examples/resource-groups.json):

```json
{
  "rootGroups": [
    {
      "name": "root",
      "hardConcurrencyLimit": 16,
      "maxQueued": 200,
      "schedulingPolicy": "weighted_fair",
      "subGroups": [
        {"name": "interactive", "hardConcurrencyLimit": 16, "maxQueued": 50, "schedulingWeight": 3,
         "priority": 80, "workloadClass": "interactive"},
        {"name": "batch", "hardConcurrencyLimit": 6, "maxQueued": 100, "schedulingWeight": 1,
         "priority": 30, "workloadClass": "batch"}
      ]
    }
  ],
  "selectors": [
    {"clientTags": ["background"], "group": "root.background"},
    {"queryType": "WRITE", "group": "root.batch"}
  ],
  "defaultGroup": "root.interactive"
}
```

What each field controls:

| Field | Meaning |
|---|---|
| `hardConcurrencyLimit` | Most queries of the group that run at once. |
| `maxQueued` | Most queries of the group that may wait for a slot. |
| `schedulingPolicy`, `schedulingWeight` | How queued queries of sibling groups share the slots. |
| `priority` | Which query is aborted first when the memory pool is exhausted: the lowest priority goes first. |
| `perQueryMemoryLimit` | Memory one query of the group may hold, as a size or a percentage of the pool. |
| `workloadClass` | The class the group's queries run as, for example `interactive`, `batch`, `background` or `agent`. |
| `selectors` | Rules that route a query to a group, by `source` (a pattern), by `clientTags` or by `queryType`. |
| `defaultGroup` | The group for a query no selector matches. |

If the file is missing or does not parse, the server does not start. The error is
`KRISHIV_RESOURCE_GROUP_CONFIG_INVALID`, and it names the problem.

This setting is new in the release after 0.2.0.

## What happens under memory pressure

When the memory pool is full, the server aborts queries in this order: the lowest `priority` first, then the query
furthest above its share of the pool. If every group uses the default priority, the second rule alone decides, as
it did before resource groups.

## See the queue

`system.queries` shows each query's group and wait:

```sql
SELECT query_id, resource_group, queued_ns, status
FROM system.queries
ORDER BY queued_ns DESC;
```

`resource_group` is the group the query ran in. `queued_ns` is the time it waited for a slot, in nanoseconds.

## Tag requests and cancel by tag

A client can tag a query when it starts it, so that a group of related queries can be found and stopped together.
The Rust API takes tags in `QueryOptions::tags`. The tags are recorded in `system.queries` and in the admission
request, so a query waiting in a queue can be cancelled by its tag too.

- `Session::interrupt_tag(tag)` cancels the session's own queries with an exact tag match.
- `EngineRuntime::cancel_by_tag(selector, caller, all_principals)` cancels across sessions. A user cancels their
  own queries; cancelling another principal's queries needs the admin role. It returns the ids it cancelled.

A tag such as `job=42` can also select a resource group when the query arrives through Spark Connect.
