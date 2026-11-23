<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api, type CacheRow, type MemoryRow, type UdfPool, type UdfWorkerOwner } from "../api";
  import { bytes, count, percent } from "../format";
  import { consoleInfo } from "../console.svelte";
  import ResponsiveTable, { type Column } from "$lib/components/responsive-table.svelte";
  import { keyOf } from "$lib/utils";

  const q = createQuery<{ memory: MemoryRow[]; caches: CacheRow[] }>(() => ({
    queryKey: ["resources"],
    queryFn: () => api("/api/internal/resources"),
    refetchInterval: 2000,
  }));

  // The worker pool is a server's: the other consoles have none and do not ask.
  const workers = createQuery<{ pool: UdfPool | null }>(() => ({
    queryKey: ["udf-workers"],
    queryFn: () => api("/api/internal/udf-workers"),
    refetchInterval: 2000,
    enabled: consoleInfo.mode === "server",
  }));
  const pool = $derived(workers.data?.pool ?? null);

  const workerColumns: Column[] = [
    { key: "session", label: "Session", title: true },
    { key: "principal", label: "Principal" },
    { key: "tenant", label: "Tenant" },
    { key: "live", label: "Workers", num: true },
    { key: "busy", label: "Busy", num: true },
  ];
  function workerCell(o: UdfWorkerOwner, key: string): string {
    switch (key) {
      case "session": return o.session_id;
      case "principal": return o.principal;
      case "tenant": return o.tenant ?? "–";
      case "live": return count(o.live);
      default: return count(o.busy);
    }
  }

  const memoryColumns: Column[] = [
    { key: "path", label: "Pool", title: true },
    { key: "kind", label: "Kind" },
    { key: "reserved", label: "In use", num: true },
    { key: "peak", label: "Peak", num: true },
    { key: "limit", label: "Limit", num: true },
    { key: "spillable", label: "Spillable" },
  ];
  const cacheColumns: Column[] = [
    { key: "cache", label: "Cache", title: true },
    { key: "entries", label: "Entries", num: true },
    { key: "bytes", label: "Size", num: true },
    { key: "capacity", label: "Capacity", num: true },
    { key: "hits", label: "Hits", num: true },
    { key: "misses", label: "Misses", num: true },
    { key: "rate", label: "Hit rate", num: true },
  ];

  function memoryCell(r: MemoryRow, key: string): string {
    switch (key) {
      case "path": return r.path;
      case "kind": return r.kind;
      case "reserved": return bytes(r.reserved_bytes);
      case "peak": return bytes(r.peak_bytes);
      case "limit": return bytes(r.limit_bytes);
      default: return r.spillable ? "yes" : "no";
    }
  }
  function cacheCell(c: CacheRow, key: string): string {
    switch (key) {
      case "cache": return c.cache;
      case "entries": return count(c.entries);
      case "bytes": return bytes(c.bytes);
      case "capacity": return bytes(c.capacity_bytes);
      case "hits": return count(c.hits);
      case "misses": return count(c.misses);
      default: return percent(c.hit_rate);
    }
  }
</script>

<h1 class="mb-4">Resources</h1>
{#if consoleInfo.mode === "embedded"}
  <p class="mb-3 text-muted-foreground">Memory pools and caches of this process; query pools are this session's.</p>
{/if}
{#if q.isPending}
  <p class="text-muted-foreground">Loading…</p>
{:else if q.isError}
  <p class="text-error">Could not load resources: {q.error.message}</p>
{:else}
  <h2 class="mb-3">Memory</h2>
  <ResponsiveTable columns={memoryColumns} rows={q.data.memory} rowKey={keyOf(q.data.memory, (r) => `${r.worker_id ?? ""}/${r.path}`)}>
    {#snippet cell(r, col)}<span class={col.key === "path" ? "mono" : ""}>{memoryCell(r, col.key)}</span>{/snippet}
  </ResponsiveTable>
  <h2 class="mt-6 mb-3">Caches</h2>
  <ResponsiveTable columns={cacheColumns} rows={q.data.caches} rowKey={keyOf(q.data.caches, (c) => `${c.worker_id ?? ""}/${c.cache}`)}>
    {#snippet cell(c, col)}{cacheCell(c, col.key)}{/snippet}
  </ResponsiveTable>
{/if}
{#if workers.isError}
  <h2 class="mt-6 mb-3">Python UDF workers</h2>
  <p class="text-error">Could not load the Python UDF workers: {workers.error.message}</p>
{:else if pool}
  <h2 class="mt-6 mb-3">Python UDF workers</h2>
  <p class="mb-3 text-muted-foreground" data-udf-pool>
    {count(pool.live)} of {count(pool.max_total)} worker processes, {count(pool.busy)} busy. A session runs at most
    {count(pool.max_per_session)}, a tenant (or a principal without one) {count(pool.max_per_tenant)}.
  </p>
  {#if pool.owners.length === 0}
    <p class="text-muted-foreground" data-no-udf-workers>No worker is running: one starts with a session's first Python UDF call.</p>
  {:else}
    <ResponsiveTable columns={workerColumns} rows={pool.owners} rowKey={(o) => `${o.tenant ?? ""}/${o.principal}/${o.session_id}`}>
      {#snippet cell(o, col)}<span class={col.key === "session" ? "mono" : ""}>{workerCell(o, col.key)}</span>{/snippet}
    </ResponsiveTable>
    {#if pool.truncated}
      <p class="mt-2 text-muted-foreground">Only the sessions with the most workers are listed.</p>
    {/if}
  {/if}
{/if}
