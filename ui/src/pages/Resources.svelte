<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api, type CacheRow, type MemoryRow } from "../api";
  import { bytes, count, percent } from "../format";
  import ResponsiveTable, { type Column } from "$lib/components/responsive-table.svelte";

  const q = createQuery<{ memory: MemoryRow[]; caches: CacheRow[] }>(() => ({
    queryKey: ["resources"],
    queryFn: () => api("/api/internal/resources"),
    refetchInterval: 2000,
  }));

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
{#if q.isPending}
  <p class="text-muted-foreground">Loading…</p>
{:else if q.isError}
  <p class="text-error">Could not load resources: {q.error.message}</p>
{:else}
  <h2 class="mb-3">Memory</h2>
  <ResponsiveTable columns={memoryColumns} rows={q.data.memory} rowKey={(r) => r.path}>
    {#snippet cell(r, col)}<span class={col.key === "path" ? "mono" : ""}>{memoryCell(r, col.key)}</span>{/snippet}
  </ResponsiveTable>
  <h2 class="mt-6 mb-3">Caches</h2>
  <ResponsiveTable columns={cacheColumns} rows={q.data.caches} rowKey={(c) => c.cache}>
    {#snippet cell(c, col)}{cacheCell(c, col.key)}{/snippet}
  </ResponsiveTable>
{/if}
