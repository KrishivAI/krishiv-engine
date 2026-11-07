<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api, type CacheRow, type Health } from "../api";
  import { bytes, count, percent, seconds } from "../format";
  import { Card, CardLabel, CardValue } from "$lib/components/ui/card";
  import ResponsiveTable, { type Column } from "$lib/components/responsive-table.svelte";

  const q = createQuery<Health>(() => ({
    queryKey: ["health"],
    queryFn: () => api<Health>("/api/internal/health"),
    refetchInterval: 2000,
  }));

  const columns: Column[] = [
    { key: "cache", label: "Cache", title: true },
    { key: "entries", label: "Entries", num: true },
    { key: "bytes", label: "Size", num: true },
    { key: "hit_rate", label: "Hit rate", num: true },
  ];
  function cacheCell(c: CacheRow, key: string): string {
    if (key === "cache") return c.cache;
    if (key === "entries") return count(c.entries);
    if (key === "bytes") return bytes(c.bytes);
    return percent(c.hit_rate);
  }
</script>

<h1 class="mb-4">Health</h1>
{#if q.isPending}
  <p class="text-muted-foreground">Loading…</p>
{:else if q.isError}
  <p class="text-error">Could not load health: {q.error.message}</p>
{:else}
  {@const h = q.data}
  <div class="grid grid-cols-[repeat(auto-fit,minmax(min(100%,11rem),1fr))] gap-3">
    <Card><CardLabel>Version</CardLabel><CardValue>{h.version}</CardValue></Card>
    <Card><CardLabel>Uptime</CardLabel><CardValue>{seconds(h.uptime_seconds)}</CardValue></Card>
    <Card><CardLabel>Workers</CardLabel><CardValue>{h.workers}</CardValue></Card>
    <Card><CardLabel>Queries</CardLabel><CardValue>{h.running_queries} running · {h.queued_queries} queued</CardValue></Card>
    <Card>
      <CardLabel>Memory in use</CardLabel>
      <CardValue>{bytes(h.memory?.reserved_bytes)}{#if h.memory?.limit_bytes} / {bytes(h.memory.limit_bytes)}{/if}</CardValue>
    </Card>
    <Card><CardLabel>Spilled (all queries)</CardLabel><CardValue>{bytes(h.spill_bytes_total)}</CardValue></Card>
  </div>
  <h2 class="mt-6 mb-3">Caches</h2>
  <ResponsiveTable {columns} rows={h.caches} rowKey={(c) => `${c.worker_id ?? ""}/${c.cache}`} empty="No caches yet.">
    {#snippet cell(c, col)}{cacheCell(c, col.key)}{/snippet}
  </ResponsiveTable>
{/if}
