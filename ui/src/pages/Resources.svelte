<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api, type CacheRow, type MemoryRow } from "../api";
  import { bytes, count, percent } from "../format";

  const q = createQuery<{ memory: MemoryRow[]; caches: CacheRow[] }>(() => ({
    queryKey: ["resources"],
    queryFn: () => api("/api/internal/resources"),
    refetchInterval: 2000,
  }));

  function share(r: MemoryRow): number {
    return r.limit_bytes ? Math.min(1, r.reserved_bytes / r.limit_bytes) : 0;
  }
</script>

<h1>Resources</h1>
{#if q.isPending}
  <p class="muted">Loading…</p>
{:else if q.isError}
  <p class="error">Could not load resources: {q.error.message}</p>
{:else}
  <h2>Memory</h2>
  <table>
    <thead>
      <tr><th>Pool</th><th>Kind</th><th class="num">In use</th><th></th><th class="num">Peak</th><th class="num">Limit</th><th>Spillable</th></tr>
    </thead>
    <tbody>
      {#each q.data.memory as r (r.path)}
        <tr>
          <td class="mono">{r.path}</td>
          <td>{r.kind}</td>
          <td class="num">{bytes(r.reserved_bytes)}</td>
          <td>
            {#if r.limit_bytes}
              <div class="bar"><span style:width={`${Math.round(share(r) * 100)}%`}></span></div>
            {/if}
          </td>
          <td class="num">{bytes(r.peak_bytes)}</td>
          <td class="num">{bytes(r.limit_bytes)}</td>
          <td>{r.spillable ? "yes" : "no"}</td>
        </tr>
      {/each}
    </tbody>
  </table>
  <h2>Caches</h2>
  <table>
    <thead>
      <tr><th>Cache</th><th class="num">Entries</th><th class="num">Size</th><th class="num">Capacity</th><th class="num">Hits</th><th class="num">Misses</th><th class="num">Hit rate</th></tr>
    </thead>
    <tbody>
      {#each q.data.caches as c (c.cache)}
        <tr>
          <td>{c.cache}</td>
          <td class="num">{count(c.entries)}</td>
          <td class="num">{bytes(c.bytes)}</td>
          <td class="num">{bytes(c.capacity_bytes)}</td>
          <td class="num">{count(c.hits)}</td>
          <td class="num">{count(c.misses)}</td>
          <td class="num">{percent(c.hit_rate)}</td>
        </tr>
      {/each}
    </tbody>
  </table>
{/if}
