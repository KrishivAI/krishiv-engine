<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api, type Health } from "../api";
  import { bytes, count, percent, seconds } from "../format";

  const q = createQuery<Health>(() => ({
    queryKey: ["health"],
    queryFn: () => api<Health>("/api/internal/health"),
    refetchInterval: 2000,
  }));
</script>

<h1>Health</h1>
{#if q.isPending}
  <p class="muted">Loading…</p>
{:else if q.isError}
  <p class="error">Could not load health: {q.error.message}</p>
{:else}
  {@const h = q.data}
  <div class="cards">
    <div class="card"><div class="label">Version</div><div class="value">{h.version}</div></div>
    <div class="card"><div class="label">Uptime</div><div class="value">{seconds(h.uptime_seconds)}</div></div>
    <div class="card"><div class="label">Workers</div><div class="value">{h.workers}</div></div>
    <div class="card">
      <div class="label">Queries</div>
      <div class="value">{h.running_queries} running · {h.queued_queries} queued</div>
    </div>
    <div class="card">
      <div class="label">Memory in use</div>
      <div class="value">
        {bytes(h.memory?.reserved_bytes)}{#if h.memory?.limit_bytes} / {bytes(h.memory.limit_bytes)}{/if}
      </div>
    </div>
    <div class="card"><div class="label">Spilled (all queries)</div><div class="value">{bytes(h.spill_bytes_total)}</div></div>
  </div>
  <h2>Caches</h2>
  <table>
    <thead>
      <tr><th>Cache</th><th class="num">Entries</th><th class="num">Size</th><th class="num">Hit rate</th></tr>
    </thead>
    <tbody>
      {#each h.caches as c}
        <tr>
          <td>{c.cache}</td>
          <td class="num">{count(c.entries)}</td>
          <td class="num">{bytes(c.bytes)}</td>
          <td class="num">{percent(c.hit_rate)}</td>
        </tr>
      {:else}
        <tr><td colspan="4" class="muted">No caches yet.</td></tr>
      {/each}
    </tbody>
  </table>
{/if}
