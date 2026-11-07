<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api, type Compare, type Query } from "../api";
  import { bytes, count, nanos, oneLine, when } from "../format";

  let stateFilter = $state("");
  let principal = $state("");
  let picked = $state<string[]>([]);
  let compared = $state<Compare | null>(null);
  let compareError = $state("");

  const q = createQuery<{ queries: Query[] }>(() => ({
    queryKey: ["history", stateFilter, principal],
    queryFn: () => {
      const p = new URLSearchParams({ limit: "200" });
      if (stateFilter) p.set("state", stateFilter);
      if (principal) p.set("principal", principal);
      return api(`/api/internal/history?${p}`);
    },
    refetchInterval: 5000,
  }));

  function toggle(id: string) {
    picked = picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id].slice(-2);
    compared = null;
  }

  async function compare() {
    compareError = "";
    try {
      compared = await api<Compare>(`/api/internal/compare?a=${encodeURIComponent(picked[0])}&b=${encodeURIComponent(picked[1])}`);
    } catch (e) {
      compareError = (e as Error).message;
    }
  }

  const fields: [string, string, (n: number) => string][] = [
    ["wall_ns", "Wall time", nanos],
    ["cpu_ns", "CPU time", nanos],
    ["peak_memory_bytes", "Peak memory", bytes],
    ["rows_produced", "Rows produced", count],
    ["bytes_read", "Bytes read", bytes],
    ["spill_bytes", "Spilled", bytes],
  ];
  const sign = (n: number) => (n > 0 ? "up" : n < 0 ? "down" : "");
</script>

<h1>History</h1>
<div class="toolbar">
  <label>State
    <select bind:value={stateFilter}>
      <option value="">any</option>
      <option value="finished">finished</option>
      <option value="failed">failed</option>
      <option value="cancelled">cancelled</option>
    </select>
  </label>
  <label>Principal <input bind:value={principal} placeholder="any" size="14" /></label>
  <button disabled={picked.length !== 2} onclick={compare}>Compare the two selected</button>
  {#if compareError}<span class="error">{compareError}</span>{/if}
</div>

{#if compared}
  <h2>Comparison</h2>
  <table>
    <thead><tr><th></th><th class="num">A <span class="mono">{compared.a.query_id.slice(0, 8)}</span></th><th class="num">B <span class="mono">{compared.b.query_id.slice(0, 8)}</span></th><th class="num">B − A</th></tr></thead>
    <tbody>
      {#each fields as [key, label, fmt]}
        {@const a = (compared.a as unknown as Record<string, number | null>)[key]}
        {@const b = (compared.b as unknown as Record<string, number | null>)[key]}
        {@const d = compared.delta[key]}
        <tr>
          <td>{label}</td>
          <td class="num">{a === null || a === undefined ? "–" : fmt(a)}</td>
          <td class="num">{b === null || b === undefined ? "–" : fmt(b)}</td>
          <td class="num delta {d === undefined ? '' : sign(d)}">{d === undefined ? "–" : (d > 0 ? "+" : "") + fmt(d)}</td>
        </tr>
      {/each}
    </tbody>
  </table>
{/if}

{#if q.isPending}
  <p class="muted">Loading…</p>
{:else if q.isError}
  <p class="error">Could not load history: {q.error.message}</p>
{:else}
  <table>
    <thead>
      <tr><th></th><th>Query</th><th>State</th><th>Principal</th><th>Submitted</th><th class="num">Wall time</th><th class="num">Peak memory</th></tr>
    </thead>
    <tbody>
      {#each q.data.queries as r (r.query_id)}
        <tr>
          <td><input type="checkbox" checked={picked.includes(r.query_id)} onchange={() => toggle(r.query_id)} aria-label="select for comparison" /></td>
          <td><a href={`#/queries/${encodeURIComponent(r.query_id)}`}>{oneLine(r.query_text, 90)}</a></td>
          <td><span class="pill {r.state}">{r.state}</span>{#if r.error_class} <span class="muted mono">{r.error_class}</span>{/if}</td>
          <td>{r.principal ?? "–"}</td>
          <td>{when(r.submitted_at)}</td>
          <td class="num">{nanos(r.wall_ns)}</td>
          <td class="num">{bytes(r.peak_memory_bytes)}</td>
        </tr>
      {:else}
        <tr><td colspan="7" class="muted">No finished queries match.</td></tr>
      {/each}
    </tbody>
  </table>
{/if}
