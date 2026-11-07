<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api, type QueryDetail } from "../api";
  import { bytes, count, nanos, when } from "../format";

  let { id }: { id: string } = $props();

  const q = createQuery<QueryDetail>(() => ({
    queryKey: ["query", id],
    queryFn: () => api<QueryDetail>(`/api/internal/queries/${encodeURIComponent(id)}`),
    refetchInterval: (query) => (["running", "queued"].includes(query.state.data?.query.state ?? "") ? 1000 : false),
  }));

  const maxCompute = $derived(Math.max(1, ...(q.data?.operators ?? []).map((o) => o.compute_ns ?? 0)));
  const hotId = $derived(q.data?.hot_operator?.operator_id);
</script>

<p><a href="#/queries">← Queries</a> · <a href="#/history">History</a></p>
{#if q.isPending}
  <p class="muted">Loading…</p>
{:else if q.isError}
  <h1>Query not found</h1>
  <p class="error">{q.error.message}</p>
{:else}
  {@const d = q.data}
  <h1>Query <span class="mono">{d.query.query_id}</span> <span class="pill {d.query.state}">{d.query.state}</span></h1>
  <p class="mono">{d.query.query_text ?? "(text not recorded)"}{d.query.query_text_truncated ? " …" : ""}</p>
  <div class="cards">
    <div class="card"><div class="label">Submitted</div><div class="value">{when(d.query.submitted_at)}</div></div>
    <div class="card"><div class="label">Wall time</div><div class="value">{nanos(d.query.wall_ns)}</div></div>
    <div class="card"><div class="label">CPU time</div><div class="value">{nanos(d.query.cpu_ns)}</div></div>
    <div class="card"><div class="label">Peak memory</div><div class="value">{bytes(d.query.peak_memory_bytes)}</div></div>
    <div class="card"><div class="label">Rows produced</div><div class="value">{count(d.query.rows_produced)}</div></div>
    <div class="card"><div class="label">Spilled</div><div class="value">{bytes(d.query.spill_bytes)}</div></div>
  </div>
  {#if d.query.error_class}
    <p class="error">Failed with <code>{d.query.error_class}</code></p>
  {/if}

  {#if d.warnings.length}
    <h2>Warnings</h2>
    {#each d.warnings as w}
      <div class="warning {w.kind}"><strong>{w.operator}</strong> · {w.kind}<br />{w.hint}</div>
    {/each}
  {/if}

  <h2>Operators</h2>
  <p class="muted">
    Compute time per operator; the busiest one is highlighted{d.hot_operator ? `: ${d.hot_operator.operator}` : ""}.
  </p>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Operator</th><th class="num">Rows in</th><th class="num">Rows out</th>
        <th class="num">Compute</th><th></th><th class="num">Waiting (io / memory / sink)</th><th class="num">Spilled</th>
      </tr>
    </thead>
    <tbody>
      {#each d.operators as o (o.operator_id)}
        <tr class:hot={o.operator_id === hotId}>
          <td>{o.operator_id}</td>
          <td>{o.name}{#if o.drivers && o.drivers > 1} <span class="muted">×{o.drivers}</span>{/if}</td>
          <td class="num">{count(o.rows_in)}</td>
          <td class="num">{count(o.rows_out)}</td>
          <td class="num">{nanos(o.compute_ns)}</td>
          <td>
            <div class="bar" class:hot={o.operator_id === hotId}>
              <span style:width={`${Math.round(((o.compute_ns ?? 0) / maxCompute) * 100)}%`}></span>
            </div>
          </td>
          <td class="num">{nanos(o.blocked_io_ns)} / {nanos(o.blocked_memory_ns)} / {nanos(o.blocked_sink_ns)}</td>
          <td class="num">{bytes(o.spilled_bytes)}</td>
        </tr>
      {:else}
        <tr><td colspan="8" class="muted">No operator metrics recorded for this query.</td></tr>
      {/each}
    </tbody>
  </table>
{/if}
