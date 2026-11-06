<script lang="ts">
  import { api, token, type Query } from "../api";
  import { bytes, count, elapsed, oneLine } from "../format";

  let queries = $state<Query[]>([]);
  let status = $state<"connecting" | "live" | "lost">("connecting");
  let now = $state(Date.now());
  let message = $state("");

  $effect(() => {
    const tick = setInterval(() => (now = Date.now()), 1000);
    const source = new EventSource(`/api/internal/events?token=${encodeURIComponent(token() ?? "")}`);
    source.addEventListener("queries", (e) => {
      const body = JSON.parse((e as MessageEvent).data);
      if (body.error) {
        message = body.error;
        return;
      }
      message = "";
      queries = body.queries;
      status = "live";
    });
    source.onerror = () => (status = "lost");
    return () => {
      clearInterval(tick);
      source.close();
    };
  });

  async function cancel(id: string) {
    try {
      await api(`/api/internal/queries/${encodeURIComponent(id)}/cancel`, { method: "POST" });
      message = `Cancelled ${id}`;
    } catch (e) {
      message = (e as Error).message;
    }
  }
</script>

<h1>Queries</h1>
<div class="toolbar">
  <span class="pill" class:running={status === "live"}>{status === "live" ? "live" : status}</span>
  <span class="muted">Refreshes once a second.</span>
  {#if message}<span class="muted">{message}</span>{/if}
</div>
<table>
  <thead>
    <tr>
      <th>Query</th><th>State</th><th>Principal</th><th class="num">Running for</th>
      <th class="num">Rows read</th><th class="num">Bytes read</th><th class="num">Peak memory</th><th></th>
    </tr>
  </thead>
  <tbody>
    {#each queries as q (q.query_id)}
      <tr>
        <td>
          <a href={`#/queries/${encodeURIComponent(q.query_id)}`}>{oneLine(q.query_text, 80)}</a>
          <div class="muted mono">{q.query_id}</div>
        </td>
        <td><span class="pill {q.state}">{q.state}</span></td>
        <td>{q.principal ?? "–"}</td>
        <td class="num">{elapsed(q.submitted_at, now)}</td>
        <td class="num">{count(q.rows_read)}</td>
        <td class="num">{bytes(q.bytes_read)}</td>
        <td class="num">{bytes(q.peak_memory_bytes)}</td>
        <td>
          {#if q.state === "running"}
            <button class="danger" onclick={() => cancel(q.query_id)}>Cancel</button>
          {/if}
        </td>
      </tr>
    {:else}
      <tr><td colspan="8" class="muted">No query is running.</td></tr>
    {/each}
  </tbody>
</table>
<p class="muted">
  Progress and an estimate of the time left appear once the engine records how many rows a query has read.
</p>
