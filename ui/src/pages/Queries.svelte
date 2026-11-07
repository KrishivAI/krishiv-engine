<script lang="ts">
  import { api, token, type Query } from "../api";
  import { bytes, count, elapsed, oneLine } from "../format";
  import { Badge, stateVariant } from "$lib/components/ui/badge";
  import { Button } from "$lib/components/ui/button";
  import Id from "$lib/components/id.svelte";
  import ResponsiveTable, { type Column } from "$lib/components/responsive-table.svelte";

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

  const columns: Column[] = [
    { key: "query", label: "Query", title: true },
    { key: "state", label: "State" },
    { key: "principal", label: "Principal" },
    { key: "running", label: "Running for", num: true },
    { key: "rows", label: "Rows read", num: true },
    { key: "bytes", label: "Bytes read", num: true },
    { key: "memory", label: "Peak memory", num: true },
  ];
</script>

<h1 class="mb-4">Queries</h1>
<div class="mb-3 flex flex-wrap items-center gap-3">
  <Badge variant={status === "live" ? "success" : "default"}>{status}</Badge>
  <span class="text-muted-foreground">Refreshes once a second.</span>
  {#if message}<span class="text-muted-foreground wrap-anywhere">{message}</span>{/if}
</div>
<ResponsiveTable {columns} rows={queries} rowKey={(q) => q.query_id} empty="No query is running.">
  {#snippet cell(q, c)}
    {#if c.key === "query"}
      <a class="block min-h-10 py-2 underline underline-offset-4" href={`#/queries/${encodeURIComponent(q.query_id)}`}>{oneLine(q.query_text, 80)}</a>
      <div><Id id={q.query_id} /></div>
    {:else if c.key === "state"}
      <Badge variant={stateVariant(q.state)}>{q.state}</Badge>
    {:else if c.key === "principal"}{q.principal ?? "–"}
    {:else if c.key === "running"}{elapsed(q.submitted_at, now)}
    {:else if c.key === "rows"}{count(q.rows_read)}
    {:else if c.key === "bytes"}{bytes(q.bytes_read)}
    {:else}{bytes(q.peak_memory_bytes)}{/if}
  {/snippet}
  {#snippet actions(q)}
    {#if q.state === "running"}
      <Button variant="destructive" onclick={() => cancel(q.query_id)}>Cancel</Button>
    {/if}
  {/snippet}
</ResponsiveTable>
<p class="mt-3 text-muted-foreground">
  Progress and an estimate of the time left appear once the engine records how many rows a query has read.
</p>
