<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api, type Operator, type QueryDetail } from "../api";
  import { bytes, count, nanos, when } from "../format";
  import { Alert } from "$lib/components/ui/alert";
  import { Badge, stateVariant } from "$lib/components/ui/badge";
  import { Card, CardLabel, CardValue } from "$lib/components/ui/card";
  import Id from "$lib/components/id.svelte";
  import ResponsiveTable, { type Column } from "$lib/components/responsive-table.svelte";

  let { id }: { id: string } = $props();

  const q = createQuery<QueryDetail>(() => ({
    queryKey: ["query", id],
    queryFn: () => api<QueryDetail>(`/api/internal/queries/${encodeURIComponent(id)}`),
    refetchInterval: (query) => (["running", "queued"].includes(query.state.data?.query.state ?? "") ? 1000 : false),
  }));

  const maxCompute = $derived(Math.max(1, ...(q.data?.operators ?? []).map((o) => o.compute_ns ?? 0)));
  const hotId = $derived(q.data?.hot_operator?.operator_id);
  const share = (o: Operator) => Math.round(((o.compute_ns ?? 0) / maxCompute) * 100);

  const columns: Column[] = [
    { key: "operator", label: "Operator", title: true },
    { key: "rows_in", label: "Rows in", num: true },
    { key: "rows_out", label: "Rows out", num: true },
    { key: "compute", label: "Compute", num: true },
    { key: "wait", label: "Waiting io / memory / sink", num: true },
    { key: "spilled", label: "Spilled", num: true },
  ];
</script>

<p class="mb-3 flex flex-wrap gap-4">
  <a class="py-2.5 underline underline-offset-4" href="#/queries">← Queries</a>
  <a class="py-2.5 underline underline-offset-4" href="#/history">History</a>
</p>
{#if q.isPending}
  <p class="text-muted-foreground">Loading…</p>
{:else if q.isError}
  <h1>Query not found</h1>
  <p class="text-error">{q.error.message}</p>
{:else}
  {@const d = q.data}
  <h1 class="mb-3 flex flex-wrap items-center gap-2">Query <Id id={d.query.query_id} /> <Badge variant={stateVariant(d.query.state)}>{d.query.state}</Badge></h1>
  <div class="mb-4 overflow-hidden rounded-lg border border-border bg-code">
    <pre class="m-0 overflow-x-auto p-3.5 font-mono text-[13px] leading-[22px] whitespace-pre-wrap text-text-2 wrap-anywhere">{d.query.query_text ?? "(text not recorded)"}{d.query.query_text_truncated ? " …" : ""}</pre>
  </div>
  <div class="grid grid-cols-[repeat(auto-fit,minmax(min(100%,11rem),1fr))] gap-3">
    <Card><CardLabel>Submitted</CardLabel><CardValue>{when(d.query.submitted_at)}</CardValue></Card>
    <Card><CardLabel>Wall time</CardLabel><CardValue>{nanos(d.query.wall_ns)}</CardValue></Card>
    <Card><CardLabel>CPU time</CardLabel><CardValue>{nanos(d.query.cpu_ns)}</CardValue></Card>
    <Card><CardLabel>Peak memory</CardLabel><CardValue>{bytes(d.query.peak_memory_bytes)}</CardValue></Card>
    <Card><CardLabel>Rows produced</CardLabel><CardValue>{count(d.query.rows_produced)}</CardValue></Card>
    <Card><CardLabel>Spilled</CardLabel><CardValue>{bytes(d.query.spill_bytes)}</CardValue></Card>
  </div>
  {#if d.query.error_class}
    <p class="mt-3 text-error">Failed with <code>{d.query.error_class}</code></p>
  {/if}

  {#if d.warnings.length}
    <h2 class="mt-6 mb-2">Warnings</h2>
    <div class="grid gap-2">
      {#each d.warnings as w}
        <Alert variant={w.kind === "skew" ? "warning" : "error"} icon={w.kind === "skew" ? "▲" : "✕"} title={`${w.operator} · ${w.kind}`}>{w.hint}</Alert>
      {/each}
    </div>
  {/if}

  <h2 class="mt-6 mb-1">Operators</h2>
  <p class="mb-3 text-muted-foreground">
    Compute time per operator; the busiest one is highlighted{d.hot_operator ? `: ${d.hot_operator.operator}` : ""}.
  </p>
  <ResponsiveTable {columns} rows={d.operators} rowKey={(o) => o.operator_id} empty="No operator metrics recorded for this query." rowClass={(o) => (o.operator_id === hotId ? "bg-accent font-bold" : "")}>
    {#snippet cell(o, c)}
      {#if c.key === "operator"}
        <span class="mono text-muted-foreground">{o.operator_id}</span> {o.name}{#if o.drivers && o.drivers > 1} <span class="font-normal text-muted-foreground">×{o.drivers}</span>{/if}
        <div class="mt-1.5 h-2 min-w-24 overflow-hidden rounded-sm bg-border" role="img" aria-label={`compute ${share(o)}% of the busiest operator`}>
          <div class={o.operator_id === hotId ? "h-full rounded-sm bg-primary" : "h-full rounded-sm bg-series-1"} style:width={`${share(o)}%`}></div>
        </div>
      {:else if c.key === "rows_in"}{count(o.rows_in)}
      {:else if c.key === "rows_out"}{count(o.rows_out)}
      {:else if c.key === "compute"}{nanos(o.compute_ns)}
      {:else if c.key === "wait"}{nanos(o.blocked_io_ns)} / {nanos(o.blocked_memory_ns)} / {nanos(o.blocked_sink_ns)}
      {:else}{bytes(o.spilled_bytes)}{/if}
    {/snippet}
  </ResponsiveTable>
{/if}
