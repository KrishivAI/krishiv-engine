<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api, type Operator, type QueryDetail } from "../api";
  import { heatRows, METRICS, reported, type HeatRow } from "../plan";
  import { bytes, count, nanos, percent, when } from "../format";
  import { consoleInfo } from "../console.svelte";
  import { Alert } from "$lib/components/ui/alert";
  import { Badge, stateVariant } from "$lib/components/ui/badge";
  import { Card, CardLabel, CardValue } from "$lib/components/ui/card";
  import Id from "$lib/components/id.svelte";
  import ResponsiveTable, { type Column } from "$lib/components/responsive-table.svelte";

  import type { EmbedLink } from "../embed";

  // `embed`: the embedded view reads through its signed link (no session) and shows no navigation.
  // `session`: the session of the row the page was opened from, when two sessions may share the id.
  let { id, embed = null, session = undefined }: { id: string; embed?: EmbedLink | null; session?: string } = $props();

  const ended = (d: QueryDetail) => !["running", "queued"].includes(d.query.state ?? "");

  // The session travels with both reads; a signed link was signed with it, so the embedded view sends its own.
  // An empty qualifier is sent as it came, so the server refuses it as it does when asked directly.
  const at = (sep: string) => (session !== undefined ? `${sep}session=${encodeURIComponent(session)}` : "");

  const q = createQuery<QueryDetail>(() => ({
    // An absent qualifier and an empty one are different requests, so different entries.
    queryKey: ["query", id, session ?? null],
    queryFn: () =>
      api<QueryDetail>(
        embed
          ? `/api/internal/embed/queries/${encodeURIComponent(id)}?exp=${encodeURIComponent(embed.exp)}&sig=${encodeURIComponent(embed.sig)}${at("&")}`
          : `/api/internal/queries/${encodeURIComponent(id)}${at("?")}`,
      ),
    refetchInterval: (query) => {
      const d = query.state.data;
      if (!d || consoleInfo.mode === "profiles") return false;
      // A running query changes; an ended one is asked for again only while the server says its plan can
      // still arrive (its profile is written a moment after it ends), which the server bounds in time.
      return !ended(d) || d.plan_pending ? 1000 : false;
    },
  }));
  const live = $derived(!!q.data && !ended(q.data));

  const maxCompute = $derived(Math.max(1, ...(q.data?.operators ?? []).map((o) => o.compute_ns ?? 0)));
  const hotId = $derived(q.data?.hot_operator?.operator_id);
  const share = (o: Operator) => Math.round(((o.compute_ns ?? 0) / maxCompute) * 100);

  let metricKey = $state("compute");
  const metric = $derived(METRICS.find((m) => m.key === metricKey) ?? METRICS[0]);
  const hasPlan = $derived(!!q.data?.plan?.length);
  const heat = $derived(heatRows(q.data?.plan, q.data?.operators ?? [], metric));
  // One linear scale from zero to the plan's largest value; the smallest positive value still shows a tint.
  const shade = (r: HeatRow) => (r.state === "value" ? `color-mix(in srgb, var(--primary) ${Math.max(3, Math.round(r.heat * 0.5))}%, transparent)` : "transparent");
  const shareText = (r: HeatRow) => (r.share == null ? "" : r.share > 0 && r.share < 0.001 ? "<0.1%" : percent(r.share));
  /** A counter the operator does not measure reads "–" in the table, as a missing figure does. */
  const waits = (o: Operator) => [reported.ioWait(o), reported.memoryWait(o), reported.sinkWait(o)].map(nanos).join(" / ");

  const columns: Column[] = [
    { key: "operator", label: "Operator", title: true },
    { key: "rows_in", label: "Rows in", num: true },
    { key: "rows_out", label: "Rows out", num: true },
    { key: "compute", label: "Compute", num: true },
    { key: "wait", label: "Waiting io / memory / sink", num: true },
    { key: "memory", label: "Peak memory", num: true },
    { key: "spilled", label: "Spilled", num: true },
  ];
</script>

{#if !embed}
  <p class="mb-3 flex flex-wrap gap-4">
    {#if consoleInfo.mode === "profiles"}
      <a class="inline-flex min-h-11 min-w-11 items-center underline underline-offset-4" href="#/history">← History</a>
    {:else}
      <a class="inline-flex min-h-11 min-w-11 items-center underline underline-offset-4" href="#/queries">← Queries</a>
      <a class="inline-flex min-h-11 min-w-11 items-center underline underline-offset-4" href="#/history">History</a>
    {/if}
  </p>
{/if}
{#if q.isPending}
  <p class="text-muted-foreground">Loading…</p>
{:else if q.isError}
  <h1>{embed ? "This link is not valid" : "Query not found"}</h1>
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

  {#if d.query.hints?.length}
    <h2 class="mt-6 mb-2">Advisor hints</h2>
    <div class="grid gap-2" data-testid="advisor-hints">
      {#each d.query.hints as h}
        <Alert variant="warning" icon="▲" title={h.code}>{h.message}</Alert>
      {/each}
    </div>
  {/if}

  {#if d.operators.length}
    <h2 class="mt-6 mb-1">Plan heatmap</h2>
    <div class="mb-3 flex flex-wrap items-center gap-2" role="group" aria-label="Heatmap metric">
      {#each METRICS as m}
        <button type="button" class="min-h-11 rounded-md border border-border px-3 py-2 {m.key === metricKey ? 'bg-primary text-primary-foreground' : 'bg-card'}" aria-pressed={m.key === metricKey} onclick={() => (metricKey = m.key)}>{m.label}</button>
      {/each}
    </div>
    {#if !hasPlan}
      <p class="mb-2 text-muted-foreground" data-testid="no-plan">
        {#if live}
          The plan tree is drawn once the query has ended; until then its operators are listed in id order.
        {:else if d.plan_pending}
          The query's profile is being stored; the plan tree is drawn when it is. Until then the operators are listed in id order.
        {:else}
          No plan tree is stored for this query, so its operators are listed in id order. The tree comes from the query history (<code>krishiv.history.path</code>).
        {/if}
      </p>
    {/if}
    <ol class="mb-2 grid list-none gap-1 p-0" aria-label={`${hasPlan ? "Plan" : "Operators"} shaded by ${metric.label.toLowerCase()}`} data-testid="plan-tree">
      {#each heat as r (r.key)}
        <li
          class="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 rounded-md border px-3 py-2 {r.state === 'zero' ? 'border-dashed border-muted-foreground bg-card' : 'border-border'}"
          style:margin-left={`${Math.min(r.depth, 12) * 1.25}rem`}
          style:background={r.state === "value" ? shade(r) : undefined}
          data-state={r.state}
          data-depth={r.depth}
        >
          <span class="min-w-0 wrap-anywhere">{#if r.id != null}<span class="mono text-muted-foreground">{r.id}</span> {/if}<span data-part="name">{r.name}</span></span>
          {#if r.value == null}
            <span class="text-muted-foreground" data-part="value">not reported</span>
          {:else}
            <span class="mono" data-part="value">{metric.show(r.value)}{#if r.share != null}{" "}<span class="text-muted-foreground" data-part="share">{shareText(r)}</span>{/if}</span>
          {/if}
        </li>
      {/each}
    </ol>
    <p class="mb-3 text-muted-foreground" data-testid="heat-note">
      {#if metric.over === "sum"}Each value is shown with its share of the plan's total.{:else if metric.over === "max"}A node shows the largest peak of its operators; peaks are reached at different times, so they have no total.{:else}A node shows the rows its output side produced.{/if}
      A dashed box is a measured zero; "not reported" is a figure the operator does not measure.
      {#if metric.time}<span data-testid="wall-note">Operator times are summed over parallel drivers and overlap while operators wait on each other, so they do not add up to the query's wall time ({nanos(d.query.wall_ns)}).</span>{/if}
    </p>
  {/if}

  <h2 class="mt-6 mb-1">Operators</h2>
  <p class="mb-3 text-muted-foreground">
    Compute time per operator; the busiest one is highlighted{d.hot_operator ? `: ${d.hot_operator.operator}` : ""}.
  </p>
  <ResponsiveTable {columns} rows={d.operators} rowKey={(o) => o.operator_id} empty={d.operators_withheld ? "Another session ran a query under this id, and operator metrics are kept by query id alone, so they are not shown for either run." : "No operator metrics recorded for this query."} rowClass={(o) => (o.operator_id === hotId ? "bg-accent font-bold" : "")}>
    {#snippet cell(o, c)}
      {#if c.key === "operator"}
        <span class="mono text-muted-foreground">{o.operator_id}</span> {o.name}{#if o.drivers && o.drivers > 1}{" "}<span class="font-normal text-muted-foreground">×{o.drivers}</span>{/if}
        <div class="mt-1.5 h-2 min-w-24 overflow-hidden rounded-sm bg-border" role="img" aria-label={`compute ${share(o)}% of the busiest operator`}>
          <div class={o.operator_id === hotId ? "h-full rounded-sm bg-primary" : "h-full rounded-sm bg-series-1"} style:width={`${share(o)}%`}></div>
        </div>
      {:else if c.key === "rows_in"}{count(reported.rowsIn(o))}
      {:else if c.key === "rows_out"}{count(reported.rowsOut(o))}
      {:else if c.key === "compute"}{nanos(o.compute_ns)}
      {:else if c.key === "wait"}{waits(o)}
      {:else if c.key === "memory"}{bytes(reported.memory(o))}
      {:else}{bytes(reported.spill(o))}{/if}
    {/snippet}
  </ResponsiveTable>
{/if}
