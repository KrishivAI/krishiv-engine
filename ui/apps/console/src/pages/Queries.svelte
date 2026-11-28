<script lang="ts">
  import { api, type Query } from "../api";
  import { bytes, count, elapsed, oneLine } from "@krishiv/ui/data/format";
  import { consoleInfo } from "../console.svelte";
  import { Badge, stateVariant } from "@krishiv/ui/components/badge";
  import { Button } from "@krishiv/ui/components/button";
  import { fieldClass } from "@krishiv/ui/components/input";
  import { cn, keyOf } from "@krishiv/ui/utils";
  import { queryHref } from "../routes";
  import Id from "@krishiv/ui/components/id";
  import ResponsiveTable, { type Column } from "@krishiv/ui/components/responsive-table";

  let queries = $state<Query[]>([]);
  let status = $state<"connecting" | "live" | "lost">("connecting");
  let now = $state(Date.now());
  let message = $state("");
  let filter = $state("");
  /** Everything of a query a person would type to find it. */
  const haystack = (q: Query) =>
    [q.query_id, q.session_id, q.principal, q.client_type, q.state, q.query_text, ...Object.entries(q.tags ?? {}).map(([k, v]) => (v ? `${k}=${v}` : k))]
      .filter(Boolean)
      .join("\n")
      .toLowerCase();
  const shown = $derived(queries.filter((q) => haystack(q).includes(filter.trim().toLowerCase())));

  $effect(() => {
    const tick = setInterval(() => (now = Date.now()), 1000);
    const source = new EventSource("/api/internal/events");
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

  // The row's session goes with the request: another session's query under the same id keeps running.
  async function cancel(q: Query) {
    const id = q.query_id;
    const of = q.session_id ? `?session=${encodeURIComponent(q.session_id)}` : "";
    try {
      await api(`/api/internal/queries/${encodeURIComponent(id)}/cancel${of}`, { method: "POST" });
      message = `Cancelled ${id}`;
    } catch (e) {
      message = (e as Error).message;
    }
  }

  const allColumns: Column[] = [
    { key: "query", label: "Query", title: true },
    { key: "state", label: "State" },
    { key: "principal", label: "Principal", wrap: true },
    { key: "running", label: "Running for", num: true },
    { key: "rows", label: "Rows read", num: true },
    { key: "bytes", label: "Bytes read", num: true },
    { key: "memory", label: "Peak memory", num: true },
  ];
  // One session has one user: the column would repeat the same name on every row.
  const columns = $derived(consoleInfo.mode === "embedded" ? allColumns.filter((c) => c.key !== "principal") : allColumns);
</script>

<h1 class="mb-4">Queries</h1>
<div class="mb-3 flex flex-wrap items-center gap-3">
  <Badge variant={status === "live" ? "success" : "default"}>{status}</Badge>
  <span class="text-muted-foreground">Refreshes once a second.</span>
  <label class="flex w-full items-center gap-2 sm:w-auto">Filter
    <input class={cn(fieldClass, "flex-1")} type="search" bind:value={filter} placeholder={consoleInfo.mode === "embedded" ? "text, state or id" : "principal, client, tag, text or id"} size="26" />
  </label>
  {#if shown.length !== queries.length}<span class="text-muted-foreground" data-query-count>{shown.length} of {queries.length} shown</span>{/if}
  {#if message}<span class="text-muted-foreground wrap-anywhere">{message}</span>{/if}
</div>
<ResponsiveTable {columns} rows={shown} rowKey={keyOf(shown, (q) => q.query_id)} empty={queries.length ? "No running query matches." : consoleInfo.mode === "embedded" ? "No query of this session is running." : "No query is running."}>
  {#snippet cell(q, c)}
    {#if c.key === "query"}
      <a class="flex min-h-11 items-center underline underline-offset-4" href={queryHref(q.query_id, q.session_id)}>{oneLine(q.query_text, 80)}</a>
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
      <Button variant="destructive" onclick={() => cancel(q)}>Cancel</Button>
    {/if}
  {/snippet}
</ResponsiveTable>
<p class="mt-3 text-muted-foreground">
  Progress and an estimate of the time left appear once the engine records how many rows a query has read.
</p>
