<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api, type AdvisedColumn, type TableAdvice } from "../api";
  import { count, percent } from "../format";
  import { Button } from "$lib/components/ui/button";
  import { fieldClass } from "$lib/components/ui/input";
  import { cn } from "$lib/utils";
  import ResponsiveTable, { type Column } from "$lib/components/responsive-table.svelte";

  // Reading the advice reads the history store: the page asks when it opens and when asked to, not on a timer.
  const q = createQuery<{ tables: TableAdvice[]; truncated: boolean }>(() => ({
    queryKey: ["advice"],
    queryFn: () => api("/api/internal/advice"),
    refetchOnWindowFocus: false,
  }));

  let filter = $state("");
  const tables = $derived(q.data?.tables ?? []);
  const shown = $derived(tables.filter((t) => t.table_name.toLowerCase().includes(filter.trim().toLowerCase())));
  /** A table's best suggestion: the answer lists a table's columns by rank. */
  const best = (t: TableAdvice): AdvisedColumn | undefined => t.columns[0];
  const others = (t: TableAdvice) => t.columns.slice(1);

  const columns: Column[] = [
    { key: "table", label: "Table", title: true },
    { key: "column", label: "Partition by" },
    { key: "queries", label: "Queries filtering on it", num: true },
    { key: "partitions", label: "Partitions", num: true },
    { key: "pruned", label: "Bytes skipped (expected)", num: true },
    { key: "others", label: "Other candidates", wrap: true },
  ];
</script>

<h1 class="mb-4">Advisor</h1>
{#if q.isPending}
  <p class="text-muted-foreground">Loading…</p>
{:else if q.isError}
  <p class="text-error">Could not load the advice: {q.error.message}</p>
  <Button class="mt-3" onclick={() => q.refetch()}>Try again</Button>
{:else}
  <p class="mb-3 max-w-prose text-muted-foreground">
    Partition columns suggested by the query history: for each table, the column its finished queries filter on
    by equality that would let them skip the most data. Nothing is changed here; to follow a suggestion, write
    the table again partitioned by the column. <code>EXPLAIN</code> prints the same advice for the tables a
    query reads.
  </p>
  <div class="mb-3 flex flex-wrap items-center gap-3">
    {#if tables.length}
      <label class="flex w-full items-center gap-2 sm:w-auto">Filter
        <input class={cn(fieldClass, "flex-1")} type="search" bind:value={filter} placeholder="table name" size="26" />
      </label>
    {/if}
    <Button variant="outline" disabled={q.isFetching} onclick={() => q.refetch()}>{q.isFetching ? "Reading the history…" : "Read the history again"}</Button>
    {#if shown.length !== tables.length}<span class="text-muted-foreground" data-advice-count>{shown.length} of {tables.length} tables shown</span>{/if}
  </div>
  {#if tables.length === 0}
    <p class="max-w-prose text-muted-foreground" data-no-advice>
      No advice yet. Advice comes from the finished queries a server keeps in its query history
      (<code>krishiv.history.path</code>): a table is listed once queries filter it by equality on a column
      that has between 2 and 10,000 distinct values and is not close to unique.
    </p>
  {:else}
    <ResponsiveTable {columns} rows={shown} rowKey={(t) => t.table_name} empty="No table matches.">
      {#snippet cell(t, c)}
        {@const b = best(t)}
        {#if c.key === "table"}
          <span class="mono font-medium wrap-anywhere">{t.table_name}</span>
        {:else if c.key === "column"}
          <span class="mono font-medium wrap-anywhere">{b?.column_name ?? "–"}</span>
        {:else if c.key === "queries"}{count(b?.column_queries)} of {count(t.table_queries)}
        {:else if c.key === "partitions"}{count(b?.partitions)}
        {:else if c.key === "pruned"}{percent(b?.pruned_fraction)}
        {:else if others(t).length === 0}–
        {:else}
          <ul class="m-0 list-none p-0" data-other-candidates>
            {#each others(t) as o (o.rank)}
              <li><span class="mono">{o.column_name}</span> <span class="text-muted-foreground">· {count(o.column_queries)} queries · {count(o.partitions)} partitions · {percent(o.pruned_fraction)}</span></li>
            {/each}
          </ul>
        {/if}
      {/snippet}
    </ResponsiveTable>
    {#if q.data.truncated}
      <p class="mt-3 text-muted-foreground" data-advice-truncated>More tables have advice than are listed here; <code>system.partition_advice</code> holds them all.</p>
    {/if}
    <p class="mt-3 max-w-prose text-muted-foreground">
      "Bytes skipped" is the share of the bytes the table's queries read that partitioning by the column is
      expected to save: the share of queries that filter on it, times the part of the table one value leaves out.
    </p>
  {/if}
{/if}
