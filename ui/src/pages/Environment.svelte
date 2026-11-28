<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api, type Environment } from "../api";
  import { consoleInfo } from "../console.svelte";
  import { Badge } from "$lib/components/ui/badge";
  import { fieldClass } from "$lib/components/ui/input";
  import { cn } from "$lib/utils";
  import ResponsiveTable, { type Column } from "$lib/components/responsive-table.svelte";

  const q = createQuery<Environment>(() => ({
    queryKey: ["environment"],
    queryFn: () => api("/api/internal/environment"),
  }));
  let filter = $state("");
  const rows = $derived(
    (q.data?.settings ?? []).filter((s) =>
      `${s.key}\n${s.spark_key ?? ""}`.toLowerCase().includes(filter.trim().toLowerCase()),
    ),
  );
  // Only explicitly set keys are listed, so an untouched session has none: that is not a filter's miss.
  const empty = $derived(filter.trim() ? "No setting matches." : "Nothing is set: every setting has its default.");
  const columns: Column[] = [
    { key: "key", label: "Setting", title: true },
    { key: "value", label: "Value" },
  ];
</script>

<h1 class="mb-4">Environment</h1>
{#if q.isError}
  <p class="text-destructive wrap-anywhere">{(q.error as Error).message}</p>
{:else if q.data}
  <p class="mb-3 text-muted-foreground">
    Version {q.data.version}. {consoleInfo.mode === "embedded" ? "The settings of this session. " : ""}Secrets show as <code>***</code>.
  </p>
  <label class="mb-3 block">
    <span class="mb-1 block text-sm font-medium">Filter settings</span>
    <input class={cn(fieldClass, "w-full")} type="search" bind:value={filter} placeholder="spark.sql" />
  </label>
  <ResponsiveTable {columns} rows={rows} rowKey={(s) => s.key} {empty}>
    {#snippet cell(s, c)}
      {#if c.key === "key"}<span class="mono wrap-anywhere">{s.key}</span>
        {#if s.spark_key}<div class="text-xs text-muted-foreground">Spark's <span class="mono wrap-anywhere">{s.spark_key}</span></div>{/if}
      {:else}<span class="mono wrap-anywhere">{s.value}</span>
        {#if s.redacted}<Badge>redacted</Badge>{/if}{/if}
    {/snippet}
  </ResponsiveTable>
{:else}
  <p class="text-muted-foreground">Loading…</p>
{/if}
