<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api, type Environment } from "../api";
  import { consoleInfo } from "../console.svelte";
  import { Badge } from "@krishiv/ui/components/badge";
  import { fieldClass } from "@krishiv/ui/components/input";
  import { cn } from "@krishiv/ui/utils";
  import ResponsiveTable, { type Column } from "@krishiv/ui/components/responsive-table";

  const q = createQuery<Environment>(() => ({
    queryKey: ["environment"],
    queryFn: () => api("/api/internal/environment"),
  }));
  let filter = $state("");
  const rows = $derived(
    (q.data?.settings ?? []).filter((s) => s.key.toLowerCase().includes(filter.trim().toLowerCase())),
  );
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
  <ResponsiveTable {columns} rows={rows} rowKey={(s) => s.key} empty="No setting matches.">
    {#snippet cell(s, c)}
      {#if c.key === "key"}<span class="mono">{s.key}</span>
      {:else}<span class="mono wrap-anywhere">{s.value}</span>
        {#if s.redacted}<Badge>redacted</Badge>{/if}{/if}
    {/snippet}
  </ResponsiveTable>
{:else}
  <p class="text-muted-foreground">Loading…</p>
{/if}
