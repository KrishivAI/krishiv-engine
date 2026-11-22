<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api, type ResourceGroup } from "../api";
  import { bytes, count, when } from "../format";
  import { Badge } from "$lib/components/ui/badge";
  import ResponsiveTable, { type Column } from "$lib/components/responsive-table.svelte";

  const q = createQuery<{ groups: ResourceGroup[] }>(() => ({
    queryKey: ["groups"],
    queryFn: () => api("/api/internal/groups"),
    refetchInterval: 2000,
  }));

  const groups = $derived(q.data?.groups ?? []);
  /** How many ancestors each group has: the table indents a group under its parent. */
  const depth = $derived.by(() => {
    const parent = new Map(groups.map((g) => [g.group_id, g.parent_id ?? null]));
    const of = new Map<string, number>();
    for (const g of groups) {
      let d = 0;
      // A group names its parent; the walk stops at a root, or after as many steps as there are groups.
      for (let p = parent.get(g.group_id) ?? null; p !== null && d < groups.length; p = parent.get(p) ?? null) d++;
      of.set(g.group_id, d);
    }
    return of;
  });
  /** A group's own name: its id without its parent's prefix. */
  const nameOf = (g: ResourceGroup) =>
    g.parent_id && g.group_id.startsWith(`${g.parent_id}.`) ? g.group_id.slice(g.parent_id.length + 1) : g.group_id;
  const policy = (g: ResourceGroup) =>
    g.scheduling_policy === "weighted_fair" ? `fair · weight ${g.scheduling_weight}` : `by priority · ${g.priority}`;
  /** `used / limit`, or the use alone when the group has no such limit. */
  const of = (used: string, limit: number | null | undefined, fmt: (n: number) => string) =>
    limit === null || limit === undefined ? used : `${used} / ${fmt(limit)}`;
  /** A group at its limit: nothing more of it runs (or waits) until something ends. */
  const full = (used: number, limit: number) => limit > 0 && used >= limit;

  const columns: Column[] = [
    { key: "group", label: "Group", title: true },
    { key: "class", label: "Class" },
    { key: "policy", label: "Scheduling" },
    { key: "running", label: "Running", num: true },
    { key: "queued", label: "Queued", num: true },
    { key: "memory", label: "Memory in use", num: true },
    { key: "query", label: "Per query", num: true },
    { key: "scan", label: "Scanned", num: true },
  ];
</script>

<h1 class="mb-4">Workload</h1>
{#if q.isPending}
  <p class="text-muted-foreground">Loading…</p>
{:else if q.isError}
  <p class="text-error">Could not load the resource groups: {q.error.message}</p>
{:else if groups.length === 0}
  <p class="text-muted-foreground" data-no-groups>
    No resource groups are configured: every query is admitted as it arrives. A server started with
    <code>krishiv.workload.resourceGroups.file</code> admits queries by group, and its groups are listed here.
  </p>
{:else}
  <p class="mb-3 text-muted-foreground">
    The resource groups queries are admitted through. A group's numbers include its subgroups'.
  </p>
  <ResponsiveTable {columns} rows={groups} rowKey={(g) => g.group_id}>
    {#snippet cell(g, c)}
      {#if c.key === "group"}
        <span class="mono inline-block font-medium" style:padding-left={`${Math.min(depth.get(g.group_id) ?? 0, 12) * 1.25}rem`} title={g.group_id} data-depth={depth.get(g.group_id) ?? 0}>{nameOf(g)}</span>
        {#if g.state !== "active"} <Badge variant="cancelled">{g.state}</Badge>{/if}
      {:else if c.key === "class"}{g.workload_class}
      {:else if c.key === "policy"}{policy(g)}
      {:else if c.key === "running"}
        <span class={full(g.running, g.hard_concurrency_limit) ? "font-bold text-foreground" : ""}>{of(count(g.running), g.hard_concurrency_limit, count)}</span>
      {:else if c.key === "queued"}
        <span class={full(g.queued, g.max_queued) ? "font-bold text-foreground" : ""}>{of(count(g.queued), g.max_queued, count)}</span>
      {:else if c.key === "memory"}{of(bytes(g.memory_reserved_bytes), g.soft_memory_limit_bytes, bytes)}
      {:else if c.key === "query"}{bytes(g.per_query_memory_limit_bytes)}
      {:else if g.scan_quota_bytes === null || g.scan_quota_bytes === undefined}–
      {:else}
        {of(bytes(g.scan_bytes_in_window ?? 0), g.scan_quota_bytes, bytes)}
        {#if g.scan_window_resets_at}<div class="text-xs text-muted-foreground">until {when(g.scan_window_resets_at)}</div>{/if}
      {/if}
    {/snippet}
  </ResponsiveTable>
  <p class="mt-3 text-muted-foreground">
    Running and Queued show the group's count and its limit; a count in bold is at the limit.
  </p>
{/if}
