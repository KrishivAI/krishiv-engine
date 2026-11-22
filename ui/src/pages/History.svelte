<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api, type Compare, type Query } from "../api";
  import { bytes, count, nanos, oneLine, when } from "../format";
  import { consoleInfo } from "../console.svelte";
  import { Badge, stateVariant } from "$lib/components/ui/badge";
  import { Button } from "$lib/components/ui/button";
  import { fieldClass } from "$lib/components/ui/input";
  import Id from "$lib/components/id.svelte";
  import ResponsiveTable, { type Column } from "$lib/components/responsive-table.svelte";
  import { cn, keyOf } from "$lib/utils";
  import { queryHref } from "../routes";

  let stateFilter = $state("");
  let principal = $state("");
  let client = $state("");
  let tag = $state("");
  // The Sessions page opens History on one session's queries: it leaves the id here, and it is read once.
  function sessionAsked(): string {
    try {
      const id = sessionStorage.getItem("krishiv_history_session") ?? "";
      sessionStorage.removeItem("krishiv_history_session");
      return id;
    } catch {
      return "";
    }
  }
  let session = $state(sessionAsked());
  let picked = $state<string[]>([]);
  let compared = $state<Compare | null>(null);
  let compareError = $state("");

  const q = createQuery<{ queries: Query[] }>(() => ({
    queryKey: ["history", stateFilter, principal, client, tag, session],
    queryFn: () => {
      const p = new URLSearchParams({ limit: "200" });
      if (stateFilter) p.set("state", stateFilter);
      if (principal) p.set("principal", principal);
      if (client.trim()) p.set("client", client.trim());
      if (tag.trim()) p.set("tag", tag.trim());
      if (session.trim()) p.set("session", session.trim());
      return api(`/api/internal/history?${p}`);
    },
    // Stored profiles do not change while the console is open.
    refetchInterval: consoleInfo.mode === "profiles" ? false : 5000,
  }));

  function toggle(id: string) {
    picked = picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id].slice(-2);
    compared = null;
  }

  async function compare() {
    compareError = "";
    try {
      compared = await api<Compare>(`/api/internal/compare?a=${encodeURIComponent(picked[0])}&b=${encodeURIComponent(picked[1])}`);
    } catch (e) {
      compareError = (e as Error).message;
    }
  }

  const fields: [string, string, (n: number) => string][] = [
    ["wall_ns", "Wall time", nanos],
    ["cpu_ns", "CPU time", nanos],
    ["peak_memory_bytes", "Peak memory", bytes],
    ["rows_produced", "Rows produced", count],
    ["bytes_read", "Bytes read", bytes],
    ["spill_bytes", "Spilled", bytes],
  ];
  const sign = (n: number) => (n > 0 ? "text-error" : n < 0 ? "text-success" : "");

  const allColumns: Column[] = [
    { key: "query", label: "Query", title: true },
    { key: "state", label: "State" },
    { key: "principal", label: "Principal", wrap: true },
    { key: "submitted", label: "Submitted" },
    { key: "wall", label: "Wall time", num: true },
    { key: "memory", label: "Peak memory", num: true },
  ];
  // One session has one user: no principal column and no principal filter in an embedded console.
  const onePrincipal = $derived(consoleInfo.mode === "embedded");
  const columns = $derived(onePrincipal ? allColumns.filter((c) => c.key !== "principal") : allColumns);
  // Queries of every client are a server's: only its console filters by client type and by tag.
  const manyClients = $derived(consoleInfo.mode === "server");
  const tagsOf = (r: Query) => Object.entries(r.tags ?? {}).map(([k, v]) => (v ? `${k}=${v}` : k));
  const numeric = (r: Query | Compare["a"], key: string) => (r as unknown as Record<string, number | null | undefined>)[key];
</script>

<h1 class="mb-4">History</h1>
{#if consoleInfo.mode === "profiles"}
  <p class="mb-3 text-muted-foreground">Saved profiles, read once when this console started. Nothing here is running.</p>
{/if}
<div class="mb-3 flex flex-wrap items-center gap-3">
  <label class="flex w-full items-center gap-2 sm:w-auto">State
    <select class={cn(fieldClass, "flex-1")} bind:value={stateFilter}>
      <option value="">any</option>
      <option value="finished">finished</option>
      <option value="failed">failed</option>
      <option value="cancelled">cancelled</option>
    </select>
  </label>
  {#if !onePrincipal}
    <label class="flex w-full items-center gap-2 sm:w-auto">Principal
      <input class={cn(fieldClass, "flex-1")} bind:value={principal} placeholder="any" size="14" />
    </label>
  {/if}
  {#if manyClients}
    <label class="flex w-full items-center gap-2 sm:w-auto">Client
      <input class={cn(fieldClass, "flex-1")} bind:value={client} placeholder="any" size="14" />
    </label>
    <label class="flex w-full items-center gap-2 sm:w-auto">Tag
      <input class={cn(fieldClass, "flex-1")} bind:value={tag} placeholder="key or key=value" size="16" />
    </label>
    <label class="flex w-full items-center gap-2 sm:w-auto">Session
      <input class={cn(fieldClass, "mono flex-1")} bind:value={session} placeholder="any" size="14" />
    </label>
  {/if}
  <Button class="w-full sm:w-auto" disabled={picked.length !== 2} onclick={compare}>Compare the two selected</Button>
  {#if compareError}<span class="text-error">{compareError}</span>{/if}
</div>

{#if compared}
  <h2 class="mb-2">Comparison</h2>
  <div data-table-box class="mb-4 max-w-full overflow-x-auto rounded-lg border border-border bg-card">
    <table class="w-full border-collapse">
      <thead>
        <tr class="text-xs font-bold tracking-[0.06em] text-muted-foreground uppercase">
          <th class="bg-muted px-3.5 py-2.5 text-left"></th>
          <th class="bg-muted px-3.5 py-2.5 text-right">A <span class="mono">…{compared.a.query_id.slice(-8)}</span></th>
          <th class="bg-muted px-3.5 py-2.5 text-right">B <span class="mono">…{compared.b.query_id.slice(-8)}</span></th>
          <th class="bg-muted px-3.5 py-2.5 text-right">B − A</th>
        </tr>
      </thead>
      <tbody>
        {#each fields as [key, label, fmt]}
          {@const a = numeric(compared.a, key)}
          {@const b = numeric(compared.b, key)}
          {@const d = compared.delta[key]}
          <tr>
            <td class="border-b border-border px-3.5 py-2.5">{label}</td>
            <td class="mono border-b border-border px-3.5 py-2.5 text-right">{a === null || a === undefined ? "–" : fmt(a)}</td>
            <td class="mono border-b border-border px-3.5 py-2.5 text-right">{b === null || b === undefined ? "–" : fmt(b)}</td>
            <td class={cn("mono border-b border-border px-3.5 py-2.5 text-right", d === undefined ? "" : sign(d))}>{d === undefined ? "–" : (d > 0 ? "+" : "") + fmt(d)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
{/if}

{#if q.isPending}
  <p class="text-muted-foreground">Loading…</p>
{:else if q.isError}
  <p class="text-error">Could not load history: {q.error.message}</p>
{:else}
  <ResponsiveTable {columns} rows={q.data.queries} rowKey={keyOf(q.data.queries, (r) => r.query_id)} empty="No finished queries match.">
    {#snippet cell(r, c)}
      {#if c.key === "query"}
        <label class="mr-2 inline-flex min-h-11 min-w-11 items-center justify-center align-middle">
          <input class="size-5 accent-[var(--k-silver)]" type="checkbox" checked={picked.includes(r.query_id)} onchange={() => toggle(r.query_id)} aria-label={`Select ${oneLine(r.query_text, 40)} (${r.query_id.slice(-8)}) for comparison`} />
        </label>
        <a class="inline-flex min-h-11 min-w-11 items-center align-middle underline underline-offset-4" href={queryHref(r.query_id, r.session_id)}>{oneLine(r.query_text, 90)}</a>
        <div><Id id={r.query_id} /></div>
        {#if tagsOf(r).length}
          <div class="mono text-xs break-all text-muted-foreground">{tagsOf(r).join(" · ")}</div>
        {/if}
      {:else if c.key === "state"}
        <Badge variant={stateVariant(r.state)}>{r.state}</Badge>{#if r.error_class} <span class="mono text-muted-foreground">{r.error_class}</span>{/if}
      {:else if c.key === "principal"}{r.principal ?? "–"}
        {#if r.client_type}<div class="ml-auto max-w-[min(12rem,40vw)] truncate text-xs text-muted-foreground xl:ml-0" title={r.client_type}>{r.client_type}</div>{/if}
      {:else if c.key === "submitted"}{when(r.submitted_at)}
      {:else if c.key === "wall"}{nanos(r.wall_ns)}
      {:else}{bytes(r.peak_memory_bytes)}{/if}
    {/snippet}
  </ResponsiveTable>
{/if}
