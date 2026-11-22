<script lang="ts">
  import { createQuery, useQueryClient } from "@tanstack/svelte-query";
  import { api, type Session } from "../api";
  import { count, elapsed, when } from "../format";
  import { Badge, type BadgeVariant } from "$lib/components/ui/badge";
  import { Button } from "$lib/components/ui/button";
  import { fieldClass } from "$lib/components/ui/input";
  import Id from "$lib/components/id.svelte";
  import ResponsiveTable, { type Column } from "$lib/components/responsive-table.svelte";
  import { cn } from "$lib/utils";

  const client = useQueryClient();
  const q = createQuery<{ sessions: Session[]; truncated: boolean }>(() => ({
    queryKey: ["sessions"],
    queryFn: () => api("/api/internal/sessions"),
    refetchInterval: 2000,
  }));

  let filter = $state("");
  let ended = $state(false);
  let now = $state(Date.now());
  // The session whose "End session" was pressed once: the second press ends it.
  let asked = $state<string | null>(null);
  let message = $state("");
  $effect(() => {
    const tick = setInterval(() => (now = Date.now()), 1000);
    return () => clearInterval(tick);
  });

  const tagsOf = (s: Session) => Object.entries(s.tags ?? {}).map(([k, v]) => (v ? `${k}=${v}` : k));
  const isOpen = (s: Session) => s.state === "active";
  /** Everything of a session a person would type to find it. */
  const haystack = (s: Session) =>
    [s.session_id, s.principal, s.client_user_id, s.client_type, s.transport, s.delegated_by, ...tagsOf(s)]
      .filter(Boolean)
      .join("\n")
      .toLowerCase();
  const all = $derived(q.data?.sessions ?? []);
  const open = $derived(all.filter(isOpen).length);
  const rows = $derived(
    all.filter((s) => (ended || isOpen(s)) && haystack(s).includes(filter.trim().toLowerCase())),
  );

  function variant(state: string): BadgeVariant {
    if (state === "active") return "info";
    if (state === "killed") return "error";
    return state === "expired" ? "cancelled" : "default";
  }

  /** Opens History on the queries of one session. */
  function queriesOf(id: string) {
    try {
      sessionStorage.setItem("krishiv_history_session", id);
    } catch {
      // No storage: History opens unfiltered.
    }
    window.location.hash = "#/history";
  }

  async function end(s: Session) {
    asked = null;
    try {
      const r = await api<{ queries_cancelled: number }>(
        `/api/internal/sessions/${encodeURIComponent(s.session_id)}/close`,
        { method: "POST" },
      );
      const n = r.queries_cancelled;
      message = `Ended the session of ${s.principal}${n ? ` and cancelled ${count(n)} ${n === 1 ? "query" : "queries"}` : ""}.`;
      await client.invalidateQueries({ queryKey: ["sessions"] });
    } catch (e) {
      message = `Could not end the session of ${s.principal}: ${(e as Error).message}`;
    }
  }

  const columns: Column[] = [
    { key: "session", label: "Session", title: true },
    { key: "state", label: "State" },
    { key: "client", label: "Client", wrap: true },
    { key: "namespace", label: "Namespace", wrap: true },
    { key: "active", label: "Last active" },
    { key: "running", label: "Running", num: true },
    { key: "started", label: "Queries", num: true },
  ];
</script>

<h1 class="mb-4">Sessions</h1>
<div class="mb-3 flex flex-wrap items-center gap-3">
  <label class="flex w-full items-center gap-2 sm:w-auto">Filter
    <input class={cn(fieldClass, "flex-1")} type="search" bind:value={filter} placeholder="principal, client, tag or id" size="26" />
  </label>
  <label class="flex min-h-11 items-center gap-2">
    <input class="size-5 accent-[var(--k-silver)]" type="checkbox" bind:checked={ended} />
    Show ended sessions
  </label>
  {#if q.data}
    <span class="text-muted-foreground" data-session-count>{count(open)} open{#if rows.length !== (ended ? all.length : open)} · {count(rows.length)} shown{/if}</span>
  {/if}
</div>
{#if message}<p class="mb-3 text-muted-foreground wrap-anywhere" role="status">{message}</p>{/if}
{#if q.isPending}
  <p class="text-muted-foreground">Loading…</p>
{:else if q.isError}
  <p class="text-error">Could not load sessions: {q.error.message}</p>
{:else}
  <ResponsiveTable {columns} {rows} rowKey={(s) => s.session_id} empty={ended ? "No session matches." : "No open session matches."}>
    {#snippet cell(s, c)}
      {#if c.key === "session"}
        <span class="font-medium wrap-anywhere">{s.principal}</span>
        {#if s.delegated_by}<span class="text-muted-foreground"> for {s.delegated_by}</span>{/if}
        <div><Id id={s.session_id} /></div>
        {#if tagsOf(s).length}
          <div class="mono text-xs break-all text-muted-foreground">{tagsOf(s).join(" · ")}</div>
        {/if}
      {:else if c.key === "state"}
        <Badge variant={variant(s.state)}>{s.state}</Badge>{#if s.close_reason} <span class="mono text-muted-foreground">{s.close_reason}</span>{/if}
      {:else if c.key === "client"}
        <span title={s.client_type ?? undefined}>{s.client_type ?? "–"}</span>
        <div class="text-xs text-muted-foreground">{s.transport}</div>
      {:else if c.key === "namespace"}
        <span class="mono">{s.current_catalog}.{s.current_namespace}</span>
      {:else if c.key === "active"}
        {#if isOpen(s)}{elapsed(s.last_active_at, now)} ago{:else}{when(s.closed_at ?? s.last_active_at)}{/if}
      {:else if c.key === "running"}{count(s.queries_running)}
      {:else if s.queries_started}
        <button class="mono inline-flex min-h-6 items-center justify-end bg-transparent px-1 text-foreground underline underline-offset-4 pointer-coarse:min-h-11 pointer-coarse:min-w-11" onclick={() => queriesOf(s.session_id)} aria-label={`The ${count(s.queries_started)} queries of this session in History`}>{count(s.queries_started)}</button>
      {:else}0{/if}
    {/snippet}
    {#snippet actions(s)}
      {#if isOpen(s) && s.endable === false}
        <span class="text-muted-foreground">The console's editor</span>
      {:else if isOpen(s)}
        {#if asked === s.session_id}
          <span class="flex flex-wrap items-center gap-2">
            <Button variant="destructive" onclick={() => end(s)}>
              {s.queries_running ? `End it and cancel ${count(s.queries_running)} running` : "End it"}
            </Button>
            <Button onclick={() => (asked = null)}>Keep</Button>
          </span>
        {:else}
          <Button onclick={() => (asked = s.session_id)}>End session</Button>
        {/if}
      {/if}
    {/snippet}
  </ResponsiveTable>
  {#if q.data.truncated}
    <p class="mt-3 text-muted-foreground">Only the 1,000 sessions active last are listed.</p>
  {/if}
{/if}
<p class="mt-3 text-muted-foreground">
  Ending a session cancels what it is running; its client must open a new session to go on.
</p>
