<script lang="ts">
  import { EditorView, minimalSetup } from "codemirror";
  import { EditorState } from "@codemirror/state";
  import { keymap } from "@codemirror/view";
  import { sql } from "@codemirror/lang-sql";
  import { api, type SqlRun } from "../api";
  import { Badge, stateVariant } from "$lib/components/ui/badge";
  import { Button } from "$lib/components/ui/button";

  let host = $state<HTMLDivElement>();
  let view: EditorView | undefined;
  let run = $state<SqlRun | null>(null);
  let runId = $state<string | null>(null);
  let message = $state("");
  let timer: ReturnType<typeof setInterval> | undefined;
  // Bumped when the page goes away or a start begins: a start that finishes later installs nothing.
  let startToken = 0;

  const text = () => view?.state.doc.toString() ?? "";

  // The editor lives in a shadow root: CodeMirror then adopts its stylesheet (a constructable one, which the page's
  // `style-src 'self'` does not govern) instead of injecting a <style> element the policy would block. Colours come
  // from the page's custom properties, which cross the shadow boundary; there is no syntax colouring, so the text
  // keeps the page's own contrast in dark and light.
  const theme = EditorView.theme({
    "&": { color: "var(--foreground)", backgroundColor: "var(--card)", minHeight: "8rem", fontSize: "14px" },
    ".cm-content": { fontFamily: "'Red Hat Mono', ui-monospace, monospace", caretColor: "var(--foreground)", padding: "8px 4px" },
    "&.cm-focused": { outline: "2px solid var(--k-accent)" },
    ".cm-cursor": { borderLeftColor: "var(--foreground)" },
    ".cm-selectionBackground, &.cm-focused .cm-selectionBackground": { backgroundColor: "color-mix(in srgb, var(--foreground) 25%, transparent)" },
  });

  $effect(() => {
    if (!host) return;
    const root = host.shadowRoot ?? host.attachShadow({ mode: "open" });
    const mountPoint = document.createElement("div");
    root.replaceChildren(mountPoint);
    const draft = sessionStorage.getItem("krishiv_sql_draft") ?? "SELECT 1";
    sessionStorage.removeItem("krishiv_sql_draft");
    view = new EditorView({
      parent: mountPoint,
      root,
      state: EditorState.create({
        doc: draft,
        extensions: [
          minimalSetup,
          theme,
          sql(),
          EditorView.contentAttributes.of({ "aria-label": "SQL statement" }),
          keymap.of([{ key: "Mod-Enter", run: () => (start(), true) }]),
        ],
      }),
    });
    return () => {
      startToken++;
      view?.destroy();
      clearInterval(timer);
    };
  });

  // Each poll belongs to the run it was started for: a late answer for an earlier run changes nothing.
  async function poll(id: string | null | undefined = runId) {
    if (!id) return;
    try {
      const r = await api<SqlRun>(`/api/internal/sql/${encodeURIComponent(id)}`);
      if (id !== runId) return;
      run = r;
      if (r.state !== "running") clearInterval(timer);
    } catch (e) {
      if (id !== runId) return;
      message = (e as Error).message;
      clearInterval(timer);
    }
  }

  async function start() {
    // The Run button is disabled while a statement runs; the keyboard shortcut must not bypass that.
    if (run?.state === "running") return;
    message = "";
    run = { state: "running" };
    const token = ++startToken;
    try {
      const r = await api<{ run_id: string }>("/api/internal/sql", { method: "POST", body: JSON.stringify({ sql: text() }) });
      if (token !== startToken) return;
      runId = r.run_id;
      clearInterval(timer);
      const id = r.run_id;
      timer = setInterval(() => void poll(id), 250);
      void poll(id);
    } catch (e) {
      if (token !== startToken) return;
      run = null;
      message = (e as Error).message;
    }
  }

  async function cancel() {
    if (!runId) return;
    try {
      await api(`/api/internal/sql/${encodeURIComponent(runId)}/cancel`, { method: "POST" });
      void poll();
    } catch (e) {
      message = (e as Error).message;
      // The run may have ended meanwhile: show its real state.
      void poll();
    }
  }

  const cellText = (v: unknown) => (v === null || v === undefined ? "NULL" : typeof v === "object" ? JSON.stringify(v) : String(v));
</script>

<h1 class="mb-4">SQL</h1>
<div bind:this={host} class="mb-3 overflow-hidden rounded-md border border-border"></div>
<div class="mb-3 flex flex-wrap items-center gap-3">
  <Button onclick={start} disabled={run?.state === "running"}>Run</Button>
  <Button variant="destructive" onclick={cancel} disabled={run?.state !== "running"}>Cancel</Button>
  {#if run}<Badge variant={stateVariant(run.state)}>{run.state}</Badge>{/if}
  <span class="text-muted-foreground">Ctrl or Cmd + Enter runs the statement.</span>
</div>
{#if message}<p class="mb-3 text-destructive wrap-anywhere">{message}</p>{/if}
{#if run?.error && run.state !== "cancelled"}
  <p class="mb-3 wrap-anywhere" role="alert">{run.error_class ? `[${run.error_class}] ` : ""}{run.error}</p>
{/if}
{#if run?.columns}
  <div class="overflow-x-auto">
    <table class="w-full text-left" aria-label="Result">
      <thead><tr>{#each run.columns as c, i (i)}<th class="py-2 pr-4">{c}</th>{/each}</tr></thead>
      <tbody>
        {#each run.rows ?? [] as r, i (i)}
          <tr>{#each r as v, j (j)}<td class="mono py-1 pr-4 wrap-anywhere">{cellText(v)}</td>{/each}</tr>
        {/each}
      </tbody>
    </table>
  </div>
  {#if run.truncated}<p class="mt-2 text-muted-foreground">Showing the first 1,000 rows.</p>{/if}
{/if}
