<script lang="ts">
  import { EditorView, minimalSetup } from "codemirror";
  import { EditorState, Prec } from "@codemirror/state";
  import { keymap } from "@codemirror/view";
  import { sql } from "@codemirror/lang-sql";
  import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
  import { tags } from "@lezer/highlight";
  import { api, type SqlRun } from "../api";
  import { consoleInfo } from "../console.svelte";
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
  // from the page's custom properties, which cross the shadow boundary, so the text keeps the page's own contrast
  // in dark and light.
  const theme = EditorView.theme({
    "&": { color: "var(--foreground)", backgroundColor: "var(--card)", minHeight: "8rem", fontSize: "14px" },
    // A long statement scrolls inside the editor, so the page's own controls stay where they are.
    ".cm-scroller": { maxHeight: "50dvh", overflow: "auto" },
    ".cm-content": { fontFamily: "'Red Hat Mono', ui-monospace, monospace", caretColor: "var(--foreground)", padding: "8px 4px" },
    "&.cm-focused": { outline: "2px solid var(--k-accent)" },
    ".cm-cursor": { borderLeftColor: "var(--foreground)" },
    ".cm-selectionBackground, &.cm-focused .cm-selectionBackground": { backgroundColor: "color-mix(in srgb, var(--foreground) 25%, transparent)" },
  });

  // Syntax colours are the page's status colours, which are legible on a surface in both schemes. Without this
  // style CodeMirror's default one applies (minimalSetup carries it as a fallback): fixed colours chosen for a
  // white page, whose keyword purple is too dark to read on the dark surface.
  const highlight = HighlightStyle.define([
    { tag: tags.keyword, color: "var(--k-info)", fontWeight: "600" },
    { tag: [tags.string, tags.special(tags.string)], color: "var(--k-success)" },
    { tag: [tags.number, tags.bool, tags.null], color: "var(--k-warning)" },
    { tag: tags.comment, color: "var(--k-muted)", fontStyle: "italic" },
  ]);

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
          syntaxHighlighting(highlight),
          sql(),
          EditorView.contentAttributes.of({ "aria-label": "SQL statement" }),
          // Ahead of the default keymap, whose Mod-Enter adds a blank line and would win otherwise.
          Prec.highest(keymap.of([{ key: "Mod-Enter", run: () => (start(), true) }])),
          // Typing over a selection that spans highlighted words: Chromium's own editing first writes a style
          // attribute onto the text it keeps, which `style-src 'self'` blocks (one console error per key, nothing
          // to see). The editor makes the change itself, so the browser's editing never runs. Composed text (an
          // input method) is not `insertText` and stays the browser's.
          EditorView.domEventHandlers({
            beforeinput(event, view) {
              if (event.inputType !== "insertText" || event.data == null || view.composing) return false;
              if (view.state.readOnly || view.state.selection.ranges.every((r) => r.empty)) return false;
              event.preventDefault();
              view.dispatch(view.state.replaceSelection(event.data), { userEvent: "input.type", scrollIntoView: true });
              return true;
            },
          }),
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
  // A value longer than this is cut short in its cell; a tap or a click on it shows it whole.
  const LONG = 120;
  let whole = $state<string[]>([]);
  const toggle = (key: string) => (whole = whole.includes(key) ? whole.filter((k) => k !== key) : [...whole, key]);
  // A new result starts with every long value cut short again.
  $effect(() => {
    void runId;
    whole = [];
  });
</script>

<h1 class="mb-4">SQL</h1>
{#if consoleInfo.mode === "embedded"}
  <p class="mb-3 text-muted-foreground">
    Statements run in your session: its temporary views, settings and variables are here, and what a statement
    changes stays changed for your program.
  </p>
{/if}
<!-- Editor and result side by side on a very wide screen, one under the other elsewhere. -->
<div class="min-[2200px]:grid min-[2200px]:grid-cols-2 min-[2200px]:items-start min-[2200px]:gap-6">
  <div class="min-w-0">
    <!-- Above the editor and held under the top bar: Run and Cancel stay on the screen under a long statement and
         with a phone's keyboard up. -->
    <div class="sticky top-14 z-20 flex flex-wrap items-center gap-3 bg-background py-2">
      <Button onclick={start} disabled={run?.state === "running"}>Run</Button>
      <Button variant="destructive" onclick={cancel} disabled={run?.state !== "running"}>Cancel</Button>
      {#if run}<Badge variant={stateVariant(run.state)}>{run.state}</Badge>{/if}
      <span class="text-muted-foreground pointer-coarse:hidden">Ctrl or Cmd + Enter runs the statement.</span>
    </div>
    <div bind:this={host} class="mb-3 overflow-hidden rounded-md border border-border"></div>
    {#if message}<p class="mb-3 text-destructive wrap-anywhere">{message}</p>{/if}
    {#if run?.error && run.state !== "cancelled"}
      <p class="mb-3 wrap-anywhere" role="alert">{run.error_class ? `[${run.error_class}] ` : ""}{run.error}</p>
    {/if}
  </div>
  <div class="min-w-0">
    {#if run?.columns}
      <!-- The result scrolls inside its own box, both ways; its header row stays in view. -->
      <div data-table-box class="max-h-[70dvh] max-w-full overflow-auto rounded-lg border border-border bg-card">
        <table class="w-full border-collapse text-left" aria-label="Result">
          <thead>
            <tr>{#each run.columns as c, i (i)}<th class="sticky top-0 border-b border-border bg-muted px-3.5 py-2.5 whitespace-nowrap">{c}</th>{/each}</tr>
          </thead>
          <tbody>
            {#each run.rows ?? [] as r, i (i)}
              <tr>
                {#each r as v, j (j)}
                  {@const s = cellText(v)}
                  {@const key = `${i}/${j}`}
                  <td class="mono max-w-[60ch] border-b border-border px-3.5 py-1.5 align-top wrap-anywhere">
                    {#if s.length > LONG}
                      <button type="button" class="min-h-11 min-w-11 cursor-pointer bg-transparent p-0 text-left wrap-anywhere" aria-expanded={whole.includes(key)} onclick={() => toggle(key)}>
                        {whole.includes(key) ? s : `${s.slice(0, LONG)}…`}<span class="sr-only"> {whole.includes(key) ? "Show less" : "Show the whole value"}</span>
                      </button>
                    {:else}{s}{/if}
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      {#if run.truncated}<p class="mt-2 text-muted-foreground">Showing the first 1,000 rows.</p>{/if}
    {/if}
  </div>
</div>
