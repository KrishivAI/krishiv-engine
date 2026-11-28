<script lang="ts" generics="Row">
  import { Virtualizer, elementScroll, observeElementOffset, observeElementRect } from "@tanstack/virtual-core";
  import { tick, type Snippet } from "svelte";
  import { cn } from "../../utils";
  import { gridModel, nextSort, spacers, virtualises, VIRTUAL_AFTER, type GridColumn, type GridSort } from "./model";

  // The grid shell of the wireframe's k-grid: TanStack Table (core) sorts, TanStack Virtual keeps only the visible
  // rows in the page past VIRTUAL_AFTER rows. The header is sticky; numbers are right-aligned in mono; a null is
  // shown as an italic "null"; the selected row is tinted with a gold edge. The box scrolls, never the page. Row
  // positions are spacer rows, and every row is `rowHeight` tall (the height the virtualizer counts with); both
  // heights are set through Svelte's `style:` (CSSOM), so no style attribute is
  // written and the grid works under `style-src 'self'`.
  let {
    columns,
    rows,
    rowId,
    label,
    cell,
    foot,
    empty,
    selected,
    onselect,
    sort = $bindable(null),
    rowHeight = 36,
    virtualAfter = VIRTUAL_AFTER,
    class: className,
  }: {
    columns: GridColumn<Row>[];
    rows: Row[];
    /** A different id for each row: the shell keys its rows on it (a repeated id stops the grid rendering). */
    rowId: (row: Row, index: number) => string;
    /** The grid's accessible name (the region and the table). */
    label: string;
    /** A cell's content; without it the value is shown as text. The cell lays its children out in a row with a
     * small gap (a flex box), so put text and a badge or link side by side as separate elements, not spaced text. */
    cell?: Snippet<[Row, GridColumn<Row>]>;
    foot?: Snippet;
    empty?: Snippet;
    selected?: string;
    onselect?: (id: string, row: Row) => void;
    sort?: GridSort;
    rowHeight?: number;
    virtualAfter?: number;
    class?: string;
  } = $props();

  const model = $derived(gridModel(rows, columns, rowId, sort));
  const virtual = $derived(virtualises(model.rows.length, virtualAfter));

  let box: HTMLDivElement | undefined = $state();
  // The sticky header's height, measured on its row (`offsetHeight`; how a row group reports its client box
  // differs by engine): the box keeps that much scroll padding at its top, so a row that takes the focus (the
  // browser's scroll on focus, in a grid that is not virtual) is scrolled clear of the header, not under it.
  let headHeight = $state(0);
  let visible = $state<{ index: number; start: number; end: number }[]>([]);
  let total = $state(0);
  let scroller: Virtualizer<HTMLDivElement, HTMLTableRowElement> | undefined;

  $effect(() => {
    const count = model.rows.length;
    const height = rowHeight;
    if (!virtual || !box) return;
    const el = box;
    const v = new Virtualizer<HTMLDivElement, HTMLTableRowElement>({
      count,
      getScrollElement: () => el,
      estimateSize: () => height,
      overscan: 12,
      scrollToFn: elementScroll,
      observeElementRect,
      observeElementOffset,
      onChange: (inst) => {
        visible = inst.getVirtualItems().map((i) => ({ index: i.index, start: i.start, end: i.end }));
        total = inst.getTotalSize();
      },
    });
    scroller = v;
    const cleanup = v._didMount();
    v._willUpdate();
    visible = v.getVirtualItems().map((i) => ({ index: i.index, start: i.start, end: i.end }));
    total = v.getTotalSize();
    return () => {
      scroller = undefined;
      cleanup();
    };
  });

  const shown = $derived(virtual ? visible.map((w) => ({ at: w.index, r: model.rows[w.index] })).filter((x) => x.r) : model.rows.map((r, at) => ({ at, r })));
  const pad = $derived(virtual ? spacers(visible, total) : { top: 0, bottom: 0 });

  // Selectable rows are one tab stop (a roving tabindex): the selected row, else the last row focused, else the
  // first. ArrowUp and ArrowDown move between rows, Home and End to the ends; Enter or Space selects. A key
  // pressed on a link or button inside the row is left to that control.
  // The last row focused is kept by its id, so a change of sort keeps the tab stop on the same row.
  let activeId = $state<string | undefined>(undefined);
  const stop = $derived.by(() => {
    const find = (id: string | undefined) => (id === undefined ? -1 : model.rows.findIndex((r) => r.id === id));
    const at = find(selected);
    const was = find(activeId);
    const want = at >= 0 ? at : was >= 0 ? was : 0;
    return shown.some((x) => x.at === want) ? want : (shown[0]?.at ?? 0);
  });

  async function pick(e: KeyboardEvent, at: number, id: string, row: Row) {
    if (e.target !== e.currentTarget) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      activeId = id;
      onselect?.(id, row);
      return;
    }
    const last = model.rows.length - 1;
    const to = e.key === "ArrowDown" ? at + 1 : e.key === "ArrowUp" ? at - 1 : e.key === "Home" ? 0 : e.key === "End" ? last : -1;
    if (to < 0 || to > last || (to === at && e.key !== "Home" && e.key !== "End")) return;
    e.preventDefault();
    activeId = model.rows[to]?.id;
    scroller?.scrollToIndex(to, { align: "auto" });
    // A virtual row is drawn after the scroll reaches it: wait a few frames for it.
    for (let i = 0; i < 10; i++) {
      await tick();
      const tr = box?.querySelector<HTMLTableRowElement>(`tbody tr[aria-rowindex="${to + 2}"]`);
      if (tr) return tr.focus();
      await new Promise((ok) => requestAnimationFrame(ok));
    }
  }

  // A struct, array or map is shown as its JSON until the SQL editor view brings typed cells (README).
  function text(v: unknown): string {
    if (Array.isArray(v) || (v !== null && typeof v === "object" && Object.getPrototypeOf(v) === Object.prototype)) {
      try {
        return JSON.stringify(v, (_, x) => (typeof x === "bigint" ? x.toString() : x));
      } catch {
        return String(v);
      }
    }
    return typeof v === "string" ? v : String(v);
  }
</script>

<!-- The box scrolls, so it takes focus: a keyboard can scroll it (axe scrollable-region-focusable). -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div
  bind:this={box}
  style:scroll-padding-top={`${headHeight}px`}
  role="region"
  aria-label={label}
  tabindex="0"
  data-grid
  class={cn("relative max-h-[480px] min-w-0 max-w-full overflow-auto rounded-lg border border-border bg-card", className)}
>
  <table class="w-full border-separate border-spacing-0 text-sm" aria-label={label} aria-rowcount={model.rows.length + 1}>
    <thead class="sticky top-0 z-10">
      <tr aria-rowindex={1} bind:offsetHeight={headHeight}>
        {#each model.headers as h (h.id)}
          <th
            scope="col"
            aria-sort={h.ariaSort}
            class={cn(
              "h-9 border-b border-border bg-muted px-3 text-left align-middle text-xs font-bold tracking-[0.06em] whitespace-nowrap text-muted-foreground uppercase",
              h.type && "h-11.5",
              h.num && "text-right",
            )}
          >
            {#if h.sortable}
              <button
                type="button"
                class={cn("inline-flex min-h-9 cursor-pointer items-center gap-1 bg-transparent font-bold tracking-[0.06em] uppercase hover:text-foreground", h.num && "flex-row-reverse")}
                onclick={() => (sort = nextSort(sort, h.id))}
              >
                <span class="grid text-left">
                  {h.label}
                  {#if h.type}<span class="font-mono text-[11px] leading-3.5 font-normal tracking-normal normal-case">{h.type}</span>{/if}
                </span>
                <i class="w-3 not-italic" aria-hidden="true">{h.ariaSort === "ascending" ? "▲" : h.ariaSort === "descending" ? "▼" : ""}</i>
              </button>
            {:else}
              <span class="grid">
                {h.label}
                {#if h.type}<span class="font-mono text-[11px] leading-3.5 font-normal tracking-normal normal-case">{h.type}</span>{/if}
              </span>
            {/if}
          </th>
        {/each}
      </tr>
    </thead>
    <tbody>
      {#if pad.top > 0}
        <tr aria-hidden="true"><td colspan={columns.length} class="p-0" style:height={`${pad.top}px`}></td></tr>
      {/if}
      {#each shown as { at, r } (r.id)}
        <tr
          aria-rowindex={at + 2}
          aria-current={selected !== undefined && selected === r.id ? "true" : undefined}
          class={cn(
            "group hover:[&>td]:bg-accent aria-[current=true]:[&>td]:bg-accent aria-[current=true]:[&>td:first-child]:shadow-[inset_2px_0_0_var(--color-primary)]",
            onselect && "cursor-pointer focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-ring",
          )}
          tabindex={onselect ? (at === stop ? 0 : -1) : undefined}
          onclick={onselect ? () => ((activeId = r.id), onselect(r.id, r.row)) : undefined}
          onkeydown={onselect ? (e) => pick(e, at, r.id, r.row) : undefined}
        >
          {#each columns as c (c.id)}
            {@const v = c.value(r.row)}
            <td class={cn("border-b border-border px-3 align-middle whitespace-nowrap", c.num && "mono text-right tabular-nums")} style:height={`${rowHeight}px`}>
              <!-- A cell's content never grows the row past rowHeight (less its 1 px border): the virtualizer counts
                   every row at that height, so a taller cell would make the window jump while scrolling. The
                   wrapper is the full height and clips only vertically, so a control of normal height keeps room
                   for its focus ring above and below, and nothing is clipped sideways. -->
              <div class={cn("flex items-center gap-1 overflow-y-clip", c.num && "justify-end")} style:height={`${rowHeight - 1}px`}>
                {#if cell}
                  {@render cell(r.row, c)}
                {:else if v === null || v === undefined}
                  <span class="text-muted-foreground italic">null</span>
                {:else}
                  {text(v)}
                {/if}
              </div>
            </td>
          {/each}
        </tr>
      {:else}
        <tr><td colspan={columns.length} class="px-3 py-2.5 text-muted-foreground">{#if empty}{@render empty()}{:else}Nothing to show.{/if}</td></tr>
      {/each}
      {#if pad.bottom > 0}
        <tr aria-hidden="true"><td colspan={columns.length} class="p-0" style:height={`${pad.bottom}px`}></td></tr>
      {/if}
    </tbody>
  </table>
  {#if foot}
    <div class="sticky left-0 flex flex-wrap items-center gap-3 border-t border-border px-3 py-2 text-[13px] text-muted-foreground">{@render foot()}</div>
  {/if}
</div>
