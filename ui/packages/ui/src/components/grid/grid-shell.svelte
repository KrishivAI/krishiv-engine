<script lang="ts" generics="Row">
  import { Virtualizer, elementScroll, observeElementOffset, observeElementRect } from "@tanstack/virtual-core";
  import type { Snippet } from "svelte";
  import { cn } from "../../utils";
  import { gridModel, nextSort, spacers, virtualises, VIRTUAL_AFTER, type GridColumn, type GridSort } from "./model";

  // The grid shell of the wireframe's k-grid: TanStack Table (core) sorts, TanStack Virtual keeps only the visible
  // rows in the page past VIRTUAL_AFTER rows. The header is sticky; numbers are right-aligned in mono; a null is
  // shown as an italic "null"; the selected row is tinted with a gold edge. The box scrolls, never the page. Row
  // positions are spacer rows whose height is set through Svelte's `style:` (CSSOM), so no style attribute is
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
    rowId: (row: Row, index: number) => string;
    /** The grid's accessible name (the region and the table). */
    label: string;
    /** A cell's content; without it the value is shown as text. */
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
  let visible = $state<{ index: number; start: number; end: number }[]>([]);
  let total = $state(0);

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
    const cleanup = v._didMount();
    v._willUpdate();
    visible = v.getVirtualItems().map((i) => ({ index: i.index, start: i.start, end: i.end }));
    total = v.getTotalSize();
    return cleanup;
  });

  const shown = $derived(virtual ? visible.map((w) => ({ at: w.index, r: model.rows[w.index] })).filter((x) => x.r) : model.rows.map((r, at) => ({ at, r })));
  const pad = $derived(virtual ? spacers(visible, total) : { top: 0, bottom: 0 });

  // A selectable row takes focus, and Enter or Space selects it, so selection does not need a mouse. A key
  // pressed on a link or button inside the row is left to that control.
  function pick(e: KeyboardEvent, id: string, row: Row) {
    if (e.target !== e.currentTarget || (e.key !== "Enter" && e.key !== " ")) return;
    e.preventDefault();
    onselect?.(id, row);
  }

  function text(v: unknown): string {
    return typeof v === "string" ? v : String(v);
  }
</script>

<!-- The box scrolls, so it takes focus: a keyboard can scroll it (axe scrollable-region-focusable). -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div
  bind:this={box}
  role="region"
  aria-label={label}
  tabindex="0"
  data-grid
  class={cn("relative max-h-[480px] min-w-0 max-w-full overflow-auto rounded-lg border border-border bg-card", className)}
>
  <table class="w-full border-separate border-spacing-0 text-sm" aria-label={label} aria-rowcount={model.rows.length + 1}>
    <thead class="sticky top-0 z-10">
      <tr aria-rowindex={1}>
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
          aria-selected={selected === undefined ? undefined : selected === r.id}
          class={cn(
            "group hover:[&>td]:bg-accent aria-selected:[&>td]:bg-accent aria-selected:[&>td:first-child]:shadow-[inset_2px_0_0_var(--color-primary)]",
            onselect && "cursor-pointer focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-ring",
          )}
          tabindex={onselect ? 0 : undefined}
          onclick={onselect ? () => onselect(r.id, r.row) : undefined}
          onkeydown={onselect ? (e) => pick(e, r.id, r.row) : undefined}
        >
          {#each columns as c (c.id)}
            {@const v = c.value(r.row)}
            <td class={cn("h-9 border-b border-border px-3 align-middle whitespace-nowrap", c.num && "mono text-right tabular-nums")}>
              {#if cell}
                {@render cell(r.row, c)}
              {:else if v === null || v === undefined}
                <span class="text-muted-foreground italic">null</span>
              {:else}
                {text(v)}
              {/if}
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
