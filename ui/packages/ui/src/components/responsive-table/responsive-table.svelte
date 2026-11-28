<script lang="ts" module>
  export interface Column {
    key: string;
    label: string;
    /** Right-aligned, tabular figures. */
    num?: boolean;
    /** The card's heading: no label on a narrow screen. */
    title?: boolean;
    /** Free text of any length (a principal's name): wraps in its cell, so one long value cannot push the columns after it out of the table's box. */
    wrap?: boolean;
  }
</script>

<script lang="ts" generics="Row">
  import type { Snippet } from "svelte";
  import { Card } from "../card";
  import { Table, TableCell, TableHead } from "../table";
  import { cn } from "../../utils";

  let {
    columns,
    rows,
    rowKey,
    cell,
    actions,
    empty = "Nothing to show.",
    rowClass,
  }: {
    columns: Column[];
    rows: Row[];
    rowKey: (row: Row) => string | number;
    cell: Snippet<[Row, Column]>;
    actions?: Snippet<[Row]>;
    empty?: string;
    rowClass?: (row: Row) => string;
  } = $props();
</script>

<!-- From the xl breakpoint (1280 px), where every column fits, a table; below it one card per row, two cards
     side by side on a tablet. A table narrower than its columns would hide the last ones (the row's action
     among them) behind a sideways scroll. -->
<div class="hidden xl:block">
  <Table>
    <thead>
      <tr>
        {#each columns as c (c.key)}
          <TableHead class={c.num ? "text-right" : ""}>{c.label}</TableHead>
        {/each}
        {#if actions}<TableHead></TableHead>{/if}
      </tr>
    </thead>
    <tbody>
      {#each rows as r (rowKey(r))}
        <tr class={cn("last:[&>td]:border-b-0", rowClass?.(r))}>
          {#each columns as c (c.key)}
            <TableCell class={c.num ? "mono text-right tabular-nums" : c.title ? "max-w-md min-w-56 whitespace-normal wrap-anywhere" : c.wrap ? "max-w-64 min-w-24 whitespace-normal wrap-anywhere" : ""}>
              {@render cell(r, c)}
            </TableCell>
          {/each}
          {#if actions}<TableCell>{@render actions(r)}</TableCell>{/if}
        </tr>
      {:else}
        <tr><TableCell class="text-muted-foreground" colspan={columns.length + (actions ? 1 : 0)}>{empty}</TableCell></tr>
      {/each}
    </tbody>
  </Table>
</div>

<ul class="grid gap-3 md:grid-cols-2 xl:hidden">
  {#each rows as r (rowKey(r))}
    <li>
      <Card class={cn("grid grid-cols-1 gap-1.5 p-3", rowClass?.(r))}>
        {#each columns as c (c.key)}
          {#if c.title}
            <div class="wrap-anywhere">{@render cell(r, c)}</div>
          {:else}
            <div class="flex flex-wrap items-baseline justify-between gap-x-4">
              <span class="text-xs font-bold tracking-[0.06em] text-muted-foreground uppercase">{c.label}</span>
              <span class={cn("ml-auto max-w-full min-w-0 text-right wrap-anywhere", c.num && "mono tabular-nums")}>{@render cell(r, c)}</span>
            </div>
          {/if}
        {/each}
        {#if actions}<div class="pt-1">{@render actions(r)}</div>{/if}
      </Card>
    </li>
  {:else}
    <li class="text-muted-foreground">{empty}</li>
  {/each}
</ul>
