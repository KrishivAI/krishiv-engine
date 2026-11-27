// SPDX-License-Identifier: Apache-2.0
import {
  constructTable,
  createSortedRowModel,
  rowSortingFeature,
  sortFn_alphanumeric,
  sortFn_basic,
  sortFn_datetime,
  sortFn_text,
  tableFeatures,
} from "@tanstack/table-core";
import { storeReactivityBindings } from "@tanstack/table-core/store-reactivity-bindings";

/** One column of a grid. */
export interface GridColumn<Row> {
  id: string;
  /** The column's name in the header. */
  header: string;
  /** The column's type, shown under the name in mono (a result grid: `bigint`, `string`). */
  type?: string;
  /** Right-aligned, mono, tabular figures. */
  num?: boolean;
  /** The cell's value; also what sorting compares. */
  value: (row: Row) => unknown;
  /** Clicking the header sorts by this column. Default true. */
  sortable?: boolean;
}

/** The sort of a grid: one column, ascending or descending, or none (the rows' own order). */
export type GridSort = { id: string; desc: boolean } | null;

/** A header as the shell draws it. */
export interface GridHeader {
  id: string;
  label: string;
  type?: string;
  num: boolean;
  sortable: boolean;
  /** The `aria-sort` of the header cell; absent on a column that cannot sort. */
  ariaSort?: "ascending" | "descending" | "none";
}

/** A row as the shell draws it, in display order. */
export interface GridRow<Row> {
  id: string;
  row: Row;
}

/** What a grid shell draws: its headers, and its rows sorted. */
export interface GridModel<Row> {
  headers: GridHeader[];
  rows: GridRow<Row>[];
}

const features = tableFeatures({
  coreReactivityFeature: storeReactivityBindings(),
  rowSortingFeature,
  sortedRowModel: createSortedRowModel(),
  sortFns: { alphanumeric: sortFn_alphanumeric, basic: sortFn_basic, datetime: sortFn_datetime, text: sortFn_text },
});

/**
 * The headers and the sorted rows of a grid, through TanStack Table's core (its sorted row model). A value that
 * is null or undefined sorts last in both directions.
 */
export function gridModel<Row>(
  rows: readonly Row[],
  columns: readonly GridColumn<Row>[],
  rowId: (row: Row, index: number) => string,
  sort: GridSort,
): GridModel<Row> {
  const known = sort && columns.some((c) => c.id === sort.id && c.sortable !== false) ? sort : null;
  // TanStack sees each row as its position, so a row type need not satisfy its `RowData` constraint.
  const items = rows.map((_, i) => ({ i }));
  const table = constructTable({
    features,
    data: items,
    columns: columns.map((c) => ({
      id: c.id,
      accessorFn: (it: { i: number }) => normalise(c.value(rows[it.i])),
      header: c.header,
      enableSorting: c.sortable !== false,
      sortUndefined: "last" as const,
    })),
    getRowId: (it: { i: number }) => rowId(rows[it.i], it.i),
    state: { sorting: known ? [known] : [] },
    onSortingChange: () => {},
  });
  const headers = columns.map((c): GridHeader => {
    const sortable = c.sortable !== false;
    const on = known?.id === c.id ? (known.desc ? "descending" : "ascending") : "none";
    return { id: c.id, label: c.header, type: c.type, num: !!c.num, sortable, ariaSort: sortable ? on : undefined };
  });
  return { headers, rows: table.getRowModel().rows.map((r) => ({ id: r.id, row: rows[r.original.i] })) };
}

/** Null becomes undefined, so TanStack's `sortUndefined: "last"` puts it at the end. */
function normalise(v: unknown): unknown {
  return v === null ? undefined : v;
}

/** The sort after a click on a header: none, then ascending, then descending, then none again. */
export function nextSort(sort: GridSort, id: string): GridSort {
  if (sort?.id !== id) return { id, desc: false };
  return sort.desc ? null : { id, desc: true };
}

/** Rows past this count are virtualised (only the visible window is in the page). */
export const VIRTUAL_AFTER = 200;

/** Whether a grid of `count` rows draws only its visible window. */
export function virtualises(count: number, after: number = VIRTUAL_AFTER): boolean {
  return count > after;
}

/** The heights of the empty rows above and below a virtual window, so the scroll bar measures every row. */
export function spacers(items: readonly { start: number; end: number }[], total: number): { top: number; bottom: number } {
  if (items.length === 0) return { top: 0, bottom: 0 };
  const top = Math.max(0, items[0].start);
  const bottom = Math.max(0, total - items[items.length - 1].end);
  return { top, bottom };
}
