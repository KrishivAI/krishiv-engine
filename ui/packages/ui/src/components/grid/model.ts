// SPDX-License-Identifier: Apache-2.0
import {
  constructSortFn,
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
  sortFns: {
    alphanumeric: sortFn_alphanumeric,
    basic: sortFn_basic,
    datetime: sortFn_datetime,
    text: sortFn_text,
    number: constructSortFn({ sort: (a: unknown, b: unknown) => compareNumber(a, b) }),
  },
});

/**
 * Spark's order of numbers: NaN is larger than every other value, +Infinity included, and equal to itself.
 * TanStack's `basic` comparator answers -1 both ways for NaN, which leaves `Array.prototype.sort` an inconsistent
 * comparator and can misorder the numbers around a NaN, not only the NaN rows.
 */
export function compareNumber(a: unknown, b: unknown): number {
  const nanA = typeof a === "number" && Number.isNaN(a);
  const nanB = typeof b === "number" && Number.isNaN(b);
  if (nanA || nanB) return nanA === nanB ? 0 : nanA ? 1 : -1;
  if (a === b) return 0;
  const x = a as number;
  const y = b as number;
  // A number and a bigint of the same value (`1`, `1n`) are equal: `<` and `>` compare across the two types
  // exactly, so neither holds. NaN is handled above.
  if (isNumeric(a) && isNumeric(b) && !(x < y) && !(x > y)) return 0;
  return x > y ? 1 : -1;
}

const isNumeric = (v: unknown): v is number | bigint => typeof v === "number" || typeof v === "bigint";

/** A column sorts as numbers when it says `num`, or when its first value that is not null is a number. */
function numeric<Row>(c: GridColumn<Row>, rows: readonly Row[]): boolean {
  if (c.num) return true;
  for (const r of rows) {
    const v = c.value(r);
    if (v !== null && v !== undefined) return typeof v === "number" || typeof v === "bigint";
  }
  return false;
}

/**
 * The headers and the sorted rows of a grid, through TanStack Table's core (its sorted row model). A value that
 * is null or undefined sorts last in both directions; a numeric column orders NaN above every number, as Spark
 * does. `rowId` must give each row a different id (the shell keys its rows on it).
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
      ...(numeric(c, rows) ? { sortFn: "number" as const } : {}),
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
