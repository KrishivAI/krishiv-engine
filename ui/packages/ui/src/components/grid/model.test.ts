// SPDX-License-Identifier: Apache-2.0
import { describe, expect, test } from "vitest";
import { compareNumber, gridModel, nextSort, spacers, virtualises, type GridColumn } from "./model";

interface Compute {
  name: string;
  size: string;
  queries: number | null;
}

const ROWS: Compute[] = [
  { name: "default", size: "XS", queries: 1 },
  { name: "etl-m", size: "M", queries: 0 },
  { name: "bi", size: "S", queries: null },
  { name: "adhoc", size: "L", queries: 12 },
];
const COLUMNS: GridColumn<Compute>[] = [
  { id: "name", header: "Compute", value: (r) => r.name },
  { id: "size", header: "Size", value: (r) => r.size, sortable: false },
  { id: "queries", header: "Queries now", type: "int", num: true, value: (r) => r.queries },
];
const ids = (s: Parameters<typeof gridModel<Compute>>[3]) => gridModel(ROWS, COLUMNS, (r) => r.name, s).rows.map((r) => r.id);

describe("gridModel", () => {
  test("keeps the rows' order without a sort", () => {
    expect(ids(null)).toEqual(["default", "etl-m", "bi", "adhoc"]);
  });
  test("sorts text and numbers both ways, null last in both", () => {
    expect(ids({ id: "name", desc: false })).toEqual(["adhoc", "bi", "default", "etl-m"]);
    expect(ids({ id: "queries", desc: false })).toEqual(["etl-m", "default", "adhoc", "bi"]);
    expect(ids({ id: "queries", desc: true })).toEqual(["adhoc", "default", "etl-m", "bi"]);
  });
  test("ignores a sort on a column that cannot sort or does not exist", () => {
    expect(ids({ id: "size", desc: false })).toEqual(["default", "etl-m", "bi", "adhoc"]);
    expect(ids({ id: "gone", desc: true })).toEqual(["default", "etl-m", "bi", "adhoc"]);
  });
  test("headers carry the label, type, alignment and aria-sort", () => {
    const h = gridModel(ROWS, COLUMNS, (r) => r.name, { id: "queries", desc: true }).headers;
    expect(h.map((x) => [x.id, x.label, x.type, x.num, x.ariaSort])).toEqual([
      ["name", "Compute", undefined, false, "none"],
      ["size", "Size", undefined, false, undefined],
      ["queries", "Queries now", "int", true, "descending"],
    ]);
  });
  test("a numeric column orders NaN above every number, +Infinity included, and null last (Spark's order)", () => {
    const temps = [3, NaN, 1, null, 2, NaN, Infinity, 0];
    const rows = temps.map((t, i) => ({ i, t }));
    const run = (desc: boolean, num: boolean) =>
      gridModel(rows, [{ id: "t", header: "t", num, value: (r) => r.t }], (r) => String(r.i), { id: "t", desc }).rows.map(
        (r) => r.row.t,
      );
    for (const num of [true, false]) {
      expect(run(false, num)).toEqual([0, 1, 2, 3, Infinity, NaN, NaN, null]);
      expect(run(true, num)).toEqual([NaN, NaN, Infinity, 3, 2, 1, 0, null]);
    }
  });
  test("compareNumber is a consistent comparator with NaN", () => {
    expect([compareNumber(NaN, 1), compareNumber(1, NaN), compareNumber(NaN, NaN)]).toEqual([1, -1, 0]);
    expect([compareNumber(Infinity, NaN), compareNumber(-0, 0), compareNumber(2n, 1n)]).toEqual([-1, 0, 1]);
  });
  test("a repeated natural key does not merge rows", () => {
    const rows = [...ROWS, ROWS[0]];
    expect(gridModel(rows, COLUMNS, (r, i) => `${r.name}#${i}`, null).rows).toHaveLength(5);
  });
});

describe("nextSort", () => {
  test("cycles none, ascending, descending, none", () => {
    const a = nextSort(null, "name");
    expect(a).toEqual({ id: "name", desc: false });
    const d = nextSort(a, "name");
    expect(d).toEqual({ id: "name", desc: true });
    expect(nextSort(d, "name")).toBeNull();
  });
  test("another column starts ascending", () => {
    expect(nextSort({ id: "name", desc: true }, "queries")).toEqual({ id: "queries", desc: false });
  });
});

describe("virtual window", () => {
  test("virtualises only past the threshold", () => {
    expect(virtualises(200)).toBe(false);
    expect(virtualises(201)).toBe(true);
    expect(virtualises(10, 5)).toBe(true);
  });
  test("spacers fill the rows above and below the window", () => {
    expect(spacers([], 1000)).toEqual({ top: 0, bottom: 0 });
    expect(spacers([{ start: 360, end: 396 }, { start: 396, end: 432 }], 3600)).toEqual({ top: 360, bottom: 3168 });
    expect(spacers([{ start: 0, end: 36 }], 36)).toEqual({ top: 0, bottom: 0 });
  });
});
