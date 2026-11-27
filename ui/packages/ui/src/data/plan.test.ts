// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";
import { heatRows, METRICS, nodeValue, planRows, reported, unplaced } from "./plan";
import type { Operator, PlanNode } from "./types";

const metric = (key: string) => {
  const m = METRICS.find((x) => x.key === key);
  if (!m) throw new Error(`no metric ${key}`);
  return m;
};

// HashJoin (id 1) over a scan (id 2) and a heading (no id) whose child is another scan (id 3).
const plan: PlanNode[] = [
  { id: 1, name: "HashJoin", children: [1, 2] },
  { id: 2, name: "ParquetScan a", children: [] },
  { id: null, name: "Subquery", children: [3] },
  { id: 3, name: "ParquetScan b", children: [] },
];
const op = (o: Partial<Operator> & { operator_id: number; name: string }): Operator => ({ ...o });
const operators: Operator[] = [
  op({ operator_id: 10, node_id: 1, name: "HashJoinBuild", role: "sink", compute_ns: 30, peak_reserved_bytes: 100 }),
  op({ operator_id: 11, node_id: 1, name: "HashJoinProbe", role: "source", compute_ns: 20, rows_out: 7, peak_reserved_bytes: 40 }),
  op({ operator_id: 12, node_id: 2, name: "ParquetScan", role: "source", compute_ns: 50, rows_out: 9, blocked_io_ns: 0 }),
  op({ operator_id: 13, node_id: 3, name: "ParquetScan", role: "source", compute_ns: 0, rows_out: 0 }),
  op({ operator_id: 14, node_id: null, name: "ResultSink", role: "sink", compute_ns: 5 }),
];

describe("planRows and unplaced", () => {
  it("walks the plan depth first and joins operators by node id; a heading has none", () => {
    const rows = planRows(plan, operators);
    expect(rows.map((r) => [r.at, r.depth, r.operators.map((o) => o.operator_id)])).toEqual([
      [0, 0, [10, 11]],
      [1, 1, [12]],
      [2, 1, []],
      [3, 2, [13]],
    ]);
    expect(unplaced(rows, operators).map((o) => o.operator_id)).toEqual([14]);
  });
  it("never loops on a malformed plan", () => {
    const cyclic: PlanNode[] = [
      { id: 1, name: "a", children: [1] },
      { id: 2, name: "b", children: [0, 5] },
    ];
    expect(planRows(cyclic, []).map((r) => r.at)).toEqual([0, 1]);
    expect(planRows([], operators)).toEqual([]);
  });
});

describe("reported", () => {
  it("gives null where a counter does not apply, not 0", () => {
    const sink = operators[0];
    const scan = operators[2];
    expect(reported.rowsOut(sink)).toBeNull();
    expect(reported.rowsIn(scan)).toBeNull();
    expect(reported.ioWait(scan)).toBe(0); // a scan waits on I/O: a measured zero
    expect(reported.ioWait(sink)).toBeNull();
    expect(reported.memory(scan)).toBeNull();
    expect(reported.spill(sink)).toBeNull(); // reserved memory but no spill counter sent
    expect(reported.wait(sink)).toBeNull();
  });
});

describe("nodeValue", () => {
  const join = operators.slice(0, 2);
  it("adds times, takes the largest memory and the source side's rows", () => {
    expect(nodeValue(join, metric("compute"))).toBe(50);
    expect(nodeValue(join, metric("memory"))).toBe(100);
    expect(nodeValue(join, metric("rows"))).toBe(7);
  });
  it("is null when no operator reported the figure", () => {
    expect(nodeValue([operators[3]], metric("memory"))).toBeNull();
    expect(nodeValue([], metric("compute"))).toBeNull();
  });
});

describe("heatRows", () => {
  it("shades on one scale from zero to the largest value and gives shares of a sum", () => {
    const rows = heatRows(plan, operators, metric("compute"));
    expect(rows.map((r) => [r.key, r.value, r.state, r.heat])).toEqual([
      ["n0", 50, "value", 100],
      ["n1", 50, "value", 100],
      ["n2", null, "none", 0],
      ["n3", 0, "zero", 0],
      ["o14", 5, "value", 10],
    ]);
    expect(rows[0].share).toBeCloseTo(50 / 105);
    expect(rows[2].share).toBeNull();
  });
  it("has no share for a metric that does not add up, and lists operators alone without a plan", () => {
    expect(heatRows(plan, operators, metric("memory")).every((r) => r.share === null)).toBe(true);
    const flat = heatRows(null, operators.slice(2, 4), metric("rows"));
    expect(flat.map((r) => [r.key, r.id, r.value])).toEqual([
      ["o12", 12, 9],
      ["o13", 13, 0],
    ]);
  });
});
