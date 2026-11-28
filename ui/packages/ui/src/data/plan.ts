// SPDX-License-Identifier: Apache-2.0
import type { Operator, PlanNode } from "./types";
import { bytes, count, nanos } from "./format";

/** One row of the drawn plan: a node at a depth, with the operators that ran it. */
export interface PlanRow {
  /** The node's index in the plan array: unique, where `node.id` can be null more than once. */
  at: number;
  node: PlanNode;
  depth: number;
  operators: Operator[];
}

/** The plan flattened depth first (the root first), each node with its operators joined by `node_id`. */
export function planRows(plan: PlanNode[], operators: Operator[]): PlanRow[] {
  const byNode = new Map<number, Operator[]>();
  for (const o of operators) {
    if (o.node_id == null) continue;
    byNode.set(o.node_id, [...(byNode.get(o.node_id) ?? []), o]);
  }
  const rows: PlanRow[] = [];
  const seen = new Set<number>();
  const walk = (at: number, depth: number) => {
    const node = plan[at];
    if (!node || seen.has(at)) return; // a malformed document never loops the page
    seen.add(at);
    // A heading (a subquery's, a shared plan's) has no id and so no operator.
    rows.push({ at, node, depth, operators: node.id == null ? [] : (byNode.get(node.id) ?? []) });
    for (const c of node.children ?? []) walk(c, depth + 1);
  };
  if (plan.length) walk(0, 0);
  return rows;
}

/** Operators no drawn node claims (the result sink, a node the tree does not reach): listed below the tree. */
export function unplaced(rows: PlanRow[], operators: Operator[]): Operator[] {
  const ids = new Set(rows.map((r) => r.node.id).filter((id) => id != null));
  return operators.filter((o) => o.node_id == null || !ids.has(o.node_id));
}

const isScan = (o: Operator) => o.role === "source" && o.name.endsWith("Scan");
const positive = (n: number | undefined) => (n ?? 0) > 0;
/** `null` when every part is; a part that is not reported adds nothing. */
const total = (...parts: (number | null)[]) => (parts.every((p) => p == null) ? null : parts.reduce<number>((a, p) => a + (p ?? 0), 0));

/**
 * What an operator measured, by counter and role. The engine writes every counter as an integer, so one an
 * operator cannot measure arrives as 0; these readers give `null` ("not reported") for it instead:
 * rows in does not apply to a source nor rows out to a sink, only a scan waits on I/O, and only an operator
 * that reserved memory has a peak or can spill. A positive figure is always shown.
 */
export const reported = {
  rowsIn: (o: Operator) => (o.role === "source" ? null : (o.rows_in ?? null)),
  rowsOut: (o: Operator) => (o.role === "sink" ? null : (o.rows_out ?? null)),
  compute: (o: Operator) => o.compute_ns ?? null,
  ioWait: (o: Operator) => (isScan(o) || positive(o.blocked_io_ns) ? (o.blocked_io_ns ?? null) : null),
  memoryWait: (o: Operator) => o.blocked_memory_ns ?? null,
  sinkWait: (o: Operator) => o.blocked_sink_ns ?? null,
  wait: (o: Operator) => total(reported.ioWait(o), reported.memoryWait(o), reported.sinkWait(o)),
  memory: (o: Operator) => (positive(o.peak_reserved_bytes) ? (o.peak_reserved_bytes ?? null) : null),
  spill: (o: Operator) => (positive(o.peak_reserved_bytes) || positive(o.spilled_bytes) ? (o.spilled_bytes ?? null) : null),
};

/**
 * A figure the plan can be shaded by. `over` is how a node's operators combine (a join has a build and a
 * probe, an aggregate a sink and a source): times and spill add up, memory is the largest, rows out are the
 * source side's.
 */
export interface Metric {
  key: string;
  label: string;
  over: "sum" | "max" | "source";
  value: (o: Operator) => number | null;
  show: (n: number) => string;
  /** Operator times are summed over drivers, so they do not add up to the wall time. */
  time?: boolean;
}

export const METRICS: Metric[] = [
  { key: "compute", label: "Compute time", over: "sum", value: reported.compute, show: nanos, time: true },
  { key: "rows", label: "Rows out", over: "source", value: reported.rowsOut, show: count },
  { key: "memory", label: "Peak memory", over: "max", value: reported.memory, show: bytes },
  { key: "wait", label: "Waiting time", over: "sum", value: reported.wait, show: nanos, time: true },
  { key: "spill", label: "Spilled bytes", over: "sum", value: reported.spill, show: bytes },
];

/**
 * The value of `metric` for a node's operators, `null` when none reported it. "Not reported" is never 0: a
 * missing figure and a measured zero are different facts.
 */
export function nodeValue(operators: Operator[], metric: Metric): number | null {
  const all = operators.map((o) => ({ role: o.role, v: metric.value(o) })).filter((x): x is { role: string | undefined; v: number } => x.v != null);
  if (!all.length) return null;
  const sources = all.filter((x) => x.role === "source");
  // A node with no source operator (a filter, a projection) has one operator that reports rows out; if it had
  // several, their rows add up.
  const taken = (metric.over === "source" && sources.length ? sources : all).map((x) => x.v);
  return metric.over === "max" ? Math.max(...taken) : taken.reduce((a, v) => a + v, 0);
}

/** One line of the heatmap: a plan node, or an operator outside the tree. */
export interface HeatRow {
  key: string;
  depth: number;
  /** The node's id in the plan, or the operator's id when there is no tree. */
  id: number | null;
  name: string;
  value: number | null;
  /** `none`: not reported (no colour); `zero`: a measured zero (its own look); `value`: shaded. */
  state: "none" | "zero" | "value";
  /** The shade, 0 to 100, on one linear scale from zero to the plan's largest value. */
  heat: number;
  /** The value's share of the plan's total, for a metric that adds up; `null` otherwise. */
  share: number | null;
}

/**
 * The heatmap of `metric`: the plan's nodes in tree order, then the operators outside it; with no plan, the
 * operators in the order given. Returned as data so the ranking can be read without the colours.
 */
export function heatRows(plan: PlanNode[] | null | undefined, operators: Operator[], metric: Metric): HeatRow[] {
  const tree = plan?.length ? planRows(plan, operators) : [];
  const loose = plan?.length ? unplaced(tree, operators) : operators;
  const rows = [
    ...tree.map((r) => ({ key: `n${r.at}`, depth: r.depth, id: r.node.id ?? null, name: r.node.name, value: nodeValue(r.operators, metric) })),
    ...loose.map((o) => ({ key: `o${o.operator_id}`, depth: 0, id: plan?.length ? null : o.operator_id, name: o.name, value: nodeValue([o], metric) })),
  ];
  const largest = Math.max(0, ...rows.map((r) => r.value ?? 0));
  const sum = metric.over === "sum" ? rows.reduce((a, r) => a + (r.value ?? 0), 0) : 0;
  return rows.map((r) => ({
    ...r,
    state: r.value == null ? "none" : r.value === 0 ? "zero" : "value",
    heat: r.value && largest > 0 ? Math.round((r.value / largest) * 100) : 0,
    share: r.value != null && sum > 0 ? r.value / sum : null,
  }));
}
