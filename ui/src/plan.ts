// SPDX-License-Identifier: Apache-2.0
import type { Operator, PlanNode } from "./api";

/** One row of the drawn plan: a node at a depth, with the operators that ran it. */
export interface PlanRow {
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
    rows.push({ node, depth, operators: byNode.get(node.id) ?? [] });
    for (const c of node.children) walk(c, depth + 1);
  };
  if (plan.length) walk(0, 0);
  return rows;
}

/** Operators no plan node claims (a sink, a shuffle): shown below the tree, never dropped. */
export function unplaced(plan: PlanNode[], operators: Operator[]): Operator[] {
  const ids = new Set(plan.map((n) => n.id));
  return operators.filter((o) => o.node_id == null || !ids.has(o.node_id));
}

/**
 * The value of one metric over the operators of a node: the sum of those that reported it, `null` when none did.
 * "Not reported" is never 0 (a missing figure and a zero are different facts).
 */
export function nodeValue(operators: Operator[], value: (o: Operator) => number | null): number | null {
  let total: number | null = null;
  for (const o of operators) {
    const v = value(o);
    if (v != null) total = (total ?? 0) + v;
  }
  return total;
}
