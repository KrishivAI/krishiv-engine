// SPDX-License-Identifier: Apache-2.0
import type { Table, Schema, RecordBatch } from "apache-arrow";
import type { Backend, ResultStream } from "./backend.js";
import { collectTable, toObjects, type ObjectOptions, type Row } from "./result.js";

// Always backtick-quoted, so reserved words (`select`, `order`) work as column names.
const q = (name: string): string => "`" + name.replaceAll("`", "``") + "`";

/** A SQL expression. Built with `col`, `lit` and the functions; combine with the methods. */
export class Column {
  constructor(readonly sql: string) {}
  private bin(op: string, other: Column | string | number | boolean | bigint | null): Column {
    return new Column(`(${this.sql} ${op} ${lit(other).sql})`);
  }
  eq(o: Parameters<Column["bin"]>[1]) { return this.bin("=", o); }
  ne(o: Parameters<Column["bin"]>[1]) { return this.bin("<>", o); }
  gt(o: Parameters<Column["bin"]>[1]) { return this.bin(">", o); }
  ge(o: Parameters<Column["bin"]>[1]) { return this.bin(">=", o); }
  lt(o: Parameters<Column["bin"]>[1]) { return this.bin("<", o); }
  le(o: Parameters<Column["bin"]>[1]) { return this.bin("<=", o); }
  plus(o: Parameters<Column["bin"]>[1]) { return this.bin("+", o); }
  minus(o: Parameters<Column["bin"]>[1]) { return this.bin("-", o); }
  times(o: Parameters<Column["bin"]>[1]) { return this.bin("*", o); }
  divide(o: Parameters<Column["bin"]>[1]) { return this.bin("/", o); }
  and(o: Column) { return this.bin("AND", o); }
  or(o: Column) { return this.bin("OR", o); }
  not() { return new Column(`(NOT ${this.sql})`); }
  isNull() { return new Column(`(${this.sql} IS NULL)`); }
  isNotNull() { return new Column(`(${this.sql} IS NOT NULL)`); }
  alias(name: string) { return new Column(`${this.sql} AS ${q(name)}`); }
  asc() { return new Column(`${this.sql} ASC`); }
  desc() { return new Column(`${this.sql} DESC`); }
}

export function col(name: string): Column {
  return new Column(name === "*" ? "*" : name.split(".").map(q).join("."));
}

/**
 * A literal; strings are quoted and escaped, `null` is NULL. Escaping follows the default parser
 * (`spark.sql.parser.escapedStringLiterals=false`); with that setting on, build the text yourself,
 * since a backslash there is kept as written and the server refuses quote and backslash values.
 */
export function lit(v: Column | string | number | boolean | bigint | null): Column {
  if (v instanceof Column) return v;
  if (v === null) return new Column("NULL");
  if (typeof v === "string") return new Column("'" + v.replaceAll("\\", "\\\\").replaceAll("'", "\\'") + "'");
  if (typeof v === "number" && !Number.isFinite(v)) throw new RangeError("a literal must be a finite number");
  return new Column(String(v));
}

const fn = (name: string) => (c: Column | string) => new Column(`${name}(${typeof c === "string" ? col(c).sql : c.sql})`);
export const sum = fn("SUM"), avg = fn("AVG"), min = fn("MIN"), max = fn("MAX");
export const count = (c: Column | string = "*") => new Column(`COUNT(${typeof c === "string" ? col(c).sql : c.sql})`);

type Cols = (Column | string)[];
const sql = (c: Column | string): string => (typeof c === "string" ? col(c).sql : c.sql);

let aliasSeq = 0;

/** A lazy query. Builds SQL text, so it runs on every backend; nothing runs until an action. */
export class DataFrame {
  constructor(private readonly backend: Backend, private readonly text: string) {}

  /** The SQL this DataFrame stands for. */
  toSQL(): string { return this.text; }

  private wrap(select: string, tail = ""): DataFrame {
    return new DataFrame(this.backend, `SELECT ${select} FROM (${this.text}) AS _t${++aliasSeq}${tail}`);
  }

  select(...cols: Cols): DataFrame { return this.wrap(cols.length ? cols.map(sql).join(", ") : "*"); }
  filter(cond: Column | string): DataFrame { return this.wrap("*", ` WHERE ${typeof cond === "string" ? cond : cond.sql}`); }
  where(cond: Column | string): DataFrame { return this.filter(cond); }
  orderBy(...cols: Cols): DataFrame { return this.wrap("*", ` ORDER BY ${cols.map(sql).join(", ")}`); }
  limit(n: number): DataFrame {
    if (!Number.isInteger(n) || n < 0) throw new RangeError("limit takes a non-negative integer");
    return this.wrap("*", ` LIMIT ${n}`);
  }
  distinct(): DataFrame { return this.wrap("DISTINCT *"); }
  withColumn(name: string, c: Column): DataFrame { return this.wrap(`*, ${c.sql} AS ${q(name)}`); }
  groupBy(...cols: Cols): GroupedData { return new GroupedData(this, cols.map(sql)); }

  /** Join on a condition, or on shared column names (USING). */
  join(other: DataFrame, on: Column | string[], how: "inner" | "left" | "right" | "full" | "cross" = "inner"): DataFrame {
    const kind = how === "full" ? "FULL OUTER" : how.toUpperCase();
    const a = `_l${++aliasSeq}`, b = `_r${aliasSeq}`;
    const cond = how === "cross" ? "" : Array.isArray(on) ? ` USING (${on.map(q).join(", ")})` : ` ON ${on.sql}`;
    return new DataFrame(this.backend, `SELECT * FROM (${this.text}) AS ${a} ${kind} JOIN (${other.text}) AS ${b}${cond}`);
  }

  /** @internal */ grouped(keys: string[], aggs: Column[]): DataFrame {
    const list = [...keys, ...aggs.map((a) => a.sql)].join(", ");
    return this.wrap(list, keys.length ? ` GROUP BY ${keys.join(", ")}` : "");
  }

  // actions
  private run(): Promise<ResultStream> { return this.backend.execute(this.text); }
  async toArrow(): Promise<Table> { return collectTable(await this.run()); }
  async collect(options?: ObjectOptions): Promise<Row[]> { return toObjects(await this.toArrow(), options); }
  async schema(): Promise<Schema> { return (await this.limit(0).run()).schema; }
  async *stream(): AsyncIterable<RecordBatch> { yield* (await this.run()).batches(); }
  async count(): Promise<number> {
    const [r] = await this.wrap("COUNT(*) AS n").collect({ bigint: "number" });
    return Number(r?.["n"] ?? 0);
  }
  async explain(): Promise<string> {
    const rows = await new DataFrame(this.backend, `EXPLAIN ${this.text}`).collect();
    return rows.map((r) => Object.values(r).join(" ")).join("\n");
  }
  async show(n = 20): Promise<string> {
    const rows = await this.limit(n).collect({ bigint: "string" });
    const names = rows[0] ? Object.keys(rows[0]) : (await this.schema()).fields.map((f) => f.name);
    const cells = rows.map((r) => names.map((k) => String(r[k] ?? "NULL")));
    const w = names.map((h, i) => Math.max(h.length, ...cells.map((c) => c[i]!.length)));
    const line = "+" + w.map((x) => "-".repeat(x + 2)).join("+") + "+";
    const fmt = (c: string[]) => "| " + c.map((x, i) => x.padEnd(w[i]!)).join(" | ") + " |";
    return [line, fmt(names), line, ...cells.map(fmt), line].join("\n");
  }
}

export class GroupedData {
  constructor(private readonly df: DataFrame, private readonly keys: string[]) {}
  agg(...aggs: Column[]): DataFrame { return this.df.grouped(this.keys, aggs); }
  count(): DataFrame { return this.agg(count().alias("count")); }
}
