// SPDX-License-Identifier: Apache-2.0
import { DataType, type RecordBatch, Schema, Table, type Vector } from "apache-arrow";
import type { ResultStream } from "./backend.js";

export type Row = Record<string, unknown>;

export interface ObjectOptions {
  /** 64-bit integers: `"bigint"` (default, lossless), `"number"` (throws past 2^53) or `"string"`. */
  bigint?: "bigint" | "number" | "string";
  /** Decimals: `"string"` (default, exact) or `"number"`. */
  decimal?: "string" | "number";
}

/** Collect a stream into one Arrow table. */
export async function collectTable(stream: ResultStream): Promise<Table> {
  const batches: RecordBatch[] = [];
  for await (const b of stream.batches()) batches.push(b);
  return batches.length === 0 ? new Table(stream.schema) : new Table(stream.schema, batches);
}

/** The stored value of row `i`: the chunk's raw values array and the slot in it (dates, decimals, timestamps). */
function slot(vec: Vector, i: number): { values: ArrayLike<unknown>; at: number } {
  let row = i;
  for (const d of vec.data) {
    if (row < d.length) return { values: d.values as ArrayLike<unknown>, at: d.offset + row };
    row -= d.length;
  }
  throw new RangeError(`row ${i} past the end`);
}

/** A 64-bit slot as a bigint: a BigInt64Array element, or two 32-bit words (low word first). */
function int64At(values: ArrayLike<unknown>, at: number): bigint {
  if (values instanceof BigInt64Array) return values[at]!;
  const words = values as Int32Array;
  return (BigInt(words[2 * at + 1]!) << 32n) | BigInt(words[2 * at]! >>> 0);
}

const pad = (n: bigint | number, w: number) => String(n).padStart(w, "0");
/** `a / b` rounded toward negative infinity (`b > 0`). */
const floorDiv = (a: bigint, b: bigint) => (a < 0n && a % b !== 0n ? a / b - 1n : a / b);

/** A UTC instant `us` microseconds from the epoch as `YYYY-MM-DDTHH:MM:SS.ffffffZ`. */
function isoMicros(us: bigint): string {
  const sec = floorDiv(us, 1_000_000n);
  const frac = us - sec * 1_000_000n;
  return `${new Date(Number(sec) * 1000).toISOString().slice(0, 19)}.${pad(frac, 6)}Z`;
}

function integer(v: bigint | number, o: Required<ObjectOptions>): bigint | number | string {
  if (typeof v === "number") return v;
  if (o.bigint === "string") return String(v);
  if (o.bigint === "number") {
    if (v > BigInt(Number.MAX_SAFE_INTEGER) || v < BigInt(Number.MIN_SAFE_INTEGER)) {
      throw new RangeError(`${v} does not fit a JavaScript number; use bigint: "bigint" or "string"`);
    }
    return Number(v);
  }
  return v;
}

/**
 * Row `i` of `vec` as a plain JavaScript value. Dates are `YYYY-MM-DD` strings and timestamps are
 * `YYYY-MM-DDTHH:MM:SS.ffffffZ` strings (microsecond-exact; a timestamp without a zone is read as UTC),
 * decimals keep their scale (`-90.50`), binary is a `Uint8Array`, lists are arrays and structs objects,
 * recursively, so the result holds no Arrow vectors and (with `bigint: "string"`) no bigints.
 */
function convert(vec: Vector, i: number, o: Required<ObjectOptions>): unknown {
  const t = vec.type;
  if (!vec.isValid(i)) return null;
  if (DataType.isInt(t)) return integer(vec.get(i) as bigint | number, o);
  if (DataType.isDate(t)) {
    const { values, at } = slot(vec, i);
    const ms = t.unit === 0 ? Number(values[at]) * 86_400_000 : Number(int64At(values, at));
    return new Date(ms).toISOString().slice(0, 10);
  }
  if (DataType.isTimestamp(t)) {
    const { values, at } = slot(vec, i);
    const scale = [1_000_000n, 1000n, 1n, 0n][t.unit]!;
    const raw = int64At(values, at);
    return isoMicros(scale === 0n ? floorDiv(raw, 1000n) : raw * scale);
  }
  if (DataType.isDecimal(t)) {
    const { values, at } = slot(vec, i);
    const words = t.bitWidth / 32;
    let unscaled = 0n;
    for (let w = words - 1; w >= 0; w--) unscaled = (unscaled << 32n) | BigInt(values[at * words + w] as number);
    if (unscaled >> BigInt(t.bitWidth - 1)) unscaled -= 1n << BigInt(t.bitWidth);
    const neg = unscaled < 0n;
    const digits = pad(neg ? -unscaled : unscaled, t.scale + 1);
    const s = t.scale === 0 ? digits : `${digits.slice(0, -t.scale)}.${digits.slice(-t.scale)}`;
    const text = neg ? `-${s}` : s;
    return o.decimal === "number" ? Number(text) : text;
  }
  if (DataType.isList(t) || DataType.isFixedSizeList(t)) {
    const items = vec.get(i) as Vector;
    return Array.from({ length: items.length }, (_, j) => convert(items, j, o));
  }
  if (DataType.isStruct(t)) {
    const out: Record<string, unknown> = {};
    t.children.forEach((c, k) => {
      out[c.name] = convert(vec.getChildAt(k)!, i, o);
    });
    return out;
  }
  return vec.get(i);
}

/** Rows as plain objects keyed by column name. */
export function toObjects(table: Table, options: ObjectOptions = {}): Row[] {
  const o: Required<ObjectOptions> = { bigint: options.bigint ?? "bigint", decimal: options.decimal ?? "string" };
  const fields = table.schema.fields;
  const rows: Row[] = [];
  const cols = fields.map((_, c) => table.getChildAt(c)!);
  for (let i = 0; i < table.numRows; i++) {
    const row: Row = {};
    for (let c = 0; c < fields.length; c++) {
      const f = fields[c]!;
      row[f.name] = convert(cols[c]!, i, o);
    }
    rows.push(row);
  }
  return rows;
}

/** Rows as arrays in column order. */
export function toRows(table: Table, options: ObjectOptions = {}): unknown[][] {
  const names = table.schema.fields.map((f) => f.name);
  return toObjects(table, options).map((r) => names.map((n) => r[n]));
}

/** Columns as arrays keyed by name. */
export function toColumns(table: Table, options: ObjectOptions = {}): Record<string, unknown[]> {
  const out: Record<string, unknown[]> = {};
  for (const f of table.schema.fields) out[f.name] = [];
  for (const row of toObjects(table, options)) for (const k of Object.keys(out)) out[k]!.push(row[k]);
  return out;
}

/** JSON text of the rows; 64-bit integers are strings, so nothing is lost or throws. */
export function toJSON(table: Table, options: ObjectOptions = {}): string {
  return JSON.stringify(toObjects(table, { ...options, bigint: options.bigint === "number" ? "number" : "string" }));
}

/** The schema as a Spark DDL string, e.g. `id BIGINT, name STRING`. */
export function schemaToDDL(schema: Schema): string {
  return schema.fields.map((f) => `${quote(f.name)} ${ddlType(f.type)}${f.nullable ? "" : " NOT NULL"}`).join(", ");
}

function quote(name: string): string {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) ? name : "`" + name.replaceAll("`", "``") + "`";
}

function ddlType(t: DataType): string {
  if (DataType.isBool(t)) return "BOOLEAN";
  if (DataType.isInt(t)) return ({ 8: "TINYINT", 16: "SMALLINT", 32: "INT", 64: "BIGINT" } as Record<number, string>)[t.bitWidth] ?? "BIGINT";
  if (DataType.isFloat(t)) return t.precision === 2 ? "DOUBLE" : "FLOAT";
  if (DataType.isDecimal(t)) return `DECIMAL(${t.precision},${t.scale})`;
  if (DataType.isUtf8(t) || DataType.isLargeUtf8(t) || DataType.isUtf8View?.(t)) return "STRING";
  if (DataType.isBinary(t) || DataType.isLargeBinary(t) || DataType.isBinaryView?.(t)) return "BINARY";
  if (DataType.isDate(t)) return "DATE";
  if (DataType.isTimestamp(t)) return t.timezone ? "TIMESTAMP" : "TIMESTAMP_NTZ";
  if (DataType.isList(t)) return `ARRAY<${ddlType(t.children[0]!.type)}>`;
  if (DataType.isStruct(t)) return `STRUCT<${t.children.map((c) => `${quote(c.name)}: ${ddlType(c.type)}`).join(", ")}>`;
  return String(t).toUpperCase();
}
