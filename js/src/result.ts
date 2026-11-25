// SPDX-License-Identifier: Apache-2.0
import { DataType, type RecordBatch, Schema, Table } from "apache-arrow";
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

function convert(value: unknown, type: DataType, o: Required<ObjectOptions>): unknown {
  if (value === null || value === undefined) return null;
  if (typeof value === "bigint") {
    if (o.bigint === "string") return value.toString();
    if (o.bigint === "number") {
      if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) {
        throw new RangeError(`${value} does not fit a JavaScript number; use bigint: "bigint" or "string"`);
      }
      return Number(value);
    }
    return value;
  }
  if (DataType.isDecimal(type)) return o.decimal === "number" ? Number(String(value)) : String(value);
  return value;
}

/** Rows as plain objects keyed by column name. */
export function toObjects(table: Table, options: ObjectOptions = {}): Row[] {
  const o: Required<ObjectOptions> = { bigint: options.bigint ?? "bigint", decimal: options.decimal ?? "string" };
  const fields = table.schema.fields;
  const rows: Row[] = [];
  for (let i = 0; i < table.numRows; i++) {
    const row: Row = {};
    for (let c = 0; c < fields.length; c++) {
      const f = fields[c]!;
      row[f.name] = convert(table.getChildAt(c)?.get(i), f.type, o);
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
  if (DataType.isUtf8(t) || DataType.isLargeUtf8(t)) return "STRING";
  if (DataType.isBinary(t) || DataType.isLargeBinary(t)) return "BINARY";
  if (DataType.isDate(t)) return "DATE";
  if (DataType.isTimestamp(t)) return t.timezone ? "TIMESTAMP" : "TIMESTAMP_NTZ";
  if (DataType.isList(t)) return `ARRAY<${ddlType(t.children[0]!.type)}>`;
  if (DataType.isStruct(t)) return `STRUCT<${t.children.map((c) => `${quote(c.name)}: ${ddlType(c.type)}`).join(", ")}>`;
  return String(t).toUpperCase();
}
