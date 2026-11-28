// SPDX-License-Identifier: Apache-2.0
// The embedded engine returns the rows of tests/fixtures/client-parity/cases.json, as the Python package does
// (python/tests/test_client_parity.py); the canonical form is in that folder's README.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { DataType, type Table, type Vector } from "apache-arrow";
import { NATIVE_ENGINE_ENV } from "../src/embedded.js";
import { Session } from "../src/index.js";

interface Case {
  name: string;
  setup: string[];
  sql: string;
  types: string[];
  expected: Record<string, unknown>[];
}

const root = fileURLToPath(new URL("../..", import.meta.url)).replace(/\/$/, "");
const file = JSON.parse(readFileSync(`${root}/tests/fixtures/client-parity/cases.json`, "utf8")) as {
  conf: Record<string, string>;
  cases: Case[];
};

/** The canonical type string of an Arrow type. */
function typeString(t: DataType): string {
  if (DataType.isInt(t)) return ({ 8: "tinyint", 16: "smallint", 32: "int", 64: "bigint" } as Record<number, string>)[t.bitWidth]!;
  if (DataType.isFloat(t)) return t.precision === 2 ? "double" : "float";
  if (DataType.isDecimal(t)) return `decimal(${t.precision},${t.scale})`;
  if (DataType.isUtf8(t) || DataType.isLargeUtf8(t) || DataType.isUtf8View(t)) return "string";
  if (DataType.isBinary(t) || DataType.isLargeBinary(t) || DataType.isBinaryView(t)) return "binary";
  if (DataType.isBool(t)) return "boolean";
  if (DataType.isDate(t)) return "date";
  if (DataType.isTimestamp(t)) return t.timezone ? "timestamp" : "timestamp_ntz";
  if (DataType.isList(t)) return `array<${typeString(t.children[0]!.type)}>`;
  if (DataType.isStruct(t)) return `struct<${t.children.map((c) => `${c.name}:${typeString(c.type)}`).join(",")}>`;
  throw new Error(`no canonical form for ${String(t)}`);
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
  let sec = us / 1_000_000n;
  let frac = us % 1_000_000n;
  if (frac < 0n) {
    frac += 1_000_000n;
    sec -= 1n;
  }
  return `${new Date(Number(sec) * 1000).toISOString().slice(0, 19)}.${pad(frac, 6)}Z`;
}

/** Row `i` of `vec` in the canonical form. */
function canonical(vec: Vector, i: number): unknown {
  const t = vec.type;
  if (!vec.isValid(i)) return null;
  if (DataType.isInt(t)) return String(vec.get(i));
  if (DataType.isFloat(t) || DataType.isBool(t) || DataType.isUtf8(t) || DataType.isLargeUtf8(t) || DataType.isUtf8View(t)) return vec.get(i);
  if (DataType.isBinary(t) || DataType.isLargeBinary(t) || DataType.isBinaryView(t)) return Buffer.from(vec.get(i) as Uint8Array).toString("hex");
  if (DataType.isDate(t)) {
    const { values, at } = slot(vec, i);
    const ms = t.unit === 0 ? Number(values[at]) * 86_400_000 : Number(int64At(values, at));
    return new Date(ms).toISOString().slice(0, 10);
  }
  if (DataType.isTimestamp(t)) {
    const { values, at } = slot(vec, i);
    const scale = [1_000_000n, 1000n, 1n, 0n][t.unit]!;
    const raw = int64At(values, at);
    // Nanoseconds to microseconds by a floor, as the other side's reader does (`/` truncates toward zero).
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
    return neg ? `-${s}` : s;
  }
  if (DataType.isList(t)) {
    const items = vec.get(i) as Vector;
    return Array.from({ length: items.length }, (_, j) => canonical(items, j));
  }
  if (DataType.isStruct(t)) {
    const out: Record<string, unknown> = {};
    t.children.forEach((c, k) => {
      out[c.name] = canonical(vec.getChildAt(k)!, i);
    });
    return out;
  }
  throw new Error(`no canonical form for ${String(t)}`);
}

function rows(table: Table): Record<string, unknown>[] {
  const fields = table.schema.fields;
  return Array.from({ length: table.numRows }, (_, i) =>
    Object.fromEntries(fields.map((f, c) => [f.name, canonical(table.getChildAt(c)!, i)])),
  );
}

const real = process.env[NATIVE_ENGINE_ENV];
describe.skipIf(real === undefined || real === "")("client parity (real addon)", () => {
  for (const c of file.cases) {
    it(c.name, async () => {
      let b = Session.builder();
      for (const [k, v] of Object.entries(file.conf)) b = b.config(k, v);
      const s = await b.getOrCreate();
      try {
        for (const stmt of c.setup) await s.sql(stmt.replaceAll("{root}", root)).toArrow();
        const table = await s.sql(c.sql.replaceAll("{root}", root)).toArrow();
        expect(table.schema.fields.map((f) => typeString(f.type))).toEqual(c.types);
        expect(rows(table)).toEqual(c.expected);
      } finally {
        await s.close();
      }
    });
  }
});
