// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";
import { DateDay, Decimal, Field, Int64, List, Struct, Table, TimestampMicrosecond, Utf8View, Vector, makeData, makeVector, vectorFromArray } from "apache-arrow";
import { schemaToDDL, toJSON, toObjects } from "../src/result.js";

const col = <T extends Parameters<typeof makeData>[0]>(init: T): Vector => new Vector([makeData(init)]);

describe("result conversion", () => {
  const int64s = (xs: bigint[]) => col({ type: new Int64(), length: xs.length, data: BigInt64Array.from(xs) });

  it("formats date and timestamp, and recurses into lists and structs", () => {
    const items = int64s([1n, 2n]);
    const table = new Table({
      ts: col({ type: new TimestampMicrosecond(), length: 1, data: BigInt64Array.of(1704164645000123n) }),
      dt: col({ type: new DateDay(), length: 1, data: Int32Array.of(19724) }),
      arr: col({ type: new List(new Field("item", new Int64())), length: 1, valueOffsets: Int32Array.of(0, 2), child: items.data[0]! }),
      st: col({ type: new Struct([new Field("a", new Int64())]), length: 1, children: [int64s([1n]).data[0]!] }),
    });
    expect(toObjects(table)[0]).toEqual({ ts: "2024-01-02T03:04:05.000123Z", dt: "2024-01-02", arr: [1n, 2n], st: { a: 1n } });
    expect(JSON.parse(toJSON(table))).toEqual([{ ts: "2024-01-02T03:04:05.000123Z", dt: "2024-01-02", arr: ["1", "2"], st: { a: "1" } }]);
  });

  it("prints a negative decimal with its scale", () => {
    // -9050 at scale 2 as a 128-bit two's complement value
    const words = Uint32Array.of((-9050) >>> 0, 0xffffffff, 0xffffffff, 0xffffffff);
    const table = new Table({ d: col({ type: new Decimal(2, 15, 128), length: 1, data: words }) });
    expect(toObjects(table)[0]).toEqual({ d: "-90.50" });
    expect(toObjects(table, { decimal: "number" })[0]).toEqual({ d: -90.5 });
  });

  it("prints string views as STRING", () => {
    const table = new Table({ s: makeVector(vectorFromArray(["x"], new Utf8View()).data[0]!) });
    expect(schemaToDDL(table.schema)).toBe("s STRING");
  });
});
