// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";
import { tableFromArrays, tableToIPC } from "apache-arrow";
import {
  checkVersion,
  embeddedBackend,
  fromNative,
  loadAddon,
  platformPackage,
  NATIVE_ENGINE_ENV,
  type NativeAddon,
  type NativeQuery,
} from "../src/embedded.js";
import { collectTable, toObjects } from "../src/result.js";
import { KrishivError, Session, col, count } from "../src/index.js";

const nativeError = (
  errorClass: string,
  sqlState: string,
  message: string,
  messageParameters: Record<string, string> = {},
) => new Error(JSON.stringify({ errorClass, sqlState, messageParameters, message }));

/**
 * A fake addon: each query answers `ipc` in chunks of 7 bytes with its 8-byte end-of-stream marker as a chunk of its
 * own (as the addon sends it), or fails with `fail`.
 */
function fake(
  opts: { ipc?: Uint8Array; fail?: Error; hang?: boolean; closeFails?: Error; cancelFails?: Error } = {},
) {
  const log = { sql: [] as string[], cancels: 0, closes: 0, conf: new Map<string, string>() };
  const addon: NativeAddon = {
    version: () => "0.0.0-test",
    NativeSession: {
      open(conf) {
        for (const [k, v] of Object.entries(conf)) log.conf.set(k, v);
        return {
          execute(sql): NativeQuery {
            log.sql.push(sql);
            const bytes = opts.ipc ?? new Uint8Array();
            const body = Math.max(0, bytes.length - 8);
            const parts: Uint8Array[] = [];
            for (let i = 0; i < body; i += 7) parts.push(bytes.subarray(i, Math.min(i + 7, body)));
            if (bytes.length >= 8) parts.push(bytes.subarray(body));
            let at = 0;
            let cancelled = false;
            return {
              async next() {
                if (cancelled) throw nativeError("OPERATION_CANCELED", "HY008", "The operation has been cancelled.");
                if (opts.fail) throw opts.fail;
                if (opts.hang && at > 0) {
                  await new Promise((r) => setTimeout(r, 5));
                  if (cancelled) throw nativeError("OPERATION_CANCELED", "HY008", "The operation has been cancelled.");
                }
                if (at >= parts.length) return null;
                return parts[at++]!;
              },
              cancel() {
                cancelled = true;
                log.cancels++;
                if (opts.cancelFails) throw opts.cancelFails;
              },
            };
          },
          setConf(k, v) {
            if (k === "bad") throw nativeError("INVALID_CONF_VALUE", "22022", "bad value");
            log.conf.set(k, v);
          },
          getConf: (k) => log.conf.get(k) ?? null,
          close() {
            log.closes++;
            if (opts.closeFails) throw opts.closeFails;
          },
        };
      },
    },
  };
  return { addon, log };
}

const ipc = tableToIPC(tableFromArrays({ id: BigInt64Array.from([1n, 2n, 3n]), name: ["a", "b", null] }), "stream");

describe("embedded backend (fake addon)", () => {
  it("streams IPC chunks into Arrow batches", async () => {
    const { addon, log } = fake({ ipc });
    const b = await embeddedBackend({ "spark.sql.session.timeZone": "UTC" }, addon);
    const t = await collectTable(await b.execute("SELECT 1"));
    expect(t.numRows).toBe(3);
    expect(toObjects(t)).toEqual([
      { id: 1n, name: "a" },
      { id: 2n, name: "b" },
      { id: 3n, name: null },
    ]);
    expect(log.sql).toEqual(["SELECT 1"]);
    expect(log.conf.get("spark.sql.session.timeZone")).toBe("UTC");
    // A stream read to the end is not cancelled.
    expect(log.cancels).toBe(0);
  });

  it("a failure is a KrishivError with the Spark error class", async () => {
    const { addon } = fake({
      fail: nativeError("TABLE_OR_VIEW_NOT_FOUND", "42P01", "no table t", { relationName: "`t`" }),
    });
    const b = await embeddedBackend({}, addon);
    const e = await b.execute("SELECT * FROM t").catch((x: unknown) => x);
    expect(e).toBeInstanceOf(KrishivError);
    expect((e as KrishivError).errorClass).toBe("TABLE_OR_VIEW_NOT_FOUND");
    expect((e as KrishivError).sqlState).toBe("42P01");
    expect((e as KrishivError).message).toBe("no table t");
    expect((e as KrishivError).messageParameters).toEqual({ relationName: "`t`" });
  });

  it("an aborted signal cancels the query", async () => {
    const { addon, log } = fake({ ipc, hang: true });
    const b = await embeddedBackend({}, addon);
    const ac = new AbortController();
    const s = await b.execute("SELECT 1", { signal: ac.signal });
    const why = new Error("user left");
    ac.abort(why);
    const e = await collectTable(s).catch((x: unknown) => x);
    expect(e).toBeInstanceOf(KrishivError);
    expect((e as KrishivError).errorClass).toBe("OPERATION_CANCELED");
    expect((e as KrishivError).sqlState).toBe("HY008");
    expect((e as KrishivError).cause).toBe(why);
    expect(log.cancels).toBeGreaterThanOrEqual(1);
  });

  it("a signal aborted before execute runs nothing", async () => {
    const { addon, log } = fake({ ipc });
    const b = await embeddedBackend({}, addon);
    const ac = new AbortController();
    ac.abort();
    const e = await b.execute("SELECT 1", { signal: ac.signal }).catch((x: unknown) => x);
    expect(e).toBeInstanceOf(KrishivError);
    expect((e as KrishivError).errorClass).toBe("OPERATION_CANCELED");
    expect((e as KrishivError).sqlState).toBe("HY008");
    expect(log.sql).toEqual([]);
  });

  it("stopping early cancels the query", async () => {
    // Two batches: the break after the first comes before the end-of-stream marker.
    const half = (from: number) => tableFromArrays({ id: Int32Array.from({ length: 5 }, (_, i) => from + i) });
    const big = tableToIPC(half(0).concat(half(5)), "stream");
    const { addon, log } = fake({ ipc: big });
    const b = await embeddedBackend({}, addon);
    const s = await b.execute("SELECT 1");
    for await (const _ of s.batches()) break;
    expect(log.cancels).toBe(1);
  });

  it("a failing cancel is swallowed when stopping early or aborting", async () => {
    const half = (from: number) => tableFromArrays({ id: Int32Array.from({ length: 5 }, (_, i) => from + i) });
    const big = tableToIPC(half(0).concat(half(5)), "stream");
    const { addon, log } = fake({ ipc: big, hang: true, cancelFails: new Error("already gone") });
    const b = await embeddedBackend({}, addon);
    const s = await b.execute("SELECT 1");
    for await (const _ of s.batches()) break;
    expect(log.cancels).toBe(1);
    const ac = new AbortController();
    const t = await b.execute("SELECT 2", { signal: ac.signal });
    ac.abort();
    const e = await collectTable(t).catch((x: unknown) => x);
    expect((e as KrishivError).errorClass).toBe("OPERATION_CANCELED");
    expect(log.cancels).toBeGreaterThanOrEqual(2);
  });

  it("batches can be read once", async () => {
    const { addon } = fake({ ipc });
    const s = await (await embeddedBackend({}, addon)).execute("SELECT 1");
    await collectTable(s);
    await expect(collectTable(s)).rejects.toBeInstanceOf(KrishivError);
  });

  it("conf and close", async () => {
    const { addon, log } = fake();
    const b = await embeddedBackend({}, addon);
    await b.setConf("k", "v");
    expect(await b.getConf("k")).toBe("v");
    expect(await b.getConf("missing")).toBeUndefined();
    const e = await b.setConf("bad", "x").catch((x: unknown) => x);
    expect((e as KrishivError).errorClass).toBe("INVALID_CONF_VALUE");
    await b.close();
    await b.close();
    expect(log.closes).toBe(1);
  });

  it("a failing close is a KrishivError", async () => {
    const { addon } = fake({ closeFails: nativeError("INTERNAL_ERROR", "XX000", "krishiv-node: panic: boom") });
    const b = await embeddedBackend({}, addon);
    const e = await b.close().catch((x: unknown) => x);
    expect(e).toBeInstanceOf(KrishivError);
    expect((e as KrishivError).errorClass).toBe("INTERNAL_ERROR");
  });
});

describe("loading the addon", () => {
  it("names the platform package", () => {
    expect(platformPackage("linux", "x64", true)).toBe("@krishiv/engine-linux-x64-gnu");
    expect(platformPackage("linux", "arm64", false)).toBe("@krishiv/engine-linux-arm64-musl");
    expect(platformPackage("darwin", "arm64")).toBe("@krishiv/engine-darwin-arm64");
    expect(platformPackage("win32", "x64")).toBe("@krishiv/engine-win32-x64-msvc");
  });

  it("refuses a binary of another version", () => {
    expect(() => checkVersion("@krishiv/engine-linux-x64-gnu", "0.0.0-test", "0.2.1")).toThrow(
      /is version 0\.0\.0-test but krishiv is 0\.2\.1/,
    );
    try {
      checkVersion("@krishiv/engine-linux-x64-gnu", "0.0.0-test", "0.2.1");
    } catch (e) {
      expect((e as KrishivError).errorClass).toBe("KRISHIV_NATIVE_ENGINE_UNAVAILABLE");
    }
    expect(() => checkVersion("p", "0.2.1", "0.2.1")).not.toThrow();
    expect(() => checkVersion("p", "0.2.1", undefined)).not.toThrow();
  });

  it("a bad path is KRISHIV_NATIVE_ENGINE_UNAVAILABLE", () => {
    const e = (() => {
      try {
        loadAddon({ [NATIVE_ENGINE_ENV]: "/no/such/krishiv.node" });
      } catch (x) {
        return x;
      }
    })();
    expect((e as KrishivError).errorClass).toBe("KRISHIV_NATIVE_ENGINE_UNAVAILABLE");
  });

  it("leaves errors that are not native JSON alone", () => {
    const plain = new Error("plain");
    expect(fromNative(plain)).toBe(plain);
    expect(fromNative("text")).toBe("text");
  });
});

// The real addon: `cargo build -p krishiv-node`, then point KRISHIV_NATIVE_ENGINE at the library copied to `*.node`.
const real = process.env[NATIVE_ENGINE_ENV];
describe.skipIf(real === undefined || real === "")("embedded engine (real addon)", () => {
  it("runs SQL and DataFrames", async () => {
    const s = await Session.builder().config("spark.sql.session.timeZone", "UTC").getOrCreate();
    try {
      expect(await s.conf.get("spark.sql.session.timeZone")).toBe("UTC");
      const rows = toObjects(await s.sql("SELECT id, CAST(id AS STRING) AS s FROM range(5)").toArrow());
      expect(rows).toHaveLength(5);
      expect(rows[4]).toEqual({ id: 4n, s: "4" });
      const g = toObjects(
        await s
          .sql("SELECT id % 2 AS k FROM range(10)")
          .groupBy("k")
          .agg(count().alias("n"))
          .orderBy(col("k"))
          .toArrow(),
      );
      expect(g).toEqual([
        { k: 0n, n: 5n },
        { k: 1n, n: 5n },
      ]);
    } finally {
      await s.close();
    }
  });

  it("a failure carries the Spark error class", async () => {
    const s = await Session.builder().getOrCreate();
    try {
      const e = await s.sql("SELECT * FROM no_such_table").toArrow().catch((x: unknown) => x);
      expect((e as KrishivError).errorClass).toBe("TABLE_OR_VIEW_NOT_FOUND");
      expect((e as KrishivError).sqlState).toBe("42P01");
      expect((e as KrishivError).messageParameters).toHaveProperty("relationName");
    } finally {
      await s.close();
    }
  });

  it("a closed session refuses with INVALID_HANDLE.SESSION_CLOSED", async () => {
    const s = await Session.builder().getOrCreate();
    await s.close();
    const e = await s.sql("SELECT 1").toArrow().catch((x: unknown) => x);
    expect((e as KrishivError).errorClass).toBe("INVALID_HANDLE.SESSION_CLOSED");
  });
});
