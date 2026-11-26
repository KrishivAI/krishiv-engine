// SPDX-License-Identifier: Apache-2.0
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import * as grpc from "@grpc/grpc-js";
import * as protobuf from "protobufjs";
import { tableFromArrays, tableToIPC, RecordBatchReader } from "apache-arrow";
import { FLIGHT_PROTO } from "../src/remote/flight-proto.js";
import { Session, KrishivError, col, count, sum, parseRemoteUrl } from "../src/index.js";

// A Flight SQL server in the test process: GetFlightInfo -> ticket -> DoGet, DoPut for updates, bearer + cookie checks.
const SQL = "type.googleapis.com/arrow.flight.protocol.sql.";
const seen: { sql: string[]; auth: (string | undefined)[]; cookies: (string | undefined)[] } = { sql: [], auth: [], cookies: [] };
let server: grpc.Server;
let port = 0;

function messages(ipc: Uint8Array): { header: Uint8Array; body: Uint8Array }[] {
  const out: { header: Uint8Array; body: Uint8Array }[] = [];
  const dv = new DataView(ipc.buffer, ipc.byteOffset, ipc.byteLength);
  let at = 0;
  while (at + 8 <= ipc.length) {
    const len = dv.getInt32(at + 4, true);
    if (len === 0) break;
    const header = ipc.subarray(at + 8, at + 8 + len);
    at += 8 + len;
    // body length lives in the flatbuffer; the test tables are small, so find it by reading to the next marker
    let end = ipc.length;
    for (let i = at; i + 8 <= ipc.length; i += 8) if (dv.getUint32(i, true) === 0xffffffff) { end = i; break; }
    out.push({ header, body: ipc.subarray(at, end) });
    at = end;
  }
  return out;
}

beforeAll(async () => {
  const root = protobuf.parse(FLIGHT_PROTO, new protobuf.Root(), { keepCase: true }).root;
  const T = (n: string) => root.lookupType(n);
  const ser = (n: string) => (m: object) => Buffer.from(T(n).encode(T(n).create(m)).finish());
  const de = (n: string) => (b: Buffer) => T(n).toObject(T(n).decode(b), { bytes: Uint8Array, defaults: true });
  const table = tableFromArrays({ id: BigInt64Array.from([1n, 2n, 3n]), name: ["a", "b", null] });
  const ipc = tableToIPC(table, "stream");
  server = new grpc.Server();
  const check = (md: grpc.Metadata) => {
    seen.auth.push(md.get("authorization")[0] as string | undefined);
    seen.cookies.push(md.get("cookie")[0] as string | undefined);
    return md.get("authorization")[0] === "Bearer good";
  };
  const reply = (md: grpc.Metadata): grpc.Metadata => { const m = new grpc.Metadata(); m.set("set-cookie", "arrow_flight_session_id=s1; Path=/"); return m; };
  const handlers = {
    GetFlightInfo: (call: grpc.ServerUnaryCall<{ cmd: Uint8Array }, unknown>, cb: grpc.sendUnaryData<unknown>) => {
      if (!check(call.metadata)) return cb({ code: grpc.status.UNAUTHENTICATED, details: "bad token" });
      const any = T("Any").toObject(T("Any").decode(call.request.cmd), { bytes: Uint8Array, defaults: true }) as { value: Uint8Array };
      const sql = (T("CommandStatementQuery").toObject(T("CommandStatementQuery").decode(any.value)) as { query: string }).query;
      seen.sql.push(sql);
      if (sql.includes("boom")) {
        const md = new grpc.Metadata();
        md.set("x-krishiv-error-class", "DIVIDE_BY_ZERO");
        md.set("x-krishiv-sqlstate", "22012");
        return cb({ code: grpc.status.INVALID_ARGUMENT, details: "Division by zero", metadata: md });
      }
      call.sendMetadata(reply(call.metadata));
      cb(null, { schema: messages(ipc)[0]!.header, endpoint: [{ ticket: { ticket: Buffer.from("t") } }] });
    },
    DoGet: (call: grpc.ServerWritableStream<unknown, unknown>) => {
      for (const m of messages(ipc)) call.write({ data_header: m.header, data_body: m.body });
      call.end();
    },
    DoPut: (call: grpc.ServerDuplexStream<unknown, unknown>) => {
      call.on("data", () => call.write({ app_metadata: T("DoPutUpdateResult").encode({ record_count: 7 }).finish() }));
      call.on("end", () => call.end());
    },
  };
  const def = (name: string, req: string, res: string, rs: boolean, cs: boolean) => ({
    path: `/arrow.flight.protocol.FlightService/${name}`, requestStream: cs, responseStream: rs,
    requestSerialize: ser(req), requestDeserialize: de(req), responseSerialize: ser(res), responseDeserialize: de(res),
  });
  server.addService(
    {
      GetFlightInfo: def("GetFlightInfo", "FlightDescriptor", "FlightInfo", false, false),
      DoGet: def("DoGet", "Ticket", "FlightData", true, false),
      DoPut: def("DoPut", "FlightData", "PutResult", true, true),
    } as grpc.ServiceDefinition,
    handlers as unknown as grpc.UntypedServiceImplementation,
  );
  port = await new Promise<number>((res, rej) => server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (e, p) => (e ? rej(e) : res(p))));
});
afterAll(() => server.forceShutdown());

describe("remote session", () => {
  it("parses URLs", () => {
    expect(parseRemoteUrl("krishiv://h:1234")).toEqual({ host: "h", port: 1234, tls: false });
    expect(parseRemoteUrl("krishivs://h")).toEqual({ host: "h", port: 15002, tls: true });
    expect(() => parseRemoteUrl("http://h:1")).toThrow(KrishivError);
  });

  it("runs a query and returns Arrow, rows and counts", async () => {
    const s = await Session.builder().remote(`krishiv://127.0.0.1:${port}`).token("good").getOrCreate();
    const df = s.sql("SELECT id, name FROM t");
    const table = await df.toArrow();
    expect(table.numRows).toBe(3);
    expect(await df.collect()).toEqual([{ id: 1n, name: "a" }, { id: 2n, name: "b" }, { id: 3n, name: null }]);
    expect(await df.collect({ bigint: "number" })).toHaveLength(3);
    expect(await df.collect({ bigint: "string" })).toEqual([
      { id: "1", name: "a" }, { id: "2", name: "b" }, { id: "3", name: null },
    ]);
    expect((await df.schema()).fields.map((f) => f.name)).toEqual(["id", "name"]);
    await s.close();
  });

  it("sends the bearer token and keeps the session cookie", async () => {
    seen.auth.length = 0; seen.cookies.length = 0;
    const s = await Session.builder().remote(`krishiv://127.0.0.1:${port}`).token("good").getOrCreate();
    await s.sql("SELECT 1").collect();
    await s.sql("SELECT 2").collect();
    expect(seen.auth.every((a) => a === "Bearer good")).toBe(true);
    expect(seen.cookies[0]).toBeUndefined();
    expect(seen.cookies.at(-1)).toBe("arrow_flight_session_id=s1");
    await s.close();
  });

  it("maps a refused token and an engine error to KrishivError", async () => {
    const bad = await Session.builder().remote(`krishiv://127.0.0.1:${port}`).token("nope").getOrCreate();
    await expect(bad.sql("SELECT 1").collect()).rejects.toThrow(/bad token/);
    await bad.close();
    const s = await Session.builder().remote(`krishiv://127.0.0.1:${port}`).token("good").getOrCreate();
    const err = await s.sql("SELECT 1/0 boom").collect().catch((e) => e);
    expect(err).toBeInstanceOf(KrishivError);
    expect(err.errorClass).toBe("DIVIDE_BY_ZERO");
    expect(err.sqlState).toBe("22012");
    await s.close();
  });

  it("builds SQL from DataFrame calls", async () => {
    seen.sql.length = 0;
    const s = await Session.builder().remote(`krishiv://127.0.0.1:${port}`).token("good").getOrCreate();
    await s.sql("SELECT * FROM t").filter(col("id").gt(1)).groupBy("name").agg(count().alias("n"), sum("id")).orderBy(col("n").desc()).limit(5).collect();
    expect(seen.sql[0]).toMatch(/GROUP BY `name`/);
    expect(seen.sql[0]).toMatch(/WHERE \(id > 1\)/);
    expect(seen.sql[0]).toMatch(/LIMIT 5/);
    await s.close();
  });

  it("applies update counts via DoPut", async () => {
    const s = await Session.builder().remote(`krishiv://127.0.0.1:${port}`).token("good").getOrCreate();
    await expect(s.conf.set("spark.sql.session.timeZone", "UTC")).resolves.toBeUndefined();
    await s.close();
  });

  it("refuses the embedded engine with a clear error when no binary is installed", async () => {
    await expect(Session.builder().getOrCreate()).rejects.toThrow(/embedded engine is not available/);
  });
});
void RecordBatchReader;
