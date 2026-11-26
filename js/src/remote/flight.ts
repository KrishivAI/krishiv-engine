// SPDX-License-Identifier: Apache-2.0
import { FLIGHT_PROTO } from "./flight-proto.js";
import { RecordBatchReader, type RecordBatch, type Schema } from "apache-arrow";
import type { Backend, ResultStream } from "../backend.js";
import { KrishivError } from "../errors.js";

const SQL_PREFIX = "type.googleapis.com/arrow.flight.protocol.sql.";

export interface RemoteOptions {
  host: string;
  port: number;
  tls: boolean;
  token?: string;
  /** Extra gRPC metadata sent with every call. */
  headers?: Record<string, string>;
  /** Milliseconds to wait for a call; 0 or unset: no deadline. */
  timeoutMs?: number;
  /** Largest gRPC message accepted, in bytes (default 256 MiB; grpc-js alone allows 4 MiB). */
  maxReceiveBytes?: number;
}

/** Parse `krishiv://host:port` (plaintext) or `krishivs://host:port` (TLS). */
export function parseRemoteUrl(url: string): Pick<RemoteOptions, "host" | "port" | "tls"> {
  const m = /^(krishivs?):\/\/([^/:?#]+|\[[^\]]+\])(?::(\d+))?\/?$/.exec(url);
  if (!m) throw new KrishivError(`not a Krishiv URL: ${url} (expected krishiv://host:port or krishivs://host:port)`);
  const tls = m[1] === "krishivs";
  const port = m[3] === undefined ? 15002 : Number(m[3]);
  if (port < 1 || port > 65535) throw new KrishivError(`port out of range in ${url}`);
  return { host: m[2]!.replace(/^\[|\]$/g, ""), port, tls };
}

// Hand-rolled protobuf subset: loaded lazily so embedded-only users never need gRPC.
interface Wire {
  grpc: typeof import("@grpc/grpc-js");
  root: import("protobufjs").Root;
}
let wire: Promise<Wire> | undefined;
function loadWire(): Promise<Wire> {
  wire ??= (async () => {
    try {
      // CommonJS packages: under ESM the exports can sit on `default`.
      const g = await import("@grpc/grpc-js");
      const p = await import("protobufjs");
      const grpc = ((g as unknown as { default?: typeof g }).default ?? g);
      const protobuf = ((p as unknown as { default?: typeof p }).default ?? p);
      const root = protobuf.parse(FLIGHT_PROTO, new protobuf.Root(), { keepCase: true }).root;
      return { grpc, root };
    } catch (e) {
      throw new KrishivError("remote mode needs the optional packages @grpc/grpc-js and protobufjs", { cause: e });
    }
  })();
  return wire;
}

function frame(chunks: { header: Uint8Array; body: Uint8Array }): Uint8Array[] {
  // One IPC message: continuation marker, padded metadata length, metadata, body (already 8-aligned by Arrow).
  const pad = (n: number) => (8 - (n % 8)) % 8;
  const metaLen = chunks.header.length + pad(chunks.header.length);
  const head = new Uint8Array(8 + metaLen);
  const dv = new DataView(head.buffer);
  dv.setUint32(0, 0xffffffff, true);
  dv.setInt32(4, metaLen, true);
  head.set(chunks.header, 8);
  return chunks.body.length === 0 ? [head] : [head, chunks.body];
}

/** Flight SQL over gRPC. One session (cookie) per `Backend`. */
export class RemoteBackend implements Backend {
  private cookie: string | undefined;
  private client: import("@grpc/grpc-js").Client | undefined;
  private w: Wire | undefined;

  constructor(private readonly opts: RemoteOptions) {}

  private async conn(): Promise<{ w: Wire; client: import("@grpc/grpc-js").Client }> {
    this.w ??= await loadWire();
    const { grpc } = this.w;
    this.client ??= new grpc.Client(
      this.opts.host.includes(":") ? `[${this.opts.host}]:${this.opts.port}` : `${this.opts.host}:${this.opts.port}`,
      this.opts.tls ? grpc.credentials.createSsl() : grpc.credentials.createInsecure(),
      { "grpc.max_receive_message_length": this.opts.maxReceiveBytes ?? 256 * 1024 * 1024 },
    );
    return { w: this.w, client: this.client };
  }

  private metadata(grpc: Wire["grpc"]): import("@grpc/grpc-js").Metadata {
    const md = new grpc.Metadata();
    if (this.opts.token) md.set("authorization", `Bearer ${this.opts.token}`);
    for (const [k, v] of Object.entries(this.opts.headers ?? {})) md.set(k.toLowerCase(), v);
    if (this.cookie) md.set("cookie", this.cookie);
    return md;
  }

  private options(): import("@grpc/grpc-js").CallOptions {
    return this.opts.timeoutMs ? { deadline: Date.now() + this.opts.timeoutMs } : {};
  }

  private remember(md: import("@grpc/grpc-js").Metadata): void {
    for (const v of md.get("set-cookie")) {
      const pair = String(v).split(";")[0]!.trim();
      if (pair.startsWith("arrow_flight_session_id=")) this.cookie = pair;
    }
  }

  private fail(err: unknown): KrishivError {
    const e = err as { code?: number; details?: string; message?: string; metadata?: import("@grpc/grpc-js").Metadata };
    const md = e.metadata;
    const first = (k: string) => (md ? (md.get(k)[0] as string | undefined) : undefined);
    const message = (e.details ?? e.message ?? "request failed").toString();
    return new KrishivError(message, {
      errorClass: first("x-krishiv-error-class"),
      sqlState: first("x-krishiv-sqlstate"),
      cause: err,
    });
  }

  private unary<Req, Res>(
    client: import("@grpc/grpc-js").Client,
    method: string,
    w: Wire,
    reqT: string,
    resT: string,
    req: object,
    signal?: AbortSignal,
  ): Promise<Res> {
    const Req_ = w.root.lookupType(reqT);
    const Res_ = w.root.lookupType(resT);
    return new Promise<Res>((resolve, reject) => {
      if (signal?.aborted) return reject(new KrishivError("query cancelled"));
      const onAbort = () => call.cancel();
      const call = client.makeUnaryRequest<Req, Res>(
        method,
        (m: Req) => Buffer.from(Req_.encode(Req_.create(m as object)).finish()),
        (b: Buffer) => Res_.toObject(Res_.decode(b), { longs: Number, bytes: Uint8Array, defaults: true }) as Res,
        req as Req,
        this.metadata(w.grpc),
        this.options(),
        (err, value) => {
          signal?.removeEventListener("abort", onAbort);
          return err ? reject(this.fail(err)) : resolve(value as Res);
        },
      );
      call.on("metadata", (md) => this.remember(md));
      signal?.addEventListener("abort", onAbort, { once: true });
    });
  }

  async execute(sql: string, opts: { signal?: AbortSignal } = {}): Promise<ResultStream> {
    const { w, client } = await this.conn();
    const cmd = this.command(w, "CommandStatementQuery", sql);
    const info = await this.unary<object, { schema: Uint8Array; endpoint: { ticket: { ticket: Uint8Array } }[] }>(
      client,
      "/arrow.flight.protocol.FlightService/GetFlightInfo",
      w,
      "FlightDescriptor",
      "FlightInfo",
      { type: 2, cmd },
      opts.signal,
    );
    const endpoints = info.endpoint;
    const schema = await schemaOf(info.schema);
    const self = this;
    return {
      schema,
      async *batches(): AsyncIterable<RecordBatch> {
        for (const ep of endpoints) {
          const reader = await RecordBatchReader.from(self.doGet(w, client, ep.ticket.ticket, opts.signal));
          await reader.open();
          for await (const b of reader) yield b;
        }
      },
    };
  }

  private command(w: Wire, type: "CommandStatementQuery" | "CommandStatementUpdate", query: string): Uint8Array {
    const T = w.root.lookupType(type);
    const Any = w.root.lookupType("Any");
    const value = T.encode(T.create({ query })).finish();
    return Any.encode(Any.create({ type_url: SQL_PREFIX + type, value })).finish();
  }

  private async *doGet(w: Wire, client: import("@grpc/grpc-js").Client, ticket: Uint8Array, signal?: AbortSignal): AsyncGenerator<Uint8Array> {
    if (signal?.aborted) throw new KrishivError("query cancelled");
    const Ticket = w.root.lookupType("Ticket");
    const FD = w.root.lookupType("FlightData");
    const call = client.makeServerStreamRequest<object, { data_header: Uint8Array; data_body: Uint8Array }>(
      "/arrow.flight.protocol.FlightService/DoGet",
      (m: object) => Buffer.from(Ticket.encode(Ticket.create(m)).finish()),
      (b: Buffer) => FD.toObject(FD.decode(b), { bytes: Uint8Array, defaults: true }) as { data_header: Uint8Array; data_body: Uint8Array },
      { ticket },
      this.metadata(w.grpc),
      this.options(),
    );
    call.on("metadata", (md) => this.remember(md));
    const onAbort = () => call.cancel();
    signal?.addEventListener("abort", onAbort, { once: true });
    try {
      for await (const fd of call) {
        // Application-metadata-only messages carry no IPC header; framing one would read as end-of-stream.
        if (fd.data_header.length === 0) continue;
        for (const part of frame({ header: fd.data_header, body: fd.data_body })) yield part;
      }
      yield new Uint8Array([0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0]);
    } catch (e) {
      throw this.fail(e);
    } finally {
      signal?.removeEventListener("abort", onAbort);
    }
  }

  async setConf(key: string, value: string): Promise<void> {
    const q = `SET ${key} = ${value}`;
    await this.update(q);
  }

  async getConf(key: string): Promise<string | undefined> {
    const s = await this.execute(`SET ${key}`);
    for await (const b of s.batches()) {
      const row = b.numRows > 0 ? b.get(0) : null;
      const v = row ? row["value"] ?? row[b.schema.fields[1]?.name ?? "value"] : undefined;
      if (v !== undefined && v !== null) return String(v);
    }
    return undefined;
  }

  /** Run a statement that returns no rows (DDL, DML); resolves with the affected row count. */
  async update(sql: string): Promise<number> {
    const { w, client } = await this.conn();
    const cmd = this.command(w, "CommandStatementUpdate", sql);
    const FD = w.root.lookupType("FlightData");
    const FDesc = w.root.lookupType("FlightDescriptor");
    const Res = w.root.lookupType("DoPutUpdateResult");
    const PR = w.root.lookupType("PutResult");
    return new Promise<number>((resolve, reject) => {
      const call = client.makeBidiStreamRequest<object, Uint8Array>(
        "/arrow.flight.protocol.FlightService/DoPut",
        (m: object) => Buffer.from(FD.encode(FD.create(m)).finish()),
        (b: Buffer) => PR.toObject(PR.decode(b), { bytes: Uint8Array, defaults: true }).app_metadata as Uint8Array,
        this.metadata(w.grpc),
        this.options(),
      );
      let count = 0;
      call.on("metadata", (md) => this.remember(md));
      call.on("data", (meta: Uint8Array) => {
        if (meta.length > 0) count = Number(Res.toObject(Res.decode(meta), { longs: Number }).record_count ?? 0);
      });
      call.on("error", (e) => reject(this.fail(e)));
      call.on("end", () => resolve(count));
      call.write({ flight_descriptor: FDesc.create({ type: 2, cmd }) });
      call.end();
    });
  }

  async close(): Promise<void> {
    this.client?.close();
    this.client = undefined;
  }
}

async function schemaOf(bytes: Uint8Array): Promise<Schema> {
  // FlightInfo.schema is an IPC schema message, with or without the encapsulation (continuation + length).
  const encapsulated = bytes.length >= 8 && new DataView(bytes.buffer, bytes.byteOffset).getUint32(0, true) === 0xffffffff;
  const msg = encapsulated ? bytes : concat(frame({ header: bytes, body: new Uint8Array(0) }));
  const reader = await RecordBatchReader.from(
    (async function* () {
      yield msg;
      yield new Uint8Array([0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0]);
    })(),
  );
  await reader.open();
  return reader.schema!;
}

function concat(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}
