// SPDX-License-Identifier: Apache-2.0
import { createRequire } from "node:module";
import { RecordBatchReader, type RecordBatch } from "apache-arrow";
import type { Backend, ResultStream } from "./backend.js";
import { KrishivError } from "./errors.js";

/** The addon's surface (`crates/krishiv-node`): what the platform package's `.node` file exports. */
export interface NativeQuery {
  /** The next Arrow IPC chunk: the schema first, then one per batch, then the end-of-stream marker; then `null`. */
  next(): Promise<Uint8Array | null>;
  cancel(): void;
}
export interface NativeSession {
  execute(sql: string): NativeQuery;
  setConf(key: string, value: string): void;
  getConf(key: string): string | null | undefined;
  close(): void;
}
export interface NativeAddon {
  NativeSession: { open(conf: Record<string, string>): NativeSession };
  version(): string;
}

/** Set to the path of a `.node` file to load it instead of the platform package (local builds, CI). */
export const NATIVE_ENGINE_ENV = "KRISHIV_NATIVE_ENGINE";

/** The platform package for this process, e.g. `@krishiv/engine-linux-x64-gnu`. */
export function platformPackage(
  platform: string = process.platform,
  arch: string = process.arch,
  glibc: boolean = isGlibc(),
): string {
  if (platform === "linux") return `@krishiv/engine-linux-${arch}-${glibc ? "gnu" : "musl"}`;
  if (platform === "win32") return `@krishiv/engine-win32-${arch}-msvc`;
  return `@krishiv/engine-${platform}-${arch}`;
}

function isGlibc(): boolean {
  if (process.platform !== "linux") return false;
  // Node's diagnostic report names the glibc it runs on; on musl the field is absent.
  const report = process.report?.getReport() as { header?: { glibcVersionRuntime?: string } } | undefined;
  return report?.header?.glibcVersionRuntime !== undefined;
}

function packageVersion(req: NodeRequire): string | undefined {
  try {
    return (req("../package.json") as { version?: string }).version;
  } catch {
    return undefined;
  }
}

/** Loads the addon: never at import, only when a session picks embedded mode. */
export function loadAddon(env: NodeJS.ProcessEnv = process.env): NativeAddon {
  // `require` in the CommonJS build; in the ES module build it is not defined.
  const req: NodeRequire = typeof require === "function" ? require : createRequire(import.meta.url);
  const path = env[NATIVE_ENGINE_ENV];
  if (path !== undefined && path !== "") {
    try {
      return req(path) as NativeAddon;
    } catch (e) {
      throw new KrishivError(`the embedded engine at ${NATIVE_ENGINE_ENV}=${path} could not be loaded`, {
        errorClass: "KRISHIV_NATIVE_ENGINE_UNAVAILABLE",
        cause: e,
      });
    }
  }
  const pkg = platformPackage();
  let addon: NativeAddon;
  try {
    addon = req(pkg) as NativeAddon;
  } catch (e) {
    throw new KrishivError(
      `the embedded engine is not available: no binary package ${pkg} is installed ` +
        `(platform ${process.platform}, arch ${process.arch}). ` +
        `Connect to a server with Session.builder().remote("krishiv://host:port") instead.`,
      { errorClass: "KRISHIV_NATIVE_ENGINE_UNAVAILABLE", cause: e },
    );
  }
  const want = packageVersion(req);
  const got = addon.version();
  if (want !== undefined && got !== want) {
    throw new KrishivError(`${pkg} is version ${got} but krishiv is ${want}; install the same version of both`, {
      errorClass: "KRISHIV_NATIVE_ENGINE_UNAVAILABLE",
    });
  }
  return addon;
}

/** A native failure carries JSON `{errorClass, sqlState, messageParameters, message}` as its message. */
export function fromNative(e: unknown): unknown {
  if (!(e instanceof Error) || e instanceof KrishivError) return e;
  let j: unknown;
  try {
    j = JSON.parse(e.message);
  } catch {
    return e;
  }
  if (typeof j !== "object" || j === null || typeof (j as { message?: unknown }).message !== "string") return e;
  const o = j as { message: string; errorClass?: unknown; sqlState?: unknown };
  return new KrishivError(o.message, {
    errorClass: typeof o.errorClass === "string" ? o.errorClass : undefined,
    sqlState: typeof o.sqlState === "string" ? o.sqlState : undefined,
    cause: e,
  });
}

function native<T>(f: () => T): T {
  try {
    return f();
  } catch (e) {
    throw fromNative(e);
  }
}

/** The IPC chunks of `query`; a break or an abort cancels it. */
async function* chunks(query: NativeQuery, signal?: AbortSignal): AsyncGenerator<Uint8Array> {
  const onAbort = () => query.cancel();
  signal?.addEventListener("abort", onAbort, { once: true });
  let done = false;
  try {
    for (;;) {
      let c: Uint8Array | null;
      try {
        c = await query.next();
      } catch (e) {
        done = true;
        throw fromNative(e);
      }
      if (c === null) {
        done = true;
        return;
      }
      yield c;
    }
  } finally {
    signal?.removeEventListener("abort", onAbort);
    if (!done) query.cancel();
  }
}

/** A `Backend` over an addon session. */
export class EmbeddedBackend implements Backend {
  private closed = false;
  constructor(private readonly session: NativeSession) {}

  async execute(sql: string, opts: { signal?: AbortSignal } = {}): Promise<ResultStream> {
    opts.signal?.throwIfAborted();
    const query = native(() => this.session.execute(sql));
    // Reading the schema starts the query, so analysis errors reject here, as in remote mode.
    const reader = await RecordBatchReader.from(chunks(query, opts.signal));
    await reader.open();
    let used = false;
    return {
      schema: reader.schema,
      async *batches(): AsyncIterable<RecordBatch> {
        if (used) throw new KrishivError("a result's batches can be read once");
        used = true;
        for await (const b of reader) yield b;
      },
    };
  }

  async setConf(key: string, value: string): Promise<void> {
    native(() => this.session.setConf(key, value));
  }

  async getConf(key: string): Promise<string | undefined> {
    return native(() => this.session.getConf(key)) ?? undefined;
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    this.session.close();
  }
}

/** The embedded engine: the napi addon from the optional package `@krishiv/engine-<os>-<arch>[-<libc>]`. */
export async function embeddedBackend(conf: Record<string, string>, addon?: NativeAddon): Promise<Backend> {
  const a = addon ?? loadAddon();
  return new EmbeddedBackend(native(() => a.NativeSession.open(conf)));
}
