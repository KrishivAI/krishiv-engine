// SPDX-License-Identifier: Apache-2.0
import { readdirSync } from "node:fs";
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
  // Node's diagnostic report names the glibc it runs on; on musl the field is absent. Without network data:
  // collecting it resolves interface names, which is slow on some hosts (napi-rs's own loader does the same).
  // `excludeNetwork` is in Node 22's report API (v22.13) but not in @types/node 22's declaration of it.
  const report = process.report as (NodeJS.ProcessReport & { excludeNetwork?: boolean }) | undefined;
  if (report !== undefined) {
    const before = report.excludeNetwork;
    report.excludeNetwork = true;
    try {
      const header = (report.getReport() as { header?: { glibcVersionRuntime?: string } }).header;
      if (header?.glibcVersionRuntime !== undefined) return true;
    } finally {
      report.excludeNetwork = before;
    }
  }
  // No glibc in the report: on musl, or a runtime whose report does not name it (Bun may not). musl's dynamic
  // loader is `/lib/ld-musl-<arch>.so.1`; without one the system is taken to be glibc.
  return !hasMuslLoader();
}

function hasMuslLoader(): boolean {
  try {
    return readdirSync("/lib").some((f) => f.startsWith("ld-musl-"));
  } catch {
    return false;
  }
}

function packageVersion(req: NodeRequire): string | undefined {
  try {
    return (req("../package.json") as { version?: string }).version;
  } catch {
    return undefined;
  }
}

/**
 * Refuses a binary of another version than the package (the platform packages are published in lockstep). The
 * `KRISHIV_NATIVE_ENGINE` override (local builds, CI) is not checked: it names a binary on purpose.
 */
export function checkVersion(pkg: string, got: string, want: string | undefined): void {
  if (want !== undefined && got !== want) {
    throw new KrishivError(`${pkg} is version ${got} but krishiv is ${want}; install the same version of both`, {
      errorClass: "KRISHIV_NATIVE_ENGINE_UNAVAILABLE",
    });
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
  checkVersion(pkg, addon.version(), packageVersion(req));
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
  const o = j as { message: string; errorClass?: unknown; sqlState?: unknown; messageParameters?: unknown };
  const messageParameters: Record<string, string> = {};
  if (typeof o.messageParameters === "object" && o.messageParameters !== null && !Array.isArray(o.messageParameters)) {
    for (const [k, v] of Object.entries(o.messageParameters)) if (typeof v === "string") messageParameters[k] = v;
  }
  return new KrishivError(o.message, {
    errorClass: typeof o.errorClass === "string" ? o.errorClass : undefined,
    sqlState: typeof o.sqlState === "string" ? o.sqlState : undefined,
    messageParameters,
    cause: e,
  });
}

/** The error of an abort, in either backend's shape: `OPERATION_CANCELED`, the reason as its cause. */
export function cancelled(signal: AbortSignal): KrishivError {
  return new KrishivError("The operation has been cancelled.", {
    errorClass: "OPERATION_CANCELED",
    sqlState: "HY008",
    cause: signal.reason,
  });
}

/** The Arrow IPC end-of-stream marker (continuation, zero length), exactly the addon's last chunk. */
function isEndOfStream(c: Uint8Array): boolean {
  return (
    c.length === 8 && c[0] === 0xff && c[1] === 0xff && c[2] === 0xff && c[3] === 0xff &&
    c[4] === 0 && c[5] === 0 && c[6] === 0 && c[7] === 0
  );
}

function native<T>(f: () => T): T {
  try {
    return f();
  } catch (e) {
    throw fromNative(e);
  }
}

/**
 * The IPC chunks of `query`; a break or an abort before the end cancels it. Arrow JS stops reading at the
 * end-of-stream marker and returns the source without asking for the `null` after it, so the marker itself ends
 * the query (a cancel after the end would be a no-op anyway).
 */
async function* chunks(query: NativeQuery, signal?: AbortSignal): AsyncGenerator<Uint8Array> {
  // Runs in an event listener, where a throw (only a panic in the addon) would be uncaught.
  // A failing cancel is swallowed: the query ends on its next step anyway.
  const cancelQuietly = () => {
    try {
      query.cancel();
    } catch {
      // ignored, see above
    }
  };
  signal?.addEventListener("abort", cancelQuietly, { once: true });
  // An abort between `execute` and here fired no event.
  if (signal?.aborted) cancelQuietly();
  let done = false;
  try {
    for (;;) {
      let c: Uint8Array | null;
      try {
        c = await query.next();
      } catch (e) {
        done = true;
        const err = fromNative(e);
        // A cancel of ours from the signal is the abort, whatever the timing.
        if (signal?.aborted && err instanceof KrishivError && err.errorClass === "OPERATION_CANCELED") {
          throw cancelled(signal);
        }
        throw err;
      }
      if (c === null) {
        done = true;
        return;
      }
      if (isEndOfStream(c)) done = true;
      yield c;
    }
  } finally {
    signal?.removeEventListener("abort", cancelQuietly);
    if (!done) cancelQuietly();
  }
}

/**
 * A `Backend` over an addon session. A result holds its engine query until its batches are read to the end, its
 * iteration stops (`break`), its signal aborts or the session closes (which ends every running query); a result
 * that is never read keeps its query until it is garbage collected, so read or abort the results you start.
 */
export class EmbeddedBackend implements Backend {
  private closed = false;
  constructor(private readonly session: NativeSession) {}

  async execute(sql: string, opts: { signal?: AbortSignal } = {}): Promise<ResultStream> {
    if (opts.signal?.aborted) throw cancelled(opts.signal);
    const query = native(() => this.session.execute(sql));
    // Reading the schema starts the query, so analysis errors reject here, as in remote mode.
    const source = chunks(query, opts.signal);
    const reader = await RecordBatchReader.from(source);
    await reader.open();
    let used = false;
    return {
      schema: reader.schema,
      async *batches(): AsyncIterable<RecordBatch> {
        if (used) throw new KrishivError("a result's batches can be read once");
        used = true;
        let ended = false;
        try {
          for await (const b of reader) yield b;
          ended = true;
        } finally {
          // Arrow's reader does not pass a `break` on to its source: closing the generator here runs its `finally`,
          // which cancels a query not read to the end (closing it again after the end is a no-op).
          if (!ended) await source.return(undefined);
        }
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
    native(() => this.session.close());
  }
}

/** The embedded engine: the napi addon from the optional package `@krishiv/engine-<os>-<arch>[-<libc>]`. */
export async function embeddedBackend(conf: Record<string, string>, addon?: NativeAddon): Promise<Backend> {
  const a = addon ?? loadAddon();
  return new EmbeddedBackend(native(() => a.NativeSession.open(conf)));
}
