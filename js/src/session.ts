// SPDX-License-Identifier: Apache-2.0
import type { Backend } from "./backend.js";
import { DataFrame } from "./dataframe.js";
import { embeddedBackend } from "./embedded.js";
import { KrishivError } from "./errors.js";
import { parseRemoteUrl, RemoteBackend } from "./remote/flight.js";

export class Reader {
  constructor(private readonly backend: Backend) {}
  /** `SELECT * FROM <format>.`path``, which Spark SQL reads straight from files. */
  private from(format: "parquet" | "csv" | "json", path: string): DataFrame {
    return new DataFrame(this.backend, `SELECT * FROM ${format}.\`${path.replaceAll("`", "``")}\``);
  }
  parquet(path: string): DataFrame { return this.from("parquet", path); }
  csv(path: string): DataFrame { return this.from("csv", path); }
  json(path: string): DataFrame { return this.from("json", path); }
}

export class Conf {
  constructor(private readonly backend: Backend) {}
  set(key: string, value: string | number | boolean): Promise<void> { return this.backend.setConf(key, String(value)); }
  get(key: string): Promise<string | undefined> { return this.backend.getConf(key); }
}

export class Session {
  readonly conf: Conf;
  readonly read: Reader;
  /** @internal use `Session.builder()` */
  constructor(private readonly backend: Backend) {
    this.conf = new Conf(backend);
    this.read = new Reader(backend);
  }

  static builder(): SessionBuilder { return new SessionBuilder(); }

  sql(text: string): DataFrame { return new DataFrame(this.backend, text); }
  close(): Promise<void> { return this.backend.close(); }
  async [Symbol.asyncDispose](): Promise<void> { await this.close(); }
}

export class SessionBuilder {
  private url: string | undefined;
  private tok: string | undefined;
  private timeout: number | undefined;
  private readonly confs: Record<string, string> = {};
  private readonly hdrs: Record<string, string> = {};

  /** Connect to a Krishiv server: `krishiv://host:port` (plaintext) or `krishivs://host:port` (TLS). */
  remote(url: string): this { this.url = url; return this; }
  /** Bearer token for the server's authentication. */
  token(token: string): this { this.tok = token; return this; }
  header(name: string, value: string): this { this.hdrs[name] = value; return this; }
  timeoutMs(ms: number): this { this.timeout = ms; return this; }
  config(key: string, value: string | number | boolean): this { this.confs[key] = String(value); return this; }

  async getOrCreate(): Promise<Session> {
    let backend: Backend;
    if (this.url !== undefined) {
      backend = new RemoteBackend({ ...parseRemoteUrl(this.url), token: this.tok, headers: this.hdrs, timeoutMs: this.timeout });
    } else {
      if (this.tok !== undefined) throw new KrishivError("token() applies to remote sessions only");
      backend = await embeddedBackend(this.confs);
    }
    const s = new Session(backend);
    for (const [k, v] of Object.entries(this.confs)) if (this.url !== undefined) await s.conf.set(k, v);
    return s;
  }
}
