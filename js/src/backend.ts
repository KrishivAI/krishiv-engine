// SPDX-License-Identifier: Apache-2.0
import type { RecordBatch, Schema } from "apache-arrow";

/** The batches of one statement, in order. `batches()` can be iterated once. */
export interface ResultStream {
  readonly schema: Schema;
  batches(): AsyncIterable<RecordBatch>;
}

/** What a backend (embedded engine or remote server) implements; nothing else of it reaches `Session`. */
export interface Backend {
  execute(sql: string, opts?: { signal?: AbortSignal }): Promise<ResultStream>;
  setConf(key: string, value: string): Promise<void>;
  getConf(key: string): Promise<string | undefined>;
  close(): Promise<void>;
}
