// SPDX-License-Identifier: Apache-2.0
import type { Backend } from "./backend.js";
import { KrishivError } from "./errors.js";

/** The embedded engine: the napi addon from the optional package `@krishiv/engine-<os>-<arch>`. */
export async function embeddedBackend(_conf: Record<string, string>): Promise<Backend> {
  const pkg = `@krishiv/engine-${process.platform}-${process.arch}`;
  throw new KrishivError(
    `the embedded engine is not available: no binary package ${pkg} is installed. ` +
      `Connect to a server with Session.builder().remote("krishiv://host:port") instead.`,
  );
}
