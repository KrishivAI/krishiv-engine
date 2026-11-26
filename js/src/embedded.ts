// SPDX-License-Identifier: Apache-2.0
import type { Backend } from "./backend.js";
import { KrishivError } from "./errors.js";

/** The embedded engine: not implemented yet, always throws. */
export async function embeddedBackend(_conf: Record<string, string>): Promise<Backend> {
  throw new KrishivError(
    `the embedded engine is not available: this package does not provide an in-process engine yet. ` +
      `Connect to a server with Session.builder().remote("krishiv://host:port") instead.`,
  );
}
