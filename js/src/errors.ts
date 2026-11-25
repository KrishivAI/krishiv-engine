// SPDX-License-Identifier: Apache-2.0

/** Every failure of the package. Carries the Spark error class and SQLSTATE when the engine gave them. */
export class KrishivError extends Error {
  readonly errorClass: string | undefined;
  readonly sqlState: string | undefined;
  /** The error class's message parameters (Spark's `getMessageParameters`); empty when the engine gave none. */
  readonly messageParameters: Readonly<Record<string, string>>;

  constructor(
    message: string,
    opts: { errorClass?: string; sqlState?: string; messageParameters?: Record<string, string>; cause?: unknown } = {},
  ) {
    super(message, opts.cause === undefined ? undefined : { cause: opts.cause });
    this.name = "KrishivError";
    this.errorClass = opts.errorClass;
    this.sqlState = opts.sqlState;
    this.messageParameters = Object.freeze({ ...(opts.messageParameters ?? {}) });
  }
}
