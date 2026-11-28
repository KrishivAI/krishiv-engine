// SPDX-License-Identifier: Apache-2.0

/** What a field hands its control: spread it on the input, select trigger or textarea the field labels. */
export interface FieldControl {
  id: string;
  "aria-describedby": string | undefined;
  "aria-invalid": "true" | undefined;
}

/**
 * The ids a field's help and error text take, and the `aria-describedby` that names them: the error first, then
 * the help, then any ids the caller adds (`extra`, a space-separated list; never dropped, never repeated).
 */
export function fieldIds(
  id: string,
  help: boolean,
  error: boolean,
  extra?: string,
): { help: string; error: string; describedby: string | undefined } {
  const ids = { help: `${id}-help`, error: `${id}-error` };
  const named = [...new Set([error ? ids.error : "", help ? ids.help : "", ...(extra ?? "").split(/\s+/)].filter(Boolean))];
  return { ...ids, describedby: named.length ? named.join(" ") : undefined };
}

/**
 * The attributes a field's control takes. With an error the control is always invalid; without one the caller's
 * own `aria-invalid` (if any) stands.
 */
export function fieldControl(
  id: string,
  help: boolean,
  error: boolean,
  extra?: { describedby?: string; invalid?: "true" | "false" | boolean },
): FieldControl {
  const invalid = error || extra?.invalid === true || extra?.invalid === "true" ? "true" : undefined;
  return { id, "aria-describedby": fieldIds(id, help, error, extra?.describedby).describedby, "aria-invalid": invalid };
}
