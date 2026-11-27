// SPDX-License-Identifier: Apache-2.0

/** What a field hands its control: spread it on the input, select trigger or textarea the field labels. */
export interface FieldControl {
  id: string;
  "aria-describedby": string | undefined;
  "aria-invalid": "true" | undefined;
}

/** The ids a field's help and error text take, and the `aria-describedby` that names them (error first). */
export function fieldIds(
  id: string,
  help: boolean,
  error: boolean,
): { help: string; error: string; describedby: string | undefined } {
  const ids = { help: `${id}-help`, error: `${id}-error` };
  const named = [error ? ids.error : "", help ? ids.help : ""].filter(Boolean);
  return { ...ids, describedby: named.length ? named.join(" ") : undefined };
}

/** The attributes a field's control takes. */
export function fieldControl(id: string, help: boolean, error: boolean): FieldControl {
  return { id, "aria-describedby": fieldIds(id, help, error).describedby, "aria-invalid": error ? "true" : undefined };
}
