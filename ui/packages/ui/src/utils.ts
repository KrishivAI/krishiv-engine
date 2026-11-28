import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Class names, with Tailwind conflicts resolved (the shadcn helper). */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * A row key for a keyed list whose natural key can repeat: the key carries the occurrence number, or Svelte
 * refuses the duplicate key and the list never renders.
 */
export function keyOf<T>(rows: T[], key: (r: T) => string): (r: T) => string {
  const seen = new Map<string, number>();
  const keys = new Map<T, string>();
  for (const r of rows) {
    const base = key(r);
    const n = seen.get(base) ?? 0;
    seen.set(base, n + 1);
    keys.set(r, `${base}#${n}`);
  }
  return (r) => keys.get(r) ?? key(r);
}
