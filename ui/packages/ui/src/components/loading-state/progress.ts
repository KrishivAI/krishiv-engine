// SPDX-License-Identifier: Apache-2.0

// Fixed Tailwind classes (not computed widths): the scanner must see each class, and no style attribute is used.
const WIDTHS = ["w-3/5", "w-4/5", "w-2/5", "w-4/5"] as const;

/** The width classes of `n` skeleton lines: a heading line, then body lines of uneven length. */
export function skeletonWidths(n: number): string[] {
  const count = Math.max(0, Math.floor(n));
  return Array.from({ length: count }, (_, i) => WIDTHS[i % WIDTHS.length]);
}
