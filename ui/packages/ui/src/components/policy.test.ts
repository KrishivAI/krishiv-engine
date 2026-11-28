// SPDX-License-Identifier: Apache-2.0
import { describe, expect, test } from "vitest";

// Vite's glob import (the package has no Node types): every source of the package, as text.
declare global {
  interface ImportMeta {
    glob(pattern: string[], options: { query: "?raw"; import: "default"; eager: true }): Record<string, string>;
  }
}

// The consoles run under `style-src 'self'`: a style attribute in markup, `setAttribute("style")` or a <style>
// element made at run time is refused by the browser. Styles go through classes, or Svelte's `style:` directive and
// `el.style` (CSSOM), which the policy allows. The page check (ui/tests/check_gallery.py) proves it in a browser;
// this catches the markup before a build.
const SOURCES = import.meta.glob(["../**/*.svelte", "../**/*.ts", "!../**/*.test.ts"], { query: "?raw", import: "default", eager: true });

const BANNED: [string, RegExp][] = [
  ["a style attribute", /<[a-zA-Z][^>]*\sstyle\s*=/],
  ['setAttribute("style")', /setAttribute\(\s*["'`]style["'`]/],
  ["a <style> element made at run time", /createElement\(\s*["'`]style["'`]/],
  // components.json's `$ui` alias is for the shadcn-svelte CLI only: a console compiles these files and has no such
  // alias, so a component taken from the registry has its `$ui/` imports rewritten to relative paths.
  ["an import through the `$ui` alias", /from\s+["']\$ui[/"']/],
];

describe("no run-time styles in the package", () => {
  const sources = Object.entries(SOURCES);
  test("finds the sources", () => {
    expect(sources.length).toBeGreaterThan(20);
  });
  for (const [what, re] of BANNED) {
    test(`no ${what}`, () => {
      expect(sources.filter(([, text]) => re.test(text)).map(([path]) => path)).toEqual([]);
    });
  }
});
