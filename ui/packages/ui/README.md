# @krishiv/ui

Krishiv's console tokens, components and data helpers, shipped as Svelte source (Apache-2.0). The engine console
(`ui/apps/console`) and the Krishiv Console compile it with their own Svelte and Tailwind, so a page holds one copy
of each. Private until its first publication with an engine release; until then a console pins a commit.

| Import | What |
|---|---|
| `@krishiv/ui/tokens.css` | The engine console's tokens (`--k-*`), fonts and Tailwind theme; import it after `tailwindcss` |
| `@krishiv/ui/tokens/product.css` | The unified product tokens (brand + product additions, decision UI11), generated |
| `@krishiv/ui/tokens/<file>` | Marks and fonts |
| `@krishiv/ui/components/<name>` | `alert`, `badge`, `button`, `card`, `id`, `input`, `responsive-table`, `sheet`, `table` |
| `@krishiv/ui/data/<name>` | `format`, `plan`, `types` |
| `@krishiv/ui/utils` | `cn` |

`tokens.css` tells Tailwind to scan the package's components (its own `@source` rules), so a console needs no path of
its own. Import **one** of `tokens.css` (the engine console's `--k-*` set, TTF fonts) and `tokens/product.css` (the
unified set, woff2 fonts): both declare the Red Hat font faces.

`tokens-src/brand-tokens.json` and `tokens-src/product-tokens.json` are the token source; `pnpm tokens` writes
`src/tokens/product.css` from them and `pnpm test` fails when the committed file differs. Fonts: SIL OFL 1.1,
`src/tokens/fonts/LICENSE.txt`.
