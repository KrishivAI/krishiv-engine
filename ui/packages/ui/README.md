# @krishiv/ui

Krishiv's console tokens, components and data helpers, shipped as Svelte source (Apache-2.0). The engine console
(`ui/apps/console`) and the Krishiv Console compile it with their own Svelte and Tailwind, so a page holds one copy
of each. Private until its first publication with an engine release; until then a console pins a commit.

| Import | What |
|---|---|
| `@krishiv/ui/tokens.css` | The engine console's tokens (`--k-*`), fonts and Tailwind theme; import it after `tailwindcss` |
| `@krishiv/ui/tokens/product.css` | The unified product tokens (brand + product additions, decision UI11), generated |
| `@krishiv/ui/tokens/<file>` | Marks and fonts |
| `@krishiv/ui/components/<name>` | `alert`, `badge`, `button`, `callout`, `card`, `dialog`, `empty-state`, `error-state`, `field`, `grid`, `id`, `input`, `loading-state`, `menu`, `responsive-table`, `select`, `sheet`, `table`, `tabs` |
| `@krishiv/ui/data/<name>` | `format`, `plan`, `types` |
| `@krishiv/ui/utils` | `cn` |

`tokens.css` tells Tailwind to scan the package's components (its own `@source` rules), so a console needs no path of
its own. Import **one** of `tokens.css` (the engine console's `--k-*` set, TTF fonts) and `tokens/product.css` (the
unified set, woff2 fonts): both declare the Red Hat font faces.

`tokens-src/brand-tokens.json` and `tokens-src/product-tokens.json` are the token source; `pnpm tokens` writes
`src/tokens/product.css` from them and `pnpm test` fails when the committed file differs. Fonts: SIL OFL 1.1,
`src/tokens/fonts/LICENSE.txt`.

## Components and the gallery

The components follow shadcn-svelte's layout (Bits UI 2, tailwind-variants, `cn()`), styled only through the token
classes. `components.json` is the shadcn-svelte CLI's configuration: Tailwind 4, the stylesheet
`src/tokens/index.css` (where the `--k-*` tokens are mapped to shadcn's variable names) and the `$ui` alias of
`tsconfig.json`. A console compiles these files and has no `$ui` alias, so a component taken from the registry has
its `$ui/` imports rewritten to relative paths (`src/components/policy.test.ts` refuses them).

No component writes a style attribute or a `<style>` element: the consoles run under `style-src 'self'`. Floating
layers (select, menu, dialog) are positioned by Bits UI through CSSOM, which the policy allows; the grid shell's
virtual rows are spacer rows sized with Svelte's `style:` (CSSOM). `policy.test.ts` refuses the markup forms.

`ui/apps/gallery` (private, never in the console's assets) has a page per component; `pnpm -C ui gallery` serves
it, `pnpm -C ui test` builds it, and `ui/tests/check_gallery.py` opens every page under the production CSP with
axe, and opens every popover kind (part of `check_pages.py`).

| Component | What |
|---|---|
| `button` | Variants `default` (the one gold action), `outline`, `ghost`, `destructive`; `size="icon"`; disabled; busy is `aria-busy="true"` with a `Spinner` (from `loading-state`) |
| `field` | `Field`: label above, help below, the error in words with an icon; draws a text input, or hands `FieldControl` attributes to the `control` snippet (a `Select`) |
| `select` | `Select`: one choice of `items` in a Bits UI listbox |
| `callout` | `Callout`: `note`, `warning`, `error`, `success`, `plain`; a title, a body and an `actions` snippet |
| `dialog` | `Dialog`, `DialogTrigger`, `DialogContent`, `DialogTitle`, `DialogDescription`, `DialogFooter`, `DialogClose` |
| `tabs` | `Tabs`, `TabsList`, `TabsTrigger`, `TabsContent`; the active tab is underlined in gold |
| `menu` | `Menu`, `MenuTrigger`, `MenuContent`, `MenuGroup`, `MenuLabel`, `MenuItem` (`variant="danger"`), `MenuSeparator` |
| `grid` | `GridShell`: TanStack Table (core) sorts, TanStack Virtual draws only the visible rows past 200; sticky header, column type under the name, numbers in mono, `null` in italics, a selected row; `gridModel` and the virtual-window helpers are plain functions |
| `empty-state`, `error-state`, `loading-state` | Empty (what will be here, the action that fills it), error (class, SQLSTATE, what to do, support id), loading (skeleton, then a `<progress>` with the stage) |
