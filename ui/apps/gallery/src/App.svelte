<script lang="ts">
  import type { Component } from "svelte";
  import Badge from "./pages/Badge.svelte";
  import Button from "./pages/Button.svelte";
  import Callout from "./pages/Callout.svelte";
  import Dialog from "./pages/Dialog.svelte";
  import Field from "./pages/Field.svelte";
  import Grid from "./pages/Grid.svelte";
  import Menu from "./pages/Menu.svelte";
  import Select from "./pages/Select.svelte";
  import States from "./pages/States.svelte";
  import Tabs from "./pages/Tabs.svelte";

  // One page per component, by hash (#/select): the page check opens each under the production CSP.
  const PAGES: [string, string, Component][] = [
    ["button", "Button", Button],
    ["field", "Field", Field],
    ["select", "Select", Select],
    ["callout", "Callout", Callout],
    ["badge", "Badge", Badge],
    ["dialog", "Dialog", Dialog],
    ["tabs", "Tabs", Tabs],
    ["menu", "Menu", Menu],
    ["grid", "Grid shell", Grid],
    ["states", "Empty, loading, error", States],
  ];

  let hash = $state(location.hash);
  $effect(() => {
    const on = () => (hash = location.hash);
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  });
  const current = $derived(PAGES.find(([slug]) => `#/${slug}` === hash) ?? PAGES[0]);
  const Page = $derived(current[2]);
</script>

<div class="grid min-h-dvh grid-cols-[minmax(0,1fr)] md:grid-cols-[14rem_minmax(0,1fr)]">
  <nav aria-label="Components" class="border-b border-border bg-muted p-3 md:border-r md:border-b-0">
    <p class="px-2 pb-2 font-display font-bold">@krishiv/ui</p>
    <ul class="flex flex-wrap gap-1 md:grid">
      {#each PAGES as [slug, name] (slug)}
        <li>
          <a
            href={`#/${slug}`}
            aria-current={current[0] === slug ? "page" : undefined}
            class="flex min-h-10 items-center rounded-md px-2 text-text-2 no-underline hover:bg-accent hover:text-foreground aria-[current=page]:bg-accent aria-[current=page]:text-foreground"
            >{name}</a
          >
        </li>
      {/each}
    </ul>
  </nav>
  <main class="grid min-w-0 grid-cols-[minmax(0,1fr)] content-start gap-6 p-4 md:p-6">
    <Page />
  </main>
</div>
