<script lang="ts">
  import { QueryClient, QueryClientProvider } from "@tanstack/svelte-query";
  import { initToken } from "./api";
  import Health from "./pages/Health.svelte";
  import Queries from "./pages/Queries.svelte";
  import QueryDetail from "./pages/QueryDetail.svelte";
  import History from "./pages/History.svelte";
  import Resources from "./pages/Resources.svelte";
  import { Button } from "$lib/components/ui/button";
  import { Sheet, SheetClose, SheetContent, SheetDescription, SheetTitle, SheetTrigger } from "$lib/components/ui/sheet";
  import { cn } from "$lib/utils";
  import markDark from "./brand/mark-dark-flat.svg";
  import markLight from "./brand/mark-light.svg";
  import "./app.css";

  const hasToken = initToken() !== null;
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });

  let hash = $state(window.location.hash || "#/health");
  let menuOpen = $state(false);
  $effect(() => {
    const on = () => {
      hash = window.location.hash || "#/health";
      menuOpen = false;
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  });

  const parts = $derived(hash.replace(/^#\//, "").split("/"));
  const page = $derived(parts[0]);
  const links = [
    ["health", "Health"],
    ["queries", "Queries"],
    ["history", "History"],
    ["resources", "Resources"],
  ];
  const linkClass = (id: string, wide: boolean) =>
    cn(
      "rounded-md px-3 text-sm font-medium text-text-2 no-underline hover:bg-accent",
      wide ? "py-2.5" : "flex min-h-12 items-center",
      page === id && "bg-accent text-foreground",
    );
</script>

<QueryClientProvider {client}>
  <header class="sticky top-0 z-30 flex items-center gap-4 border-b border-border bg-card px-4 py-1.5">
    <picture class="flex shrink-0">
      <source media="(prefers-color-scheme: dark)" srcset={markDark} />
      <img class="h-6 w-auto" src={markLight} alt="Krishiv" />
    </picture>
    <nav class="hidden gap-0.5 md:flex" aria-label="Pages">
      {#each links as [id, label]}
        <a href={`#/${id}`} class={linkClass(id, true)} aria-current={page === id ? "page" : undefined}>{label}</a>
      {/each}
    </nav>
    <div class="ml-auto md:hidden">
      <Sheet bind:open={menuOpen}>
        <SheetTrigger>
          {#snippet child({ props })}
            <Button {...props} variant="ghost" aria-label="Open the menu">
              <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                <path d="M2 4.5h14M2 9h14M2 13.5h14" />
              </svg>
              Menu
            </Button>
          {/snippet}
        </SheetTrigger>
        <SheetContent>
          <SheetTitle class="font-display text-base font-bold">Krishiv</SheetTitle>
          <SheetDescription class="sr-only">Pages of the admin UI</SheetDescription>
          <nav class="flex flex-col" aria-label="Pages">
            {#each links as [id, label]}
              <SheetClose>
                {#snippet child({ props })}
                  <a {...props} href={`#/${id}`} class={linkClass(id, false)} aria-current={page === id ? "page" : undefined}>{label}</a>
                {/snippet}
              </SheetClose>
            {/each}
          </nav>
        </SheetContent>
      </Sheet>
    </div>
  </header>
  <main class="mx-auto max-w-7xl px-3 py-4 md:px-4 md:py-6">
    {#if !hasToken}
      <h1 class="mb-4">Open the address the server printed</h1>
      <p class="text-muted-foreground">
        The UI needs the token that <code>krishiv server</code> prints at start-up: open the
        <code>http://…/?token=…</code> address it shows.
      </p>
    {:else if page === "queries" && parts[1]}
      <QueryDetail id={decodeURIComponent(parts[1])} />
    {:else if page === "queries"}
      <Queries />
    {:else if page === "history"}
      <History />
    {:else if page === "resources"}
      <Resources />
    {:else}
      <Health />
    {/if}
  </main>
</QueryClientProvider>
