<script lang="ts">
  import { QueryClient, QueryClientProvider } from "@tanstack/svelte-query";
  import { initSession } from "./api";
  import { consoleInfo, loadConsole, pagesOf } from "./console.svelte";
  import Catalog from "./pages/Catalog.svelte";
  import Editor from "./pages/Editor.svelte";
  import Environment from "./pages/Environment.svelte";
  import Health from "./pages/Health.svelte";
  import Queries from "./pages/Queries.svelte";
  import QueryDetail from "./pages/QueryDetail.svelte";
  import History from "./pages/History.svelte";
  import Resources from "./pages/Resources.svelte";
  import { Badge } from "$lib/components/ui/badge";
  import { Button } from "$lib/components/ui/button";
  import Id from "$lib/components/id.svelte";
  import { Sheet, SheetClose, SheetContent, SheetDescription, SheetTitle, SheetTrigger } from "$lib/components/ui/sheet";
  import { cn } from "$lib/utils";
  import markDark from "./brand/mark-dark-flat.svg";
  import markLight from "./brand/mark-light.svg";
  import "./app.css";

  let session = $state<"checking" | "yes" | "no">("checking");
  // The mode is read before the first page is drawn, so no page of another mode flashes by or asks for data
  // this console does not have.
  initSession()
    .then((ok) => (ok ? loadConsole().then(() => true) : false))
    .then((ok) => (session = ok ? "yes" : "no"), () => (session = "no"));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });

  let hash = $state(window.location.hash);
  let menuOpen = $state(false);
  $effect(() => {
    const on = () => {
      hash = window.location.hash;
      menuOpen = false;
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  });

  const parts = $derived(hash.replace(/^#\//, "").split("/"));
  const labels: Record<string, string> = {
    health: "Health",
    queries: "Queries",
    history: "History",
    resources: "Resources",
    catalog: "Catalog",
    sql: "SQL",
    environment: "Environment",
  };
  const pages = $derived(pagesOf(consoleInfo.mode));
  const links = $derived(pages.map((id) => [id, labels[id]]));
  // An address of a page this mode lacks (or none) opens the mode's first page. A stored profile is a query's
  // page, so `queries/<id>` stays reachable where the live list is not.
  const page = $derived(pages.includes(parts[0]) || (parts[0] === "queries" && parts[1]) ? parts[0] : pages[0]);
  const linkClass = (id: string, wide: boolean) =>
    cn(
      "rounded-md px-3 text-sm font-medium text-text-2 no-underline hover:bg-accent",
      wide ? "py-2.5" : "flex min-h-12 items-center",
      // The active link: the surface fill, plus the gold accent as an inset underline (no layout change).
      page === id && "bg-accent text-foreground shadow-[inset_0_-2px_0_var(--k-accent)]",
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
    {#if session === "yes" && consoleInfo.mode !== "server"}
      <!-- Which console this is: a server shows every session and says nothing here. -->
      <p class="m-0 ml-auto flex min-w-0 items-center gap-1.5 text-sm text-muted-foreground" data-console-mode={consoleInfo.mode}>
        {#if consoleInfo.mode === "embedded"}
          <Badge>This session</Badge>
          {#if consoleInfo.sessionId}<span class="hidden sm:inline"><Id id={consoleInfo.sessionId} /></span>{/if}
        {:else}
          <Badge>Saved profiles</Badge>
        {/if}
      </p>
    {/if}
    <div class={cn("md:hidden", (session !== "yes" || consoleInfo.mode === "server") && "ml-auto")}>
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
    {#if session === "checking"}
      <p class="text-muted-foreground">Opening the session…</p>
    {:else if session === "no"}
      <h1 class="mb-4">Open the address Krishiv printed</h1>
      <p class="text-muted-foreground">
        The console needs the token in the <code>http://…/?token=…</code> address that <code>krishiv server</code>,
        <code>krishiv ui</code> or <code>krishiv.ui()</code> printed when it started: open that address.
      </p>
    {:else if page === "queries" && parts[1]}
      <QueryDetail id={decodeURIComponent(parts[1])} />
    {:else if page === "queries"}
      <Queries />
    {:else if page === "history"}
      <History />
    {:else if page === "resources"}
      <Resources />
    {:else if page === "catalog"}
      <Catalog />
    {:else if page === "sql"}
      <Editor />
    {:else if page === "environment"}
      <Environment />
    {:else}
      <Health />
    {/if}
  </main>
</QueryClientProvider>
