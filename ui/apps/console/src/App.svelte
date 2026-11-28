<script lang="ts">
  import { QueryClient, QueryClientProvider } from "@tanstack/svelte-query";
  import { api, ApiError, initSession } from "./api";
  import { consoleInfo, loadConsole, pagesOf } from "./console.svelte";
  import { applyTheme, embedLink, listenForTheme, validateTheme } from "./embed";
  import { parseRoute } from "./routes";
  import Catalog from "./pages/Catalog.svelte";
  import Editor from "./pages/Editor.svelte";
  import Environment from "./pages/Environment.svelte";
  import Health from "./pages/Health.svelte";
  import Queries from "./pages/Queries.svelte";
  import QueryDetail from "./pages/QueryDetail.svelte";
  import History from "./pages/History.svelte";
  import Resources from "./pages/Resources.svelte";
  import Sessions from "./pages/Sessions.svelte";
  import Workload from "./pages/Workload.svelte";
  import Advisor from "./pages/Advisor.svelte";
  import { Badge } from "@krishiv/ui/components/badge";
  import { Button } from "@krishiv/ui/components/button";
  import Id from "@krishiv/ui/components/id";
  import { Sheet, SheetClose, SheetContent, SheetDescription, SheetTitle, SheetTrigger } from "@krishiv/ui/components/sheet";
  import { cn } from "@krishiv/ui/utils";
  import markDark from "@krishiv/ui/tokens/mark-dark-flat.svg";
  import markLight from "@krishiv/ui/tokens/mark-light.svg";
  import "./app.css";

  // The embedded view (`?embed=1` on /queries/{id}/profile) has no session: its signed link is the credential.
  const embed = embedLink(window.location.search);
  // The mark shows only once the config says it may (no flash in an embed that hides it).
  let hideAttribution = $state(true);
  // "failed": the browser was let in, but the console's first answer did not arrive. Asking for the token
  // again would be wrong then (it is gone from the address, and the cookie is set): the page offers a retry.
  let session = $state<"checking" | "yes" | "no" | "failed">(embed ? "yes" : "checking");
  let startError = $state("");
  // The mode is read before the first page is drawn, so no page of another mode flashes by or asks for data
  // this console does not have.
  async function open(first: boolean) {
    session = "checking";
    try {
      if (first && !(await initSession())) {
        session = "no";
        return;
      }
      await loadConsole();
      session = "yes";
    } catch (e) {
      if (e instanceof ApiError && (e.status === 401 || e.status === 403)) {
        session = "no";
      } else {
        startError = e instanceof Error ? e.message : String(e);
        session = "failed";
      }
    }
  }
  if (!embed) void open(true);
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
  $effect(() => {
    if (!embed) return;
    document.documentElement.dataset.embed = "1";
    // A theme in the link, then messages from the embedding page (allowed origins only).
    if (embed.theme) {
      try {
        const r = validateTheme(JSON.parse(embed.theme));
        if (r.ok) applyTheme(r.theme);
        else console.warn(`krishiv: ignored the theme in the link: ${r.reason}`);
      } catch {
        console.warn("krishiv: ignored the theme in the link: not JSON");
      }
    }
    const config = api<{ allowed_origins: string[]; hide_attribution: boolean }>("/api/internal/embed/config");
    config.then(
      (c) => {
        hideAttribution = c.hide_attribution;
      },
      () => {},
    );
    return listenForTheme(config.then((c) => c.allowed_origins));
  });

  const route = $derived(parseRoute(window.location.pathname, hash, window.location.search));
  const labels: Record<string, string> = {
    health: "Health",
    queries: "Queries",
    history: "History",
    sessions: "Sessions",
    workload: "Workload",
    advisor: "Advisor",
    resources: "Resources",
    catalog: "Catalog",
    sql: "SQL",
    environment: "Environment",
  };
  const pages = $derived(pagesOf(consoleInfo.mode, consoleInfo.role));
  const links = $derived(pages.map((id) => [id, labels[id]]));
  // An address of a page this mode lacks (or none) opens the mode's first page. A stored profile is a query's
  // page, so `queries/<id>` stays reachable where the live list is not. The embedded view is always a query's.
  const page = $derived(
    pages.includes(route.page) || (route.page === "queries" && route.id) ? route.page : pages[0],
  );
  const linkClass = (id: string, wide: boolean) =>
    cn(
      "rounded-md px-3 text-sm font-medium text-text-2 no-underline hover:bg-accent",
      wide ? "flex min-h-11 items-center" : "flex min-h-12 items-center",
      // The active link: the surface fill, plus the gold accent as an inset underline (no layout change).
      page === id && "bg-accent text-foreground shadow-[inset_0_-2px_0_var(--k-accent)]",
    );
</script>

<QueryClientProvider {client}>
  {#if embed}
    <main class="mx-auto max-w-7xl px-3 py-4 md:px-4">
      {#if page === "queries" && route.id && route.sub === "profile"}
        <QueryDetail id={route.id} {embed} session={route.session} />
      {:else}
        <h1>This view cannot be embedded</h1>
      {/if}
      {#if !hideAttribution}
        <p class="mt-6 text-right text-xs text-muted-foreground">Powered by Krishiv</p>
      {/if}
    </main>
  {:else}
  <!-- The bar's items sit in the same centred box as the content, so on a very wide screen they line up with it
       instead of going to the screen's corners. -->
  <header class="sticky top-0 z-30 border-b border-border bg-card">
   <div data-shell class="mx-auto flex min-h-14 max-w-[120rem] items-center gap-4 px-3 md:px-4">
    <picture class="flex shrink-0">
      <source media="(prefers-color-scheme: dark)" srcset={markDark} />
      <img class="h-6 w-auto" src={markLight} alt="Krishiv" />
    </picture>
    <!-- Ten links do not fit one row on a tablet or at a large text size: the row wraps and the bar grows. -->
    <nav class="hidden min-w-0 flex-wrap gap-0.5 md:flex" aria-label="Pages">
      {#each links as [id, label]}
        <a href={`#/${id}`} class={linkClass(id, true)} aria-current={page === id ? "page" : undefined}>{label}</a>
      {/each}
    </nav>
    {#if session === "yes" && consoleInfo.mode !== "server"}
      <!-- Which console this is: a server shows every session and says nothing here. -->
      <p class="m-0 ml-auto flex min-w-0 items-center gap-1.5 text-sm text-muted-foreground" data-console-mode={consoleInfo.mode}>
        {#if consoleInfo.mode === "embedded"}
          <Badge>This session</Badge>
          {#if consoleInfo.sessionId}<span class="hidden sm:max-md:inline lg:inline"><Id id={consoleInfo.sessionId} /></span>{/if}
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
   </div>
  </header>
  <main class="mx-auto max-w-[120rem] px-3 py-4 md:px-4 md:py-6">
    {#if session === "checking"}
      <p class="text-muted-foreground">Opening the session…</p>
    {:else if session === "no"}
      <h1 class="mb-4">Open the address with the token</h1>
      <p class="text-muted-foreground">
        The console needs a token in its address. <code>krishiv ui</code>, <code>krishiv.ui()</code> and a
        <code>krishiv server</code> without configured tokens print an address ending in <code>?token=…</code> when
        they start: open that address. On a server with configured tokens (<code>krishiv.auth.*</code>, the Docker
        image) nothing is printed: add <code>?token=</code> and your token to this page's address.
      </p>
    {:else if session === "failed"}
      <h1 class="mb-4">The console did not load</h1>
      <p class="mb-4 text-muted-foreground">
        Krishiv did not answer the console's first request (<span data-start-error>{startError}</span>). This
        browser's session is kept, so the address with the token is not needed again.
      </p>
      <Button onclick={() => open(false)}>Try again</Button>
    {:else if page === "queries" && route.id}
      <QueryDetail id={route.id} session={route.session} />
    {:else if page === "queries"}
      <Queries />
    {:else if page === "history"}
      <History />
    {:else if page === "sessions"}
      <Sessions />
    {:else if page === "workload"}
      <Workload />
    {:else if page === "advisor"}
      <Advisor />
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
  {/if}
</QueryClientProvider>
