<script lang="ts">
  import { QueryClient, QueryClientProvider } from "@tanstack/svelte-query";
  import { initToken } from "./api";
  import Health from "./pages/Health.svelte";
  import Queries from "./pages/Queries.svelte";
  import QueryDetail from "./pages/QueryDetail.svelte";
  import History from "./pages/History.svelte";
  import Resources from "./pages/Resources.svelte";
  import "./app.css";

  const hasToken = initToken() !== null;
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });

  let hash = $state(window.location.hash || "#/health");
  $effect(() => {
    const on = () => (hash = window.location.hash || "#/health");
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
</script>

<QueryClientProvider {client}>
  <nav>
    <span class="brand">Krishiv</span>
    {#each links as [id, label]}
      <a href={`#/${id}`} class:active={page === id}>{label}</a>
    {/each}
  </nav>
  <main>
    {#if !hasToken}
      <h1>Open the address the server printed</h1>
      <p class="muted">
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
