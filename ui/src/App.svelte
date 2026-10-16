<script lang="ts">
  import { QueryClient, QueryClientProvider } from "@tanstack/svelte-query";
  import Version from "./Version.svelte";

  const client = new QueryClient();
</script>

<QueryClientProvider {client}>
  <main>
    <h1>Krishiv</h1>
    <Version />
  </main>
</QueryClientProvider>

<style>
  main {
    font-family: system-ui, sans-serif;
    max-width: 40rem;
    margin: 4rem auto;
    padding: 0 1rem;
  }
</style>
