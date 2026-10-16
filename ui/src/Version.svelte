<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";

  interface VersionInfo {
    version: string;
  }

  const query = createQuery<VersionInfo>(() => ({
    queryKey: ["version"],
    queryFn: async () => {
      const res = await fetch("/api/v1/version");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    },
  }));
</script>

{#if query.isPending}
  <p>Loading version...</p>
{:else if query.isError}
  <p>Could not load version: {query.error.message}</p>
{:else}
  <p>Version {query.data.version}</p>
{/if}
