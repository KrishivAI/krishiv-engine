<script lang="ts">
  import type { Snippet } from "svelte";

  let { title, children }: { title: string; children: Snippet } = $props();
</script>

<section class="grid grid-cols-[minmax(0,1fr)] gap-3">
  <h2>{title}</h2>
  <div class="flex flex-wrap items-start gap-3 rounded-lg border border-border bg-background p-4">{@render children()}</div>
</section>
