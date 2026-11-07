<script lang="ts">
  import type { HTMLTableAttributes } from "svelte/elements";
  import { cn } from "$lib/utils";

  let { class: className, children, ...rest }: HTMLTableAttributes = $props();
</script>

<!-- The table scrolls inside its own box, never the page. -->
<div class="max-w-full overflow-x-auto rounded-lg border border-border bg-card">
  <table class={cn("w-full border-collapse", className)} {...rest}>{@render children?.()}</table>
</div>
