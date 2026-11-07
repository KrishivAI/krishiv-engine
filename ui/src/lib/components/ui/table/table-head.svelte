<script lang="ts">
  import type { HTMLThAttributes } from "svelte/elements";
  import { cn } from "$lib/utils";

  let { class: className, children, ...rest }: HTMLThAttributes = $props();
</script>

<th class={cn("border-b border-border bg-muted px-3.5 py-2.5 text-left align-top text-xs font-bold tracking-[0.06em] whitespace-nowrap text-muted-foreground uppercase", className)} {...rest}>
  {@render children?.()}
</th>
