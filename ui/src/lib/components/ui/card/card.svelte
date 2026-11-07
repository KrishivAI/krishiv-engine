<script lang="ts">
  import type { HTMLAttributes } from "svelte/elements";
  import { cn } from "$lib/utils";

  let { class: className, children, ...rest }: HTMLAttributes<HTMLDivElement> = $props();
</script>

<div class={cn("min-w-0 rounded-lg border border-border bg-card p-4 text-card-foreground shadow-sm", className)} {...rest}>
  {@render children?.()}
</div>
