<script lang="ts">
  import type { HTMLAttributes } from "svelte/elements";
  import { cn } from "$lib/utils";

  let { class: className, children, ...rest }: HTMLAttributes<HTMLDivElement> = $props();
</script>

<div class={cn("text-xs font-bold tracking-[0.06em] text-muted-foreground uppercase", className)} {...rest}>
  {@render children?.()}
</div>
