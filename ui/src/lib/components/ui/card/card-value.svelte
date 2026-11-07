<script lang="ts">
  import type { HTMLAttributes } from "svelte/elements";
  import { cn } from "$lib/utils";

  let { class: className, children, ...rest }: HTMLAttributes<HTMLDivElement> = $props();
</script>

<div class={cn("mt-1 font-display text-xl leading-7 font-bold wrap-anywhere", className)} {...rest}>
  {@render children?.()}
</div>
