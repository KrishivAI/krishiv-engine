<script lang="ts">
  import type { HTMLAttributes } from "svelte/elements";
  import { cn } from "../../utils";

  let { class: className, children, ...rest }: HTMLAttributes<HTMLDivElement> = $props();
</script>

<div class={cn("flex flex-wrap justify-end gap-2 border-t border-border px-5 py-3", className)} {...rest}>
  {@render children?.()}
</div>
