<script lang="ts">
  import type { HTMLTdAttributes } from "svelte/elements";
  import { cn } from "$lib/utils";

  let { class: className, children, ...rest }: HTMLTdAttributes = $props();
</script>

<td class={cn("border-b border-border px-3.5 py-2.5 align-top whitespace-nowrap", className)} {...rest}>
  {@render children?.()}
</td>
