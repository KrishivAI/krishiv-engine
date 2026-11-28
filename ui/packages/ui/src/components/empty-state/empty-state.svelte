<script lang="ts">
  import type { Snippet } from "svelte";
  import type { HTMLAttributes } from "svelte/elements";
  import { cn } from "../../utils";

  // Empty says what will be here and how to make it appear: a title, one line, and the action that fills it.
  let {
    class: className,
    title,
    icon = "○",
    centered = false,
    children,
    actions,
    ...rest
  }: HTMLAttributes<HTMLDivElement> & { title: string; icon?: string; centered?: boolean; actions?: Snippet } = $props();
</script>

<div class={cn("grid max-w-[56ch] justify-items-start gap-2 p-7", centered && "mx-auto justify-items-center text-center", className)} {...rest}>
  <i class="grid size-10 place-items-center rounded-[10px] border border-dashed border-border-strong text-muted-foreground not-italic" aria-hidden="true">{icon}</i>
  <h3 class="text-base leading-6 font-bold">{title}</h3>
  {#if children}<p class="text-muted-foreground">{@render children()}</p>{/if}
  {#if actions}<div class="mt-1.5 flex flex-wrap gap-2">{@render actions()}</div>{/if}
</div>
