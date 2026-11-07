<script lang="ts">
  import { Dialog as SheetPrimitive } from "bits-ui";
  import type { Snippet } from "svelte";
  import { cn } from "$lib/utils";

  let {
    class: className,
    side = "left",
    children,
    ...rest
  }: SheetPrimitive.ContentProps & { side?: "left" | "right"; children?: Snippet } = $props();
</script>

<SheetPrimitive.Portal>
  <SheetPrimitive.Overlay class="fixed inset-0 z-40 bg-black/60" />
  <SheetPrimitive.Content
    class={cn(
      "fixed inset-y-0 z-50 flex w-72 max-w-[85vw] flex-col gap-2 border-border bg-card p-4 text-card-foreground shadow-lg",
      side === "left" ? "left-0 border-r" : "right-0 border-l",
      className,
    )}
    {...rest}
  >
    {@render children?.()}
  </SheetPrimitive.Content>
</SheetPrimitive.Portal>
