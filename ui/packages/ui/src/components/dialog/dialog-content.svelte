<script lang="ts">
  import { Dialog as DialogPrimitive } from "bits-ui";
  import type { Snippet } from "svelte";
  import { cn } from "../../utils";

  // A modal for a decision that needs attention: the title is the question, the footer repeats the verb. Bits UI
  // traps focus, closes on Escape and returns focus to the trigger.
  let { class: className, children, ...rest }: DialogPrimitive.ContentProps & { children?: Snippet } = $props();
</script>

<DialogPrimitive.Portal>
  <DialogPrimitive.Overlay class="fixed inset-0 z-40 bg-black/60" />
  <DialogPrimitive.Content
    class={cn(
      "fixed top-1/2 left-1/2 z-50 grid max-h-[calc(100dvh-2rem)] w-[min(520px,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 overflow-auto rounded-xl border border-border-strong bg-popover text-popover-foreground shadow-pop",
      className,
    )}
    {...rest}
  >
    {@render children?.()}
  </DialogPrimitive.Content>
</DialogPrimitive.Portal>
