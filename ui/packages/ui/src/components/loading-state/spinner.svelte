<script lang="ts">
  import { cn } from "../../utils";

  // A busy mark for a button that is working (put `aria-busy="true"` on the button; the text stays).
  let { class: className }: { class?: string } = $props();
</script>

<span class={cn("inline-block size-4 animate-spin rounded-full border-2 border-current border-r-transparent", className)} aria-hidden="true"></span>
