<script lang="ts">
  import { Tabs as TabsPrimitive } from "bits-ui";
  import { cn } from "../../utils";

  // Tabs scroll sideways on a phone instead of wrapping. Vertical tabs (orientation="vertical", whose arrows are
  // Up and Down) stand in a column with the rule on their right.
  let { ref = $bindable(null), class: className, ...rest }: TabsPrimitive.ListProps = $props();
</script>

<TabsPrimitive.List bind:ref class={cn("flex gap-1 overflow-x-auto border-b border-border [scrollbar-width:none] data-[orientation=vertical]:flex-col data-[orientation=vertical]:overflow-x-visible data-[orientation=vertical]:overflow-y-auto data-[orientation=vertical]:border-r data-[orientation=vertical]:border-b-0", className)} {...rest} />
