<script lang="ts">
  import { Tabs as TabsPrimitive } from "bits-ui";
  import { cn } from "../../utils";

  // Tabs scroll sideways on a phone instead of wrapping.
  let { class: className, ...rest }: TabsPrimitive.ListProps = $props();
</script>

<TabsPrimitive.List class={cn("flex gap-1 overflow-x-auto border-b border-border [scrollbar-width:none]", className)} {...rest} />
