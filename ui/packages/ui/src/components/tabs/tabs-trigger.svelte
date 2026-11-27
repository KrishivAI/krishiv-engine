<script lang="ts">
  import { Tabs as TabsPrimitive } from "bits-ui";
  import { cn } from "../../utils";

  // The active tab is the gold element of its region: a 2 px underline, and its text at full strength.
  let { class: className, ...rest }: TabsPrimitive.TriggerProps = $props();
</script>

<TabsPrimitive.Trigger
  class={cn(
    "relative inline-flex min-h-10 shrink-0 cursor-pointer items-center gap-1.5 bg-transparent px-3 text-sm font-medium whitespace-nowrap text-muted-foreground hover:text-foreground disabled:cursor-not-allowed disabled:opacity-45 data-[state=active]:text-foreground data-[state=active]:after:absolute data-[state=active]:after:inset-x-2 data-[state=active]:after:-bottom-px data-[state=active]:after:h-0.5 data-[state=active]:after:rounded-sm data-[state=active]:after:bg-primary pointer-coarse:min-h-11",
    className,
  )}
  {...rest}
/>
