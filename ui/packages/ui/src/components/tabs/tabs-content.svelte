<script lang="ts">
  import { Tabs as TabsPrimitive } from "bits-ui";
  import { cn } from "../../utils";

  let { class: className, ...rest }: TabsPrimitive.ContentProps = $props();
</script>

<TabsPrimitive.Content class={cn("pt-4", className)} {...rest} />
