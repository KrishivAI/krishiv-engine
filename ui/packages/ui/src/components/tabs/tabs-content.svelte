<script lang="ts">
  import { Tabs as TabsPrimitive } from "bits-ui";
  import { cn } from "../../utils";

  let { ref = $bindable(null), class: className, ...rest }: TabsPrimitive.ContentProps = $props();
</script>

<TabsPrimitive.Content bind:ref class={cn("pt-4", className)} {...rest} />
