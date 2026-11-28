<script lang="ts">
  import { DropdownMenu as MenuPrimitive } from "bits-ui";
  import { cn } from "../../utils";

  // The heading of a MenuGroup (what the items act on).
  let { class: className, ...rest }: MenuPrimitive.GroupHeadingProps = $props();
</script>

<MenuPrimitive.GroupHeading class={cn("px-2.5 pt-1.5 pb-1 text-xs font-semibold text-muted-foreground", className)} {...rest} />
