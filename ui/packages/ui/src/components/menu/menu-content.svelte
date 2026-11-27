<script lang="ts">
  import { DropdownMenu as MenuPrimitive } from "bits-ui";
  import { cn } from "../../utils";

  // A floating layer (one of the few things with a shadow), positioned by Bits UI through CSSOM.
  let { class: className, sideOffset = 4, align = "start", ...rest }: MenuPrimitive.ContentProps = $props();
</script>

<MenuPrimitive.Portal>
  <MenuPrimitive.Content
    {sideOffset}
    {align}
    class={cn("z-50 min-w-56 rounded-lg border border-border-strong bg-popover p-1.5 text-popover-foreground shadow-pop", className)}
    {...rest}
  />
</MenuPrimitive.Portal>
