<script lang="ts">
  import { DropdownMenu as MenuPrimitive } from "bits-ui";
  import { cn } from "../../utils";

  let { class: className, ...rest }: MenuPrimitive.SeparatorProps = $props();
</script>

<MenuPrimitive.Separator class={cn("my-1.5 h-px bg-border", className)} {...rest} />
