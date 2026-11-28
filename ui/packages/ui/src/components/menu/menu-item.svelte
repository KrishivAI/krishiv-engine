<script lang="ts" module>
  import { tv, type VariantProps } from "tailwind-variants";

  export const menuItemVariants = tv({
    base: "flex min-h-9 w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 text-left text-sm text-text-2 select-none data-disabled:cursor-not-allowed data-disabled:opacity-45 data-highlighted:bg-accent data-highlighted:text-foreground pointer-coarse:min-h-11",
    variants: {
      variant: {
        default: "",
        danger: "text-error data-highlighted:bg-error-bg data-highlighted:text-error",
      },
    },
    defaultVariants: { variant: "default" },
  });
  export type MenuItemVariant = VariantProps<typeof menuItemVariants>["variant"];
</script>

<script lang="ts">
  import { DropdownMenu as MenuPrimitive } from "bits-ui";
  import { cn } from "../../utils";

  let { class: className, variant = "default", ...rest }: MenuPrimitive.ItemProps & { variant?: MenuItemVariant } = $props();
</script>

<MenuPrimitive.Item class={cn(menuItemVariants({ variant }), className)} {...rest} />
