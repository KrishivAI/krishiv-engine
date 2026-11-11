<script lang="ts" module>
  import { tv, type VariantProps } from "tailwind-variants";

  export const buttonVariants = tv({
    base: "inline-flex min-h-10 shrink-0 cursor-pointer items-center justify-center gap-2 rounded-md border px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors disabled:cursor-not-allowed disabled:opacity-45",
    variants: {
      variant: {
        default: "border-transparent bg-primary text-primary-foreground hover:bg-primary-hover",
        outline: "border-input bg-card text-foreground hover:bg-accent",
        ghost: "border-transparent bg-transparent text-foreground hover:bg-accent",
        destructive: "border-error/45 bg-card text-error hover:bg-error-bg",
      },
      size: { default: "", icon: "size-10 p-0" },
    },
    defaultVariants: { variant: "outline", size: "default" },
  });

  export type ButtonVariant = VariantProps<typeof buttonVariants>["variant"];
  export type ButtonSize = VariantProps<typeof buttonVariants>["size"];
</script>

<script lang="ts">
  import type { HTMLButtonAttributes } from "svelte/elements";
  import { cn } from "$lib/utils";

  let {
    class: className,
    variant = "outline",
    size = "default",
    children,
    ...rest
  }: HTMLButtonAttributes & { variant?: ButtonVariant; size?: ButtonSize } = $props();
</script>

<button class={cn(buttonVariants({ variant, size }), className)} {...rest}>
  {@render children?.()}
</button>
