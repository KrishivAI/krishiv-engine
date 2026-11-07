<script lang="ts" module>
  import { tv, type VariantProps } from "tailwind-variants";

  export const alertVariants = tv({
    base: "grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 rounded-lg border border-l-[3px] border-border bg-card px-4 py-3",
    variants: {
      variant: {
        default: "",
        warning: "border-l-warning [&>i]:text-warning",
        error: "border-l-error [&>i]:text-error",
      },
    },
    defaultVariants: { variant: "default" },
  });
  export type AlertVariant = VariantProps<typeof alertVariants>["variant"];
</script>

<script lang="ts">
  import type { HTMLAttributes } from "svelte/elements";
  import { cn } from "$lib/utils";

  let {
    class: className,
    variant = "default",
    icon = "",
    title,
    children,
    ...rest
  }: HTMLAttributes<HTMLDivElement> & { variant?: AlertVariant; icon?: string; title: string } = $props();
</script>

<div role="status" class={cn(alertVariants({ variant }), className)} {...rest}>
  <i class="row-span-2 leading-5 not-italic" aria-hidden="true">{icon}</i>
  <b class="text-sm leading-5 font-bold">{title}</b>
  <span class="text-text-2 wrap-anywhere">{@render children?.()}</span>
</div>
