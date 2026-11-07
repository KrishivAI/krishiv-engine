<script lang="ts" module>
  import { tv, type VariantProps } from "tailwind-variants";

  // A status pill: the colour says state and nothing else (brand: status colours are never series colours).
  export const badgeVariants = tv({
    base: "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[13px] leading-[18px] font-medium whitespace-nowrap before:text-[11px] before:leading-none",
    variants: {
      variant: {
        default: "border-border bg-secondary text-text-2",
        success: "border-transparent bg-success-bg text-foreground before:text-success before:content-['✓']",
        info: "border-transparent bg-info-bg text-foreground before:text-info before:content-['●']",
        warning: "border-transparent bg-warning-bg text-foreground before:text-warning before:content-['▲']",
        error: "border-transparent bg-error-bg text-foreground before:text-error before:content-['✕']",
        queued: "border-border bg-secondary text-text-2 before:content-['○']",
        cancelled: "border-border bg-secondary text-text-2 before:content-['–']",
      },
    },
    defaultVariants: { variant: "default" },
  });

  export type BadgeVariant = VariantProps<typeof badgeVariants>["variant"];

  /** The badge for a query state of `system.queries`. */
  export function stateVariant(state: string): BadgeVariant {
    switch (state) {
      case "finished":
        return "success";
      case "running":
        return "info";
      case "failed":
        return "error";
      case "queued":
        return "queued";
      case "cancelled":
        return "cancelled";
      default:
        return "default";
    }
  }
</script>

<script lang="ts">
  import type { HTMLAttributes } from "svelte/elements";
  import { cn } from "$lib/utils";

  let { class: className, variant = "default", children, ...rest }: HTMLAttributes<HTMLSpanElement> & { variant?: BadgeVariant } = $props();
</script>

<span class={cn(badgeVariants({ variant }), className)} {...rest}>{@render children?.()}</span>
