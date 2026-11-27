<script lang="ts" module>
  import { tv, type VariantProps } from "tailwind-variants";

  // A callout carries state (note, warning, error, success) or a plain fact to know before acting. Panels have no
  // shadow; the icon is a glyph and the title says it in words, so the colour is never the only signal.
  export const calloutVariants = tv({
    slots: {
      base: "grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 rounded-lg border px-4 py-3 text-sm",
      icon: "leading-5 not-italic",
    },
    variants: {
      variant: {
        note: { base: "border-transparent bg-info-bg", icon: "text-info" },
        warning: { base: "border-transparent bg-warning-bg", icon: "text-warning" },
        error: { base: "border-transparent bg-error-bg", icon: "text-error" },
        success: { base: "border-transparent bg-success-bg", icon: "text-success" },
        plain: { base: "border-border bg-muted", icon: "text-muted-foreground" },
      },
    },
    defaultVariants: { variant: "note" },
  });
  export type CalloutVariant = VariantProps<typeof calloutVariants>["variant"];

  /** The glyph of each variant (aria-hidden: the title carries the meaning). */
  export const calloutIcons: Record<NonNullable<CalloutVariant>, string> = {
    note: "ℹ",
    warning: "▲",
    error: "✕",
    success: "✓",
    plain: "•",
  };
</script>

<script lang="ts">
  import type { Snippet } from "svelte";
  import type { HTMLAttributes } from "svelte/elements";
  import { cn } from "../../utils";

  let {
    class: className,
    variant = "note",
    title,
    children,
    actions,
    ...rest
  }: HTMLAttributes<HTMLDivElement> & { variant?: CalloutVariant; title: string; actions?: Snippet } = $props();

  const s = $derived(calloutVariants({ variant }));
</script>

<div class={cn(s.base(), className)} {...rest}>
  <i class={s.icon()} aria-hidden="true">{calloutIcons[variant ?? "note"]}</i>
  <div class="min-w-0">
    <strong class="font-bold text-foreground">{title}</strong>
    {#if children}<div class="text-text-2 wrap-anywhere">{@render children()}</div>{/if}
  </div>
  {#if actions}<div class="col-start-2 mt-1.5 flex flex-wrap gap-2">{@render actions()}</div>{/if}
</div>
