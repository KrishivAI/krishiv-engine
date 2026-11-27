<script lang="ts">
  import type { Snippet } from "svelte";
  import type { HTMLAttributes } from "svelte/elements";
  import { cn } from "../../utils";

  // An error says what happened, what to do, and carries a support id. The error class and SQLSTATE are the
  // engine's (its registry), shown as they came: never rewritten here.
  let {
    class: className,
    title,
    errorClass,
    sqlstate,
    message,
    supportId,
    children,
    actions,
    ...rest
  }: HTMLAttributes<HTMLDivElement> & {
    title: string;
    errorClass?: string;
    sqlstate?: string;
    message?: string;
    supportId?: string;
    actions?: Snippet;
  } = $props();
</script>

<div role="alert" class={cn("grid max-w-[72ch] gap-2.5 rounded-xl border border-error bg-error-bg p-4", className)} {...rest}>
  <h3 class="flex items-center gap-2 text-base leading-6 font-bold">
    <i class="text-error not-italic" aria-hidden="true">✕</i>{title}
  </h3>
  {#if errorClass || message || sqlstate}
    <p class="mono text-text-2 wrap-anywhere">
      {#if errorClass}[{errorClass}]{/if}
      {message ?? ""}
      {#if sqlstate}SQLSTATE {sqlstate}{/if}
    </p>
  {/if}
  {#if children}<div class="text-text-2">{@render children()}</div>{/if}
  {#if actions || supportId}
    <div class="flex flex-wrap items-center gap-x-4 gap-y-2">
      {@render actions?.()}
      {#if supportId}<span class="text-muted-foreground">Support id <code class="text-foreground">{supportId}</code></span>{/if}
    </div>
  {/if}
</div>
