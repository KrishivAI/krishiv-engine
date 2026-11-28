<script lang="ts">
  import type { Snippet } from "svelte";
  import type { HTMLAttributes } from "svelte/elements";
  import { cn } from "../../utils";
  import { Button } from "../button";

  // An error says what happened, what to do, and carries a support id. The error class and SQLSTATE are the
  // engine's (its registry), shown as they came: never rewritten here.
  let {
    class: className,
    title,
    errorClass,
    sqlstate,
    message,
    supportId,
    copyLabel = "Copy support id",
    copiedLabel = "Copied",
    children,
    actions,
    ...rest
  }: HTMLAttributes<HTMLDivElement> & {
    title: string;
    errorClass?: string;
    sqlstate?: string;
    message?: string;
    supportId?: string;
    /** The copy button's text (its accessible name) and its confirmation; a console passes its own catalog's text. */
    copyLabel?: string;
    copiedLabel?: string;
    actions?: Snippet;
  } = $props();

  // console-00 §3, §4: the support id is copyable. "Copied" is shown and announced (aria-live) for 2 s, and only
  // for the id that was written: a new support id (another error) within the 2 s, or while a write is still in
  // flight, shows the copy button again.
  let copiedId: string | undefined = $state();
  const copied = $derived(copiedId !== undefined && copiedId === supportId);
  let timer: ReturnType<typeof setTimeout> | undefined;
  async function copy() {
    const id = supportId;
    if (!id) return;
    try {
      await navigator.clipboard.writeText(id);
    } catch {
      return; // no clipboard (an insecure origin, or the permission refused): the id stays selectable text
    }
    copiedId = id;
    clearTimeout(timer);
    timer = setTimeout(() => (copiedId = undefined), 2000);
  }
  $effect(() => () => clearTimeout(timer));
</script>

<!-- The alert (an assertive, atomic live region) holds only what happened; the support id row is its sibling, so
     the copy button's "Copied" does not make a screen reader read the whole error again. -->
<div class={cn("grid max-w-[72ch] gap-2.5 rounded-xl border border-error bg-error-bg p-4", className)} {...rest}>
  <div role="alert" class="grid gap-2.5">
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
  </div>
  {#if actions || supportId}
    <div class="flex flex-wrap items-center gap-x-4 gap-y-2">
      {@render actions?.()}
      {#if supportId}
        <span class="inline-flex items-center gap-2 text-muted-foreground">
          Support id <code class="text-foreground select-all">{supportId}</code>
          <Button variant="outline" onclick={copy}>{copied ? copiedLabel : copyLabel}</Button>
          <span class="sr-only" aria-live="polite">{copied ? copiedLabel : ""}</span>
        </span>
      {/if}
    </div>
  {/if}
</div>
