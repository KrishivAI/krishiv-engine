<script lang="ts">
  import type { Snippet } from "svelte";
  import type { HTMLInputAttributes } from "svelte/elements";
  import { fieldClass } from "../input";
  import { fieldControl, fieldIds, type FieldControl } from "./ids";
  import { cn } from "../../utils";

  // Label above, help below, the error in words with an icon (the wireframe's k-field). Without `control` the
  // field draws a text input; with it, the caller draws its own control and spreads the attributes it is given
  // (`id` and the `aria-*` links). A `control` takes its own `name`, `required`, `autocomplete` and value: the
  // field's `value` and other input attributes apply only to the text input it draws.
  let {
    id,
    label,
    help,
    error,
    value = $bindable(""),
    class: className,
    control,
    "aria-describedby": describedby,
    "aria-invalid": invalid,
    ...rest
  }: Omit<HTMLInputAttributes, "id" | "value" | "aria-invalid"> & {
    id: string;
    label: string;
    help?: string;
    error?: string;
    value?: string;
    control?: Snippet<[FieldControl]>;
    "aria-invalid"?: "true" | "false" | boolean;
  } = $props();

  const ids = $derived(fieldIds(id, !!help, !!error));
  // The caller's own description ids join the field's; an error keeps the control invalid whatever the caller says.
  const attrs = $derived(fieldControl(id, !!help, !!error, { describedby: describedby ?? undefined, invalid }));
</script>

<div class={cn("grid max-w-xl gap-1.5", className)} data-field-error={error ? "" : undefined}>
  <label for={id} class="text-[13px] leading-[18px] font-semibold text-foreground">{label}</label>
  {#if control}
    {@render control(attrs)}
  {:else}
    <input class={cn(fieldClass, "w-full", error && "border-2 border-error")} bind:value {...rest} {...attrs} />
  {/if}
  {#if error}
    <span id={ids.error} class="flex items-start gap-1.5 text-[13px] leading-[18px] text-error">
      <i class="not-italic" aria-hidden="true">✕</i><span>{error}</span>
    </span>
  {/if}
  {#if help}
    <span id={ids.help} class="text-[13px] leading-[18px] text-muted-foreground">{help}</span>
  {/if}
</div>
