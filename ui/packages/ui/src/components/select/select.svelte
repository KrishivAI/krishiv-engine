<script lang="ts" module>
  /** One choice of a select. */
  export interface SelectItem {
    value: string;
    label: string;
    disabled?: boolean;
  }
</script>

<script lang="ts">
  import { Select as SelectPrimitive } from "bits-ui";
  import { fieldClass } from "../input";
  import { cn } from "../../utils";

  // A single-choice select on Bits UI (a listbox in a floating layer, positioned through CSSOM, so it works under
  // `style-src 'self'`). Inside a Field, spread the field's control attributes on it: the trigger takes the id the
  // label points at, and the description and invalid state.
  let {
    items,
    value = $bindable(""),
    placeholder = "Choose…",
    id,
    name,
    disabled = false,
    class: className,
    onValueChange,
    "aria-describedby": describedby,
    "aria-invalid": invalid,
    "aria-label": ariaLabel,
  }: {
    items: SelectItem[];
    value?: string;
    placeholder?: string;
    id?: string;
    name?: string;
    disabled?: boolean;
    class?: string;
    onValueChange?: (value: string) => void;
    "aria-describedby"?: string;
    "aria-invalid"?: "true" | "false" | boolean;
    "aria-label"?: string;
  } = $props();

  const chosen = $derived(items.find((i) => i.value === value)?.label);
</script>

<SelectPrimitive.Root type="single" bind:value {items} {name} {disabled} {onValueChange}>
  <SelectPrimitive.Trigger
    {id}
    aria-describedby={describedby}
    aria-invalid={invalid}
    aria-label={ariaLabel}
    class={cn(
      fieldClass,
      "inline-flex w-full cursor-pointer items-center justify-between gap-2 text-left disabled:cursor-not-allowed disabled:opacity-45 aria-invalid:border-2 aria-invalid:border-error",
      className,
    )}
  >
    <span class={cn("truncate", !chosen && "text-muted-foreground")}>{chosen ?? placeholder}</span>
    <i class="text-muted-foreground not-italic" aria-hidden="true">▾</i>
  </SelectPrimitive.Trigger>
  <SelectPrimitive.Portal>
    <SelectPrimitive.Content
      sideOffset={4}
      class="z-50 min-w-(--bits-select-anchor-width) overflow-hidden rounded-lg border border-border-strong bg-popover p-1.5 text-popover-foreground shadow-pop"
    >
      <!-- The viewport holds the height cap (19 rem, or less where the page has less room), so a long list scrolls
           inside the layer instead of being clipped by it. -->
      <SelectPrimitive.Viewport class="max-h-[min(19rem,var(--bits-select-content-available-height,19rem))] overflow-y-auto">
        {#each items as item (item.value)}
          <SelectPrimitive.Item
            value={item.value}
            label={item.label}
            disabled={item.disabled}
            aria-disabled={item.disabled ? "true" : undefined}
            class="flex min-h-9 cursor-pointer items-center gap-2 rounded-md px-2.5 text-sm text-text-2 select-none data-disabled:cursor-not-allowed data-disabled:opacity-45 data-highlighted:bg-accent data-highlighted:text-foreground data-selected:text-foreground pointer-coarse:min-h-11"
          >
            {#snippet children({ selected })}
              <i class="w-4 text-primary not-italic" aria-hidden="true">{selected ? "✓" : ""}</i>
              {item.label}
            {/snippet}
          </SelectPrimitive.Item>
        {/each}
      </SelectPrimitive.Viewport>
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>
</SelectPrimitive.Root>
