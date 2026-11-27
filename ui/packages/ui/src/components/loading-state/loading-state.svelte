<script lang="ts">
  import type { HTMLAttributes } from "svelte/elements";
  import { cn } from "../../utils";
  import { skeletonWidths } from "./progress";

  // Loading keeps the layout (skeleton lines), and past a second shows a progress bar with the stage's name. The
  // bar is a native <progress>: its fill needs no style attribute.
  let {
    class: className,
    label = "Loading",
    lines = 4,
    stage,
    value,
    max = 100,
    ...rest
  }: HTMLAttributes<HTMLDivElement> & { label?: string; lines?: number; stage?: string; value?: number; max?: number } = $props();
</script>

<div class={cn("grid gap-2.5", className)} aria-busy="true" role="status" aria-label={label} {...rest}>
  {#each skeletonWidths(lines) as w, i (i)}
    <span class={cn("block h-3 animate-pulse rounded-md bg-secondary", w, i === 0 && "h-5")}></span>
  {/each}
  {#if stage !== undefined}
    <div class="grid gap-1.5">
      <progress
        class="h-1.5 w-full appearance-none overflow-hidden rounded-full bg-secondary [&::-moz-progress-bar]:bg-primary [&::-webkit-progress-bar]:bg-secondary [&::-webkit-progress-value]:bg-primary"
        {max}
        {value}
        aria-label={stage}
      ></progress>
      <p class="text-[13px] text-muted-foreground">{stage}</p>
    </div>
  {/if}
</div>
