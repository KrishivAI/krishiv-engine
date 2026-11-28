<script lang="ts">
  import { Field } from "@krishiv/ui/components/field";
  import { Select } from "@krishiv/ui/components/select";
  import Section from "./Section.svelte";

  const SIZES = [
    { value: "xs", label: "XS, 4 vCPU, 16 GiB" },
    { value: "s", label: "S, 8 vCPU, 32 GiB" },
    { value: "m", label: "M, 16 vCPU, 64 GiB" },
    { value: "xl", label: "XL, 64 vCPU, 256 GiB", disabled: true },
  ];
  let size = $state("");
  let fixed = $state("s");
</script>

<h1>Select</h1>
<p>One choice from a short list, in a floating layer (Bits UI). The keyboard opens it with Enter, Space or the arrows.</p>
<Section title="In a field">
  <Field id="sel-size" label="Size" help="XL needs a quota raise.">
    {#snippet control(attrs)}<Select {...attrs} items={SIZES} bind:value={size} placeholder="Choose a size" />{/snippet}
  </Field>
  <p class="self-end" data-chosen>Chosen: <span class="mono">{size || "none"}</span></p>
</Section>
<Section title="States">
  <Field id="sel-err" label="Size" error="Choose a size the workspace allows.">
    {#snippet control(attrs)}<Select {...attrs} items={SIZES} bind:value={size} placeholder="Choose a size" />{/snippet}
  </Field>
  <Field id="sel-off" label="Size (fixed)">
    {#snippet control(attrs)}<Select {...attrs} items={SIZES} bind:value={fixed} disabled />{/snippet}
  </Field>
</Section>
