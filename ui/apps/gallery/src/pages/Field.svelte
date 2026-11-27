<script lang="ts">
  import { Field } from "@krishiv/ui/components/field";
  import { Select } from "@krishiv/ui/components/select";
  import Section from "./Section.svelte";

  let name = $state("etl-m");
  let idle = $state("0 minutes");
  let size = $state("xs");
  const idleError = $derived(/^(never|([5-9]|\d\d+) minutes)$/.test(idle) ? undefined : 'Choose 5 minutes or more, or "never" (you pay while it runs).');
</script>

<h1>Field</h1>
<p>Label above, help below, the error in words with an icon and a way out.</p>
<Section title="States">
  <Field id="f-name" label="Compute name" help="Lower case, digits and dashes. Shown in the compute picker." bind:value={name} />
  <Field id="f-idle" label="Stop after idle" bind:value={idle} error={idleError} />
  <Field id="f-off" label="Region" value="eu-west-1" help="Fixed when the compute was made." disabled />
  <Field id="f-size" label="Size" help="An hour costs the size's credits.">
    {#snippet control(attrs)}
      <Select
        {...attrs}
        bind:value={size}
        items={[
          { value: "xs", label: "XS, 4 vCPU, 16 GiB, 1 credit an hour" },
          { value: "s", label: "S, 8 vCPU, 32 GiB, 2 credits an hour" },
        ]}
      />
    {/snippet}
  </Field>
</Section>
