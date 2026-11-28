<script lang="ts">
  import { Button } from "@krishiv/ui/components/button";
  import { Callout } from "@krishiv/ui/components/callout";
  import Section from "./Section.svelte";
</script>

<h1>Callout</h1>
<p>Callouts carry state: note, warning, error, success; a plain one for facts to know before acting.</p>
<Section title="Variants">
  <div class="grid w-full gap-3 lg:grid-cols-2">
    <Callout title="The compute starts on your first query.">About 20 seconds. You pay from then until it stops.</Callout>
    <Callout variant="warning" title="This will read about 22 GB.">
      On etl-m that is about 4 minutes.
      {#snippet actions()}<Button>Run anyway</Button><Button variant="ghost">Add a filter</Button>{/snippet}
    </Callout>
    <Callout variant="error" title="The spending cap stopped etl-m.">USD 500 of 500 used this month. An admin can raise the cap.</Callout>
    <Callout variant="success" title="Table created."><code>me.uploads.sensors_2026</code>, 1.2 M rows.</Callout>
    <Callout variant="plain" title="Results are kept for 24 hours." />
  </div>
</Section>
