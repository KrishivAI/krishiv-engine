<script lang="ts">
  import { Button } from "@krishiv/ui/components/button";
  import { Spinner } from "@krishiv/ui/components/loading-state";
  import Section from "./Section.svelte";
</script>

<h1>Button</h1>
<p>One primary (gold) per region; outline for everything else, ghost in toolbars, destructive for actions that lose work.</p>
<Section title="Variants">
  <Button variant="default">Run</Button>
  <Button>Upload a file</Button>
  <Button variant="ghost">Schedule</Button>
  <Button variant="destructive">Cancel query</Button>
  <Button size="icon" aria-label="More actions">⋯</Button>
</Section>
<Section title="States">
  <Button disabled>Disabled</Button>
  <Button variant="default" disabled>Disabled primary</Button>
  <Button variant="default" aria-busy="true"><Spinner />Running</Button>
</Section>
