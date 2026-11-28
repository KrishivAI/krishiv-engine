<script lang="ts">
  import { Badge, stateVariant } from "@krishiv/ui/components/badge";
  import Section from "./Section.svelte";

  const STATES = ["running", "queued", "finished", "failed", "cancelled", "unknown-state"];
</script>

<h1>Badge</h1>
<p>A status pill: the colour says the state, and the word is always there.</p>
<Section title="Variants">
  <Badge>Default</Badge>
  <Badge variant="success">Succeeded</Badge>
  <Badge variant="info">Running</Badge>
  <Badge variant="warning">Starting</Badge>
  <Badge variant="error">Failed</Badge>
  <Badge variant="queued">Queued</Badge>
  <Badge variant="cancelled">Cancelled</Badge>
</Section>
<Section title="Query states">
  {#each STATES as s (s)}<Badge variant={stateVariant(s)}>{s}</Badge>{/each}
</Section>
