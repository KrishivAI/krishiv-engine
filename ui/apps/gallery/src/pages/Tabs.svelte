<script lang="ts">
  import { Tabs, TabsContent, TabsList, TabsTrigger } from "@krishiv/ui/components/tabs";

  const TABS = [
    ["overview", "Overview", ""],
    ["sessions", "Sessions", "4"],
    ["workload", "Workload", ""],
    ["advisor", "Advisor", "2"],
    ["logs", "Logs", ""],
    ["settings", "Settings", ""],
  ];
  let tab = $state("overview");
</script>

<h1>Tabs</h1>
<p>Tabs switch views of one object. The active tab is the gold element of its region; on a phone the row scrolls sideways.</p>
<section class="grid grid-cols-[minmax(0,1fr)] gap-3">
  <h2>Compute</h2>
  <Tabs bind:value={tab} class="min-w-0 rounded-lg border border-border bg-background p-4">
    <TabsList aria-label="Compute">
      {#each TABS as [value, label, count] (value)}
        <TabsTrigger {value}>{label}{#if count}<span class="font-mono text-xs text-muted-foreground">{count}</span>{/if}</TabsTrigger>
      {/each}
      <TabsTrigger value="billing" disabled>Billing</TabsTrigger>
    </TabsList>
    {#each TABS as [value, label] (value)}
      <TabsContent {value}>{label} panel</TabsContent>
    {/each}
    <TabsContent value="billing">Billing panel</TabsContent>
  </Tabs>
</section>
