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
  let side = $state("general");
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
<section class="grid grid-cols-[minmax(0,1fr)] gap-3">
  <h2>Vertical</h2>
  <Tabs bind:value={side} orientation="vertical" class="flex min-w-0 gap-4 rounded-lg border border-border bg-background p-4">
    <TabsList aria-label="Settings">
      <TabsTrigger value="general">General</TabsTrigger>
      <TabsTrigger value="access">Access</TabsTrigger>
      <TabsTrigger value="billing">Billing</TabsTrigger>
    </TabsList>
    <TabsContent value="general" class="pt-0">General settings</TabsContent>
    <TabsContent value="access" class="pt-0">Access settings</TabsContent>
    <TabsContent value="billing" class="pt-0">Billing settings</TabsContent>
  </Tabs>
</section>
