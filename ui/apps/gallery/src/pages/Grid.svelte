<script lang="ts">
  import { Badge } from "@krishiv/ui/components/badge";
  import { GridShell, type GridColumn, type GridSort } from "@krishiv/ui/components/grid";
  import Section from "./Section.svelte";

  interface Compute {
    name: string;
    state: string;
    size: string;
    queries: number | null;
    month: string;
  }
  const COMPUTES: Compute[] = [
    { name: "default", state: "Running", size: "XS", queries: 1, month: "USD 18.40" },
    { name: "etl-m", state: "Starting", size: "M", queries: 0, month: "USD 212.75" },
    { name: "bi", state: "Stopped", size: "S", queries: null, month: "USD 41.02" },
  ];
  const COLUMNS: GridColumn<Compute>[] = [
    { id: "name", header: "Compute", value: (r) => r.name },
    { id: "state", header: "State", value: (r) => r.state, sortable: false },
    { id: "size", header: "Size", value: (r) => r.size },
    { id: "queries", header: "Queries now", num: true, value: (r) => r.queries },
    { id: "month", header: "This month", num: true, value: (r) => r.month },
  ];
  const VARIANT: Record<string, "success" | "warning" | "default"> = { Running: "success", Starting: "warning" };
  let selected = $state("etl-m");
  let sort: GridSort = $state(null);
  let reading = $state<string | undefined>(undefined);

  // A result grid: 10,000 rows, the column type under each name; only the visible window is in the page.
  interface Reading {
    id: number;
    device: string;
    temp: number | null;
  }
  const READINGS: Reading[] = Array.from({ length: 10000 }, (_, i) => ({ id: i + 1, device: `dev-${(i * 7919) % 997}`, temp: i % 13 === 0 ? null : Math.round(((i * 37) % 400) - 100) / 10 }));
  const RESULT: GridColumn<Reading>[] = [
    { id: "id", header: "id", type: "bigint", num: true, value: (r) => r.id },
    { id: "device", header: "device", type: "string", value: (r) => r.device },
    { id: "temp", header: "temp", type: "double", num: true, value: (r) => r.temp },
  ];
  // A list grid that scrolls but is not virtual (50 rows, under VIRTUAL_AFTER).
  const JOBS = Array.from({ length: 50 }, (_, i) => ({ id: `job-${i + 1}`, runs: (i * 13) % 40 }));
  const JOB_COLUMNS: GridColumn<(typeof JOBS)[number]>[] = [
    { id: "id", header: "Job", value: (r) => r.id },
    { id: "runs", header: "Runs", num: true, value: (r) => r.runs },
  ];
  let job = $state<string | undefined>(undefined);
</script>

<h1>Grid shell</h1>
<p>Sticky header, numbers right-aligned in mono, null in italics, the selected row tinted with a gold edge. Past 200 rows it draws only the rows in view.</p>
<Section title="List grid">
  <GridShell class="w-full" label="Computes" columns={COLUMNS} rows={COMPUTES} rowId={(r) => r.name} {selected} onselect={(id) => (selected = id)} bind:sort>
    {#snippet cell(r, c)}
      {#if c.id === "state"}<Badge variant={VARIANT[r.state] ?? "default"}>{r.state}</Badge>
      {:else if c.id === "name"}<a href={`#/grid?compute=${r.name}`}>{r.name}</a>
      {:else if r.queries === null && c.id === "queries"}<span class="text-muted-foreground italic">null</span>
      {:else}{c.value(r)}{/if}
    {/snippet}
    {#snippet foot()}<span>3 computes</span>{/snippet}
  </GridShell>
</Section>
<Section title="Result grid, 10,000 rows">
  <GridShell class="w-full max-h-96" label="Readings" rowHeight={40} columns={RESULT} rows={READINGS} rowId={(r) => String(r.id)} selected={reading} onselect={(id) => (reading = id)}>
    {#snippet foot()}<span data-count>10,000 rows</span>{/snippet}
  </GridShell>
</Section>
<Section title="List grid, 50 rows (scrolls, not virtual)">
  <GridShell class="w-full max-h-72" label="Jobs" columns={JOB_COLUMNS} rows={JOBS} rowId={(r) => r.id} selected={job} onselect={(id) => (job = id)} />
</Section>
<Section title="Empty">
  <GridShell class="w-full" label="No rows" columns={RESULT} rows={[]} rowId={(r) => String(r.id)} selected={reading} onselect={(id) => (reading = id)}>
    {#snippet empty()}No readings in this range.{/snippet}
  </GridShell>
</Section>
