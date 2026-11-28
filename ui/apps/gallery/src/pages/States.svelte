<script lang="ts">
  import { Button } from "@krishiv/ui/components/button";
  import { EmptyState } from "@krishiv/ui/components/empty-state";
  import { ErrorState } from "@krishiv/ui/components/error-state";
  import { LoadingState } from "@krishiv/ui/components/loading-state";
  import Section from "./Section.svelte";
</script>

<h1>Empty, loading, error</h1>
<p>Empty says what will be here and how to make it appear; loading keeps the layout; an error says what happened and what to do.</p>
<Section title="Empty">
  <EmptyState title="No jobs yet">
    A job runs a notebook, a SQL file or a pipeline on a schedule.
    {#snippet actions()}<Button>Schedule a query</Button>{/snippet}
  </EmptyState>
  <EmptyState title="No queries match" centered>Clear a filter to see more.</EmptyState>
</Section>
<Section title="Loading">
  <LoadingState class="w-72" label="Loading the profile" />
  <LoadingState class="w-72" label="Reading" lines={2} stage="Reading samples.fleet.readings, 35%, 0:04" value={35} />
</Section>
<Section title="Error">
  <ErrorState title="The query failed" errorClass="DIVIDE_BY_ZERO" message="Division by zero." sqlstate="22012" supportId="req_8c1f2a90">
    Use <code>try_divide</code>, or set <code>spark.sql.ansi.enabled</code> to false for this session.
    {#snippet actions()}<Button>Open the line</Button>{/snippet}
  </ErrorState>
</Section>
