<script lang="ts">
  import { buttonVariants } from "@krishiv/ui/components/button";
  import { Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogTitle, DialogTrigger } from "@krishiv/ui/components/dialog";
  import Section from "./Section.svelte";

  let open = $state(false);
  let outcome = $state("none");
</script>

<h1>Dialog</h1>
<p>For a decision that needs attention: the title is the question, the primary button repeats the verb, the safe choice has the first focus.</p>
<Section title="Confirm">
  <Dialog bind:open>
    <DialogTrigger class={buttonVariants({ variant: "destructive" })}>Cancel query…</DialogTrigger>
    <DialogContent>
      <DialogTitle>Cancel Arjun Rao's query?</DialogTitle>
      <DialogDescription>It has run for 14 minutes on etl-m. Arjun is told it was cancelled by you.</DialogDescription>
      <DialogFooter>
        <DialogClose class={buttonVariants()} onclick={() => (outcome = "kept")}>Keep running</DialogClose>
        <DialogClose class={buttonVariants({ variant: "destructive" })} onclick={() => (outcome = "cancelled")}>Cancel query</DialogClose>
      </DialogFooter>
    </DialogContent>
  </Dialog>
  <p data-outcome>Outcome: <span class="mono">{outcome}</span></p>
</Section>
