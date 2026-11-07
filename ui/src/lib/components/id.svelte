<script lang="ts">
  // A query id is 36 characters. Its first characters are a timestamp (UUIDv7), so queries of the same minute
  // share them: show the last 8, which differ, and the whole id on hover (title) and on a tap.
  let { id }: { id: string } = $props();
  let open = $state(false);
</script>

<button
  class="mono inline-flex min-h-6 items-center bg-transparent px-0.5 text-left text-muted-foreground wrap-anywhere hover:text-foreground pointer-coarse:min-h-10 pointer-coarse:min-w-10"
  title={id}
  aria-label={`Query id ${id}`}
  onclick={() => (open = !open)}
>
  {open ? id : `…${id.slice(-8)}`}
</button>
