<script lang="ts">
  import { buttonVariants } from "@krishiv/ui/components/button";
  import { Menu, MenuContent, MenuGroup, MenuItem, MenuLabel, MenuSeparator, MenuTrigger } from "@krishiv/ui/components/menu";
  import Section from "./Section.svelte";

  let picked = $state("none");
</script>

<h1>Menu</h1>
<p>A dropdown of actions on one object. Arrow keys move, Enter picks, Escape closes and gives the focus back.</p>
<Section title="Query actions">
  <Menu>
    <MenuTrigger class={buttonVariants()}>Actions ▾</MenuTrigger>
    <MenuContent>
      <MenuGroup>
        <MenuLabel>Query q_7f3a91</MenuLabel>
        <MenuItem onSelect={() => (picked = "open")}>Open in SQL editor</MenuItem>
        <MenuItem onSelect={() => (picked = "copy")}>Copy link</MenuItem>
        <MenuItem disabled>Download results</MenuItem>
      </MenuGroup>
      <MenuSeparator />
      <MenuItem variant="danger" onSelect={() => (picked = "cancel")}>Cancel query</MenuItem>
    </MenuContent>
  </Menu>
  <p data-picked>Picked: <span class="mono">{picked}</span></p>
</Section>
