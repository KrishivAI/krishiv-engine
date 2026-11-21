<script lang="ts">
  import { api, type CatalogItem } from "../api";

  type Level = { title: string; key: string; items: CatalogItem[]; selected: string | null };
  let catalogs = $state<CatalogItem[]>([]);
  let schemas = $state<CatalogItem[]>([]);
  let tables = $state<CatalogItem[]>([]);
  let columns = $state<CatalogItem[]>([]);
  let catalog = $state<string | null>(null);
  let schema = $state<string | null>(null);
  let table = $state<string | null>(null);
  let message = $state("");

  const enc = encodeURIComponent;
  // A name as a quoted SQL identifier: a backtick inside is doubled, so a hostile name cannot end the quoting.
  const quoteName = (n: string | null) => `\`${(n ?? "").replaceAll("`", "``")}\``;

  async function load<T>(path: string, key: string): Promise<T[]> {
    try {
      message = "";
      return ((await api<Record<string, T[]>>(path))[key]) ?? [];
    } catch (e) {
      message = (e as Error).message;
      return [];
    }
  }

  $effect(() => {
    load<CatalogItem>("/api/internal/catalog/catalogs", "catalogs").then((c) => (catalogs = c));
  });

  // Each pick bumps the generation: an answer for an earlier selection is dropped.
  let generation = 0;
  async function pickCatalog(name: string) {
    const g = ++generation;
    catalog = name;
    schema = table = null;
    schemas = tables = columns = [];
    const r = await load<CatalogItem>("/api/internal/catalog/" + enc(name) + "/schemas", "schemas");
    if (g === generation) schemas = r;
  }
  async function pickSchema(name: string) {
    const g = ++generation;
    schema = name;
    table = null;
    tables = columns = [];
    const r = await load<CatalogItem>(`/api/internal/catalog/${enc(catalog ?? "")}/${enc(name)}/tables`, "tables");
    if (g === generation) tables = r;
  }
  async function pickTable(name: string) {
    const g = ++generation;
    table = name;
    columns = [];
    const r = await load<CatalogItem>(
      `/api/internal/catalog/${enc(catalog ?? "")}/${enc(schema ?? "")}/${enc(name)}/columns`,
      "columns",
    );
    if (g === generation) columns = r;
  }
  const levels = $derived<Level[]>([
    { title: "Catalogs", key: "catalog", items: catalogs, selected: catalog },
    { title: "Schemas", key: "schema", items: schemas, selected: schema },
    { title: "Tables", key: "table", items: tables, selected: table },
  ]);
  function pick(key: string, name: string) {
    if (key === "catalog") return pickCatalog(name);
    if (key === "schema") return pickSchema(name);
    return pickTable(name);
  }
</script>

<h1 class="mb-4">Catalog</h1>
{#if message}<p class="mb-3 text-destructive wrap-anywhere">{message}</p>{/if}
<div class="grid gap-4 md:grid-cols-3">
  {#each levels as lv (lv.key)}
    <section class="min-w-0" aria-label={lv.title}>
      <h2 class="mb-2 text-base font-bold">{lv.title}</h2>
      <ul class="m-0 list-none p-0">
        {#each lv.items as it (it.name)}
          <li>
            <button
              type="button"
              class="min-h-11 w-full rounded-md px-3 py-2 text-left wrap-anywhere hover:bg-accent {lv.selected === it.name ? 'bg-accent font-medium' : ''}"
              aria-pressed={lv.selected === it.name}
              onclick={() => pick(lv.key, it.name)}>{it.name}</button
            >
          </li>
        {:else}
          <li class="px-3 py-2 text-muted-foreground">{lv.key === "catalog" ? "No catalog." : "Pick the one before."}</li>
        {/each}
      </ul>
    </section>
  {/each}
</div>
{#if table}
  <h2 class="mt-6 mb-2 text-base font-bold">Columns of <span class="mono">{table}</span></h2>
  <div data-table-box class="overflow-x-auto">
    <table class="w-full text-left">
      <thead>
        <tr><th class="py-2 pr-4">Column</th><th class="py-2 pr-4">Type</th><th class="py-2">Comment</th></tr>
      </thead>
      <tbody>
        {#each columns as c (c.name)}
          <tr><td class="mono py-1 pr-4">{c.name}</td><td class="mono py-1 pr-4">{c.data_type ?? ""}</td><td class="py-1 wrap-anywhere">{c.comment ?? ""}</td></tr>
        {/each}
      </tbody>
    </table>
  </div>
  <p class="mt-3"><a class="underline underline-offset-4" href="#/sql" onclick={() => sessionStorage.setItem("krishiv_sql_draft", `SELECT * FROM ${[catalog, schema, table].map(quoteName).join(".")} LIMIT 100`)}>Open in the SQL editor</a></p>
{/if}
