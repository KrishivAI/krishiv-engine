// SPDX-License-Identifier: Apache-2.0
import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";

// The gallery of @krishiv/ui (dev only): built to its own dist/, served by ui/tests/check_gallery.py with the
// console's production CSP. Never copied into the engine's assets.
export default defineConfig({
  base: "./",
  plugins: [tailwindcss(), svelte()],
  build: { outDir: "dist", emptyOutDir: true },
});
