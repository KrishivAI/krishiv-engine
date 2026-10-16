import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Static SPA (ADR 0024): no SSR, no Node at runtime. Relative base so the
// assets resolve wherever the embedded server mounts them.
export default defineConfig({
  base: "./",
  plugins: [svelte()],
  server: { proxy: { "/api": "http://127.0.0.1:8080" } },
  build: { outDir: "dist", emptyOutDir: true },
});
