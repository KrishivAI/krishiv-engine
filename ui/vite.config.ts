import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";

// Static SPA (ADR 0024): no SSR, no Node at runtime. Relative base so the
// assets resolve wherever the embedded server mounts them. Tailwind compiles to
// one static stylesheet at build time: nothing injects style at run time.
export default defineConfig({
  base: "./",
  plugins: [tailwindcss(), svelte()],
  resolve: { alias: { $lib: new URL("./src/lib", import.meta.url).pathname } },
  server: { proxy: { "/api": "http://127.0.0.1:8080" } },
  build: { outDir: "dist", emptyOutDir: true },
});
