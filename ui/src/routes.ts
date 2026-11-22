// The UI's deep links (P3-T071). `routes.json` (next to `package.json`) is the versioned manifest of these paths; a
// Rust test in krishiv-server checks that every manifest path is served and appears here. A link is a plain path
// (`/queries/{id}`, `/queries/{id}/profile`); the older `#/queries/{id}` form keeps working.

export type Route = { page: string; id?: string; sub?: string };

/** Route patterns, `{id}` standing for one path segment. Keep in step with routes.json. */
export const ROUTES = [
  "/health",
  "/queries",
  "/queries/{id}",
  "/queries/{id}/profile",
  "/history",
  "/sessions",
  "/workload",
  "/resources",
  "/catalog",
  "/sql",
  "/environment",
] as const;

/** The route of a path (`/queries/q1/profile`) or of a hash (`#/queries/q1`); an explicit hash (a navigation inside the page) wins over the path the page was opened at. */
export function parseRoute(pathname: string, hash: string): Route {
  const fromPath = pathname.replace(/^\/+|\/+$/g, "");
  const fromHash = hash.replace(/^#\/?/, "");
  const raw = fromHash !== "" ? fromHash : fromPath;
  const parts = raw.split("/").filter((p) => p !== "");
  const safe = (s: string | undefined) => {
    if (s === undefined) return undefined;
    try {
      return decodeURIComponent(s);
    } catch {
      return s;
    }
  };
  return { page: parts[0] ?? "health", id: safe(parts[1]), sub: parts[2] };
}
