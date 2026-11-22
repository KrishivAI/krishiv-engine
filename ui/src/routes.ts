// The UI's deep links (P3-T071). `routes.json` (next to `package.json`) is the versioned manifest of these paths; a
// Rust test in krishiv-server checks that every manifest path is served and appears here. A link is a plain path
// (`/queries/{id}`, `/queries/{id}/profile`); the older `#/queries/{id}` form keeps working. A query's link may
// name its session (`?session=<id>`): two sessions can run a query under one id, and the lists link the row's.

export type Route = { page: string; id?: string; sub?: string; session?: string };

/** Route patterns, `{id}` standing for one path segment. Keep in step with routes.json. */
export const ROUTES = [
  "/health",
  "/queries",
  "/queries/{id}",
  "/queries/{id}/profile",
  "/history",
  "/sessions",
  "/workload",
  "/advisor",
  "/resources",
  "/catalog",
  "/sql",
  "/environment",
] as const;

/** The route of a path (`/queries/q1/profile`) or of a hash (`#/queries/q1`); an explicit hash (a navigation inside the page) wins over the path the page was opened at. */
export function parseRoute(pathname: string, hash: string, search = ""): Route {
  const fromPath = pathname.replace(/^\/+|\/+$/g, "");
  const [hashPath, hashQuery = ""] = hash.replace(/^#\/?/, "").split("?", 2);
  // The parameters travel with the address that names the page: the hash's own, or the path's.
  const [raw, query] = hashPath !== "" ? [hashPath, hashQuery] : [fromPath, search.replace(/^\?/, "")];
  const session = new URLSearchParams(query).get("session") || undefined;
  const parts = raw.split("/").filter((p) => p !== "");
  const safe = (s: string | undefined) => {
    if (s === undefined) return undefined;
    try {
      return decodeURIComponent(s);
    } catch {
      return s;
    }
  };
  return { page: parts[0] ?? "health", id: safe(parts[1]), sub: parts[2], session };
}

/** The address of a query's page inside the console, naming the query's session when the row carries one. */
export function queryHref(id: string, session?: string | null): string {
  const at = session ? `?session=${encodeURIComponent(session)}` : "";
  return `#/queries/${encodeURIComponent(id)}${at}`;
}
