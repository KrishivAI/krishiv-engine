// Which console this is. The same pages are served in three modes (`mode` of `/api/internal/health`):
// `server` (every session of a server), `embedded` (the one session of the process that started the console) and
// `profiles` (stored profiles, no engine). The page hides what its mode does not have; the backend refuses those
// requests by itself, so nothing here is a permission check.
import { api, type Health } from "./api";

export type Mode = "server" | "embedded" | "profiles";

const ALL = ["health", "queries", "history", "resources", "catalog", "sql", "environment"];
const PAGES: Record<Mode, string[]> = {
  // Every client's sessions, the groups queries are admitted through and the advice folded from every
  // principal's queries are a server's.
  server: ["health", "queries", "history", "sessions", "workload", "advisor", "resources", "catalog", "sql", "environment"],
  embedded: ALL,
  // Stored profiles: nothing runs, so there is no live list, no memory, no catalog, no editor, no settings.
  profiles: ["history"],
};

// A server's settings are the administrator's: the route answers 403 to any other role.
const ADMIN_ONLY = ["environment"];

export const consoleInfo = $state<{ mode: Mode; sessionId: string | null; role: string | null }>({
  mode: "server",
  sessionId: null,
  role: null,
});

/** Reads the mode once, after the session is open. A server that reports none is a `server` console. */
export async function loadConsole(): Promise<void> {
  const h = await api<Health>("/api/internal/health");
  consoleInfo.mode = h.mode !== undefined && h.mode in PAGES ? h.mode : "server";
  consoleInfo.sessionId = h.session_id ?? null;
  consoleInfo.role = h.role ?? null;
}

/**
 * The pages this mode has for a caller of `role`, in menu order; the first is where the console opens. A server
 * that names no role decides on each request, so every page of the mode stays in the menu.
 */
export function pagesOf(mode: Mode, role: string | null = null): string[] {
  const all = PAGES[mode];
  return role === null || role === "admin" ? all : all.filter((p) => !ADMIN_ONLY.includes(p));
}
