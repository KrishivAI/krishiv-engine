// Which console this is. The same pages are served in three modes (`mode` of `/api/internal/health`):
// `server` (every session of a server), `embedded` (the one session of the process that started the console) and
// `profiles` (stored profiles, no engine). The page hides what its mode does not have; the backend refuses those
// requests by itself, so nothing here is a permission check.
import { api, type Health } from "./api";

export type Mode = "server" | "embedded" | "profiles";

const ALL = ["health", "queries", "history", "resources", "catalog", "sql", "environment"];
const PAGES: Record<Mode, string[]> = {
  server: ALL,
  embedded: ALL,
  // Stored profiles: nothing runs, so there is no live list, no memory, no catalog, no editor, no settings.
  profiles: ["history"],
};

export const consoleInfo = $state<{ mode: Mode; sessionId: string | null }>({ mode: "server", sessionId: null });

/** Reads the mode once, after the session is open. A server that reports none is a `server` console. */
export async function loadConsole(): Promise<void> {
  const h = await api<Health>("/api/internal/health");
  consoleInfo.mode = h.mode !== undefined && h.mode in PAGES ? h.mode : "server";
  consoleInfo.sessionId = h.session_id ?? null;
}

/** The pages this mode has, in menu order; the first is where the console opens. */
export function pagesOf(mode: Mode): string[] {
  return PAGES[mode];
}
