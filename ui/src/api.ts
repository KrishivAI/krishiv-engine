// The token comes from the URL the server printed (`/?token=...`) once: the page trades it for an HttpOnly,
// SameSite=Strict session cookie (`POST /api/internal/session`) and removes it from the address bar. No script keeps
// the token, no request carries it in an address (the event stream sends the cookie), and a request that changes
// something carries the page's own `X-Krishiv-Csrf` header, which a page on another origin cannot add.

/** Opens the session from the token in the address, if any, and says whether the server accepts this browser. */
export async function initSession(): Promise<boolean> {
  const url = new URL(window.location.href);
  const fromUrl = url.searchParams.get("token");
  if (fromUrl) {
    url.searchParams.delete("token");
    history.replaceState(null, "", url.pathname + url.search + url.hash);
    const res = await fetch("/api/internal/session", {
      method: "POST",
      headers: { Authorization: `Bearer ${fromUrl}`, "X-Krishiv-Csrf": "1" },
    });
    return res.ok;
  }
  const res = await fetch("/api/internal/health", { headers: { "X-Krishiv-Csrf": "1" } });
  return res.ok;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, { ...init, headers: { ...(init.headers as Record<string, string>), "X-Krishiv-Csrf": "1" } });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(res.status, body.error ?? "error", body.message ?? `HTTP ${res.status}`);
  }
  return body as T;
}

export interface Query {
  query_id: string;
  state: string;
  principal?: string;
  statement_class?: string | null;
  query_text?: string | null;
  query_text_truncated?: boolean;
  submitted_at?: number;
  ended_at?: number | null;
  wall_ns?: number | null;
  cpu_ns?: number | null;
  peak_memory_bytes?: number | null;
  rows_produced?: number | null;
  bytes_read?: number | null;
  bytes_written?: number | null;
  spill_bytes?: number | null;
  error_class?: string | null;
  cancel_reason?: string | null;
  rows_read?: number | null;
  workers?: number | null;
  hints?: AdvisorHint[] | null;
}

/** One advisor hint (the same code, node, operator and message as EXPLAIN ANALYZE and system.queries). */
export interface AdvisorHint {
  code: string;
  node_id: number | null;
  operator_id: number | null;
  message: string;
}

export interface Operator {
  operator_id: number;
  name: string;
  role?: string;
  drivers?: number;
  rows_in?: number;
  rows_out?: number;
  compute_ns?: number;
  blocked_io_ns?: number;
  blocked_memory_ns?: number;
  blocked_sink_ns?: number;
  spilled_bytes?: number;
}

export interface QueryDetail {
  query: Query;
  operators: Operator[];
  hot_operator: { operator_id: number; operator: string; compute_ns: number } | null;
}

export interface CacheRow {
  worker_id?: string;
  cache: string;
  entries: number;
  bytes: number;
  capacity_bytes?: number | null;
  hits: number;
  misses: number;
  hit_rate: number | null;
}

export interface Health {
  version: string;
  uptime_seconds: number;
  workers: number;
  running_queries: number;
  queued_queries: number;
  memory: { reserved_bytes: number; peak_bytes: number; limit_bytes: number | null } | null;
  spill_bytes_total: number;
  caches: CacheRow[];
}

export interface MemoryRow {
  worker_id?: string;
  path: string;
  parent: string | null;
  kind: string;
  query_id?: string | null;
  reserved_bytes: number;
  peak_bytes: number;
  limit_bytes: number | null;
  spillable: boolean;
}

export interface Compare {
  a: Query;
  b: Query;
  delta: Record<string, number>;
}

export interface CatalogItem {
  name: string;
  data_type?: string;
  comment?: string;
  is_temporary?: boolean;
}

export interface Setting {
  key: string;
  value: string;
  redacted: boolean;
}

export interface Environment {
  version: string;
  settings: Setting[];
}

export interface SqlRun {
  state: "running" | "finished" | "failed" | "cancelled";
  columns?: string[];
  rows?: unknown[][];
  truncated?: boolean;
  error?: string;
  error_class?: string;
}
