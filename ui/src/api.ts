// The token comes from the URL the server printed (`/?token=...`); it is kept for the tab and removed from the
// address bar. Requests send it as a Bearer header; the event stream, which cannot set headers, as `?token=`.

const KEY = "krishiv_ui_token";

export function initToken(): string | null {
  const url = new URL(window.location.href);
  const fromUrl = url.searchParams.get("token");
  if (fromUrl) {
    sessionStorage.setItem(KEY, fromUrl);
    url.searchParams.delete("token");
    history.replaceState(null, "", url.pathname + url.search + url.hash);
  }
  return sessionStorage.getItem(KEY);
}

export function token(): string | null {
  return sessionStorage.getItem(KEY);
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
  const t = token();
  const headers: Record<string, string> = t ? { Authorization: `Bearer ${t}` } : {};
  const res = await fetch(path, { ...init, headers });
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

export interface Warning {
  operator_id: number;
  operator: string;
  kind: "spill" | "skew" | "memory";
  hint: string;
}

export interface QueryDetail {
  query: Query;
  operators: Operator[];
  hot_operator: { operator_id: number; operator: string; compute_ns: number } | null;
  warnings: Warning[];
}

export interface CacheRow {
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
