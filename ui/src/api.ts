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
  /** The session the query ran in; only the lists carry it, and a row stored before a restart has none. */
  session_id?: string | null;
  state: string;
  principal?: string;
  /** What the client of the query's session said it is; a server console only. */
  client_type?: string | null;
  /** The session's tags and the request's (a request tag has an empty value). */
  tags?: Record<string, string> | null;
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
  /** The plan node the operator runs (`plan[].id`); several operators can share one. */
  node_id?: number | null;
  name: string;
  role?: string;
  drivers?: number;
  rows_in?: number;
  rows_out?: number;
  compute_ns?: number;
  blocked_io_ns?: number;
  blocked_memory_ns?: number;
  blocked_sink_ns?: number;
  peak_reserved_bytes?: number;
  spilled_bytes?: number;
}

/**
 * One node of a stored plan (the profile's `plan`); `children` index the array. A heading has no id. The
 * server sends no other member: a node's detail text can hold the statement's literals.
 */
export interface PlanNode {
  id: number | null;
  name: string;
  children: number[];
}

export interface QueryDetail {
  query: Query;
  operators: Operator[];
  hot_operator: { operator_id: number; operator: string; compute_ns: number } | null;
  /** The plan tree, when the source keeps one for the query. */
  plan?: PlanNode[] | null;
  /** The query has ended and its plan is not stored yet but can still arrive: ask again. */
  plan_pending?: boolean;
  /** Another session ran a query under this id; operator rows name no session, so none are given. */
  operators_withheld?: boolean;
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
  /** Which console this is; absent from a server older than the modes. */
  mode?: "server" | "embedded" | "profiles";
  /** The one session an `embedded` console shows. */
  session_id?: string | null;
  /** The caller's role (`viewer`, `operator`, `admin`); absent or null when the server does not say. */
  role?: string | null;
  uptime_seconds: number;
  workers: number;
  running_queries: number;
  queued_queries: number;
  memory: { reserved_bytes: number; peak_bytes: number; limit_bytes: number | null } | null;
  spill_bytes_total: number;
  caches: CacheRow[];
}

/** One row of `system.sessions`; times are microseconds since the epoch. */
export interface Session {
  session_id: string;
  principal: string;
  client_user_id?: string | null;
  client_type?: string | null;
  transport: string;
  /** `active`, or how it ended: `closed`, `expired`, `killed`. */
  state: string;
  created_at: number;
  last_active_at: number;
  closed_at?: number | null;
  close_reason?: string | null;
  idle_timeout_ms?: number | null;
  current_catalog: string;
  current_namespace: string;
  tags?: Record<string, string> | null;
  queries_running: number;
  queries_started: number;
  delegated_by?: string | null;
  /** `false` on the session the console's editor runs in, which the console cannot end. */
  endable?: boolean | null;
}

/** One row of `system.resource_groups`; a group's counters include its subgroups'. */
export interface ResourceGroup {
  group_id: string;
  parent_id?: string | null;
  /** `active`, or `removed` while a group taken out of the configuration still holds queries. */
  state: string;
  scheduling_policy: string;
  scheduling_weight: number;
  priority: number;
  workload_class: string;
  hard_concurrency_limit: number;
  max_queued: number;
  soft_memory_limit_bytes?: number | null;
  per_query_memory_limit_bytes?: number | null;
  scan_quota_bytes?: number | null;
  scan_quota_period_s?: number | null;
  running: number;
  queued: number;
  memory_reserved_bytes: number;
  scan_bytes_in_window?: number | null;
  scan_window_resets_at?: number | null;
}

/** One partition-column suggestion for a table (`system.partition_advice`). */
export interface AdvisedColumn {
  /** 1 is the table's best suggestion. */
  rank: number;
  column_name: string;
  /** How many of the table's queries filtered on the column by equality. */
  column_queries: number;
  /** The column's distinct values: the partitions the table would have. */
  partitions: number;
  /** The share of the bytes the table's queries read that partitioning by the column is expected to skip. */
  pruned_fraction: number;
}

/** A table the query history has partition advice for; `columns` by rank. */
export interface TableAdvice {
  table_name: string;
  /** The finished queries in the history that read the table. */
  table_queries: number;
  columns: AdvisedColumn[];
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

/** The Python UDF workers of one principal and session. */
export interface UdfWorkerOwner {
  principal: string;
  session_id: string;
  tenant: string | null;
  /** Live workers: idle, running a call or starting. */
  live: number;
  /** Those of them that run a call or are starting. */
  busy: number;
}

/** A server's Python UDF worker pool (`/api/internal/udf-workers`); the owners with the most workers first. */
export interface UdfPool {
  live: number;
  busy: number;
  max_total: number;
  max_per_session: number;
  max_per_tenant: number;
  owners: UdfWorkerOwner[];
  truncated: boolean;
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
  /** The Spark name the setting also answers to, when it has one. */
  spark_key?: string | null;
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
