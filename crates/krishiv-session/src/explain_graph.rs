// SPDX-License-Identifier: Apache-2.0
//! Mermaid and Graphviz renderings of the physical plan and the profile DAG.
//!
//! [`GraphFormat`] selects the rendering. [`GraphFormat::Json`] reuses the
//! versioned JSON documents (`ExplainVerbose::to_json`,
//! [`QueryProfile::to_json`]); there is no second JSON schema.
//! [`GraphFormat::Mermaid`] and [`GraphFormat::Dot`] render the node tree as
//! text, by hand (no new dependency):
//!
//! - [`mermaid_of_plan`] and [`dot_of_plan`] render an [`ExplainNode`] tree,
//!   the physical plan [`DataFrame::explain_with`] graphs;
//! - [`mermaid_of_profile`] and [`dot_of_profile`] render a [`QueryProfile`]:
//!   the same tree with each node's operators and their counters, as
//!   [`DataFrame::explain_analyze_with`] graphs.
//!
//! Node ids (`n0`, `n1`, ...) are the rendered tree's pre-order numbers.
//! Edges run from a child to its parent, the direction data flows.
//! Labels escape `\`, `"`, newlines, `&`, `<`, `>` and `|`, so a label never
//! breaks out of its quotes. Operators no plan node holds (the result sink)
//! print under a `Result:` section of the root's label.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::time::Duration;

use krishiv_common::error::{InvalidParameterValueString, KrishivError};
use krishiv_exec::planner::ExplainNode;
use krishiv_exec::profile::{ProfileNode, QueryProfile};
use krishiv_exec::{Metric, OperatorEntry, OperatorId};

use crate::dataframe::{DataFrame, ExplainMode};

/// How to render an explained plan or profile.
///
/// [`DataFrame::explain_with`], [`DataFrame::explain_verbose_with`] and
/// [`DataFrame::explain_analyze_with`] take one of these; `Json` reuses the
/// versioned JSON documents, `Mermaid` and `Dot` render the node tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphFormat {
    /// The versioned JSON document (`ExplainVerbose::to_json` or
    /// [`QueryProfile::to_json`], experimental as those are). A plan mode
    /// with no JSON document ([`DataFrame::explain_with`]) rejects it with
    /// `INVALID_PARAMETER_VALUE`.
    Json,
    /// Mermaid `flowchart TD` text: one `n{id}["label"]` per node, one
    /// `child --> parent` edge per input.
    Mermaid,
    /// Graphviz `digraph` text: one `n{id} [label="..."];` per node, one
    /// `child -> parent` edge per input.
    Dot,
}

impl GraphFormat {
    /// `json`, `mermaid` or `dot`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Mermaid => "mermaid",
            Self::Dot => "dot",
        }
    }

    /// Parses `text` (ASCII case-insensitive; surrounding whitespace is
    /// ignored).
    ///
    /// ```
    /// use krishiv_session::GraphFormat;
    /// assert_eq!(GraphFormat::parse("mermaid"), Ok(GraphFormat::Mermaid));
    /// assert_eq!(GraphFormat::parse(" DOT "), Ok(GraphFormat::Dot));
    /// assert!(GraphFormat::parse("yaml").is_err());
    /// ```
    ///
    /// # Errors
    /// `INVALID_PARAMETER_VALUE.STRING` naming the allowed values for an
    /// unknown format.
    pub fn parse(text: &str) -> Result<Self, KrishivError> {
        match text.trim().to_ascii_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "mermaid" => Ok(Self::Mermaid),
            "dot" => Ok(Self::Dot),
            _ => Err(InvalidParameterValueString {
                parameter: "format".to_owned(),
                function_name: "DataFrame.explain_with".to_owned(),
                invalid_value: format!("{text} (expected one of: json, mermaid, dot)"),
            }
            .into()),
        }
    }
}

/// The `format` of [`DataFrame::explain_with`]'s error for a plan mode with
/// no JSON document: JSON exists for `EXPLAIN VERBOSE` and `EXPLAIN ANALYZE`
/// only (as [`DataFrame::explain`] has no JSON form either).
pub(crate) fn no_json_for_mode(mode: ExplainMode) -> KrishivError {
    let name = match mode {
        ExplainMode::Simple => "simple",
        ExplainMode::Extended => "extended",
        ExplainMode::Formatted => "formatted",
        ExplainMode::Cost => "cost",
        ExplainMode::Codegen => "codegen",
    };
    InvalidParameterValueString {
        parameter: "format".to_owned(),
        function_name: "DataFrame.explain_with".to_owned(),
        invalid_value: format!(
            "json (explain mode {name} has no JSON form; mermaid and dot are available, \
             JSON exists for EXPLAIN VERBOSE and EXPLAIN ANALYZE)"
        ),
    }
    .into()
}

/// The physical plan as Mermaid `flowchart TD` text.
///
/// ```
/// use krishiv_exec::planner::ExplainNode;
/// use krishiv_session::mermaid_of_plan;
///
/// let plan = ExplainNode::new(
///     "Sort",
///     "x#0 ASC NULLS FIRST",
///     vec![ExplainNode::leaf("Scan", "t")],
/// );
/// let text = mermaid_of_plan(&plan);
/// assert!(text.starts_with("flowchart TD\n"));
/// assert!(text.contains("n1 --> n0"));
/// ```
#[must_use]
pub fn mermaid_of_plan(root: &ExplainNode) -> String {
    emit_mermaid(&of_plan(root))
}

/// The physical plan as Graphviz `digraph` text.
///
/// ```
/// use krishiv_exec::planner::ExplainNode;
/// use krishiv_session::dot_of_plan;
///
/// let plan = ExplainNode::leaf("Scan", "t");
/// assert_eq!(dot_of_plan(&plan), "digraph {\n    n0 [label=\"Scan t\"];\n}\n");
/// ```
#[must_use]
pub fn dot_of_plan(root: &ExplainNode) -> String {
    emit_dot(&of_plan(root))
}

/// The profile ([`QueryProfile`]) as Mermaid `flowchart TD` text: the plan
/// tree with each node's operators and their counters.
///
/// ```
/// use krishiv_exec::profile::{ProfileNode, QueryProfile};
/// use krishiv_exec::QueryMetrics;
/// use krishiv_session::mermaid_of_profile;
///
/// let profile = QueryProfile {
///     plan: ProfileNode {
///         id: Some(0),
///         name: "Scan".to_owned(),
///         detail: "t".to_owned(),
///         children: Vec::new(),
///     },
///     metrics: QueryMetrics::default(),
///     detailed: false,
/// };
/// let text = mermaid_of_profile(&profile);
/// assert!(text.starts_with("flowchart TD\n"));
/// assert!(text.contains("Scan t"));
/// ```
#[must_use]
pub fn mermaid_of_profile(profile: &QueryProfile) -> String {
    emit_mermaid(&of_profile(profile))
}

/// The profile ([`QueryProfile`]) as Graphviz `digraph` text: the plan tree
/// with each node's operators and their counters.
#[must_use]
pub fn dot_of_profile(profile: &QueryProfile) -> String {
    emit_dot(&of_profile(profile))
}

/// A rendered tree: one label per node in pre-order (`ni` is the `i`th),
/// and one `(child, parent)` edge per input.
struct Graph {
    labels: Vec<String>,
    edges: Vec<(usize, usize)>,
}

fn of_plan(root: &ExplainNode) -> Graph {
    let mut graph = Graph {
        labels: Vec::new(),
        edges: Vec::new(),
    };
    walk_plan(root, None, &mut graph);
    graph
}

fn walk_plan(node: &ExplainNode, parent: Option<usize>, graph: &mut Graph) {
    let id = graph.labels.len();
    graph.labels.push(plan_label(&node.name, &node.detail));
    if let Some(parent) = parent {
        graph.edges.push((id, parent));
    }
    for child in &node.children {
        walk_plan(child, Some(id), graph);
    }
}

fn of_profile(profile: &QueryProfile) -> Graph {
    let by_node = operators_by_node(profile);
    let mut graph = Graph {
        labels: Vec::new(),
        edges: Vec::new(),
    };
    walk_profile(&profile.plan, None, &by_node, &mut graph);
    let unplaced: Vec<(OperatorId, &OperatorEntry)> = profile
        .metrics
        .operators
        .iter()
        .filter(|(_, entry)| entry.node.is_none_or(|node| !by_node.contains_key(&node)))
        .map(|(id, entry)| (*id, entry))
        .collect();
    if !unplaced.is_empty()
        && let Some(root) = graph.labels.first_mut()
    {
        root.push_str("\nResult:");
        for (id, entry) in unplaced {
            root.push('\n');
            root.push_str(&operator_line(id, entry));
        }
    }
    graph
}

fn walk_profile(
    node: &ProfileNode,
    parent: Option<usize>,
    by_node: &BTreeMap<u32, Vec<(OperatorId, &OperatorEntry)>>,
    graph: &mut Graph,
) {
    let id = graph.labels.len();
    let mut label = plan_label(&node.name, &node.detail);
    if let Some(node_id) = node.id
        && let Some(operators) = by_node.get(&node_id)
    {
        for (operator_id, entry) in operators {
            label.push('\n');
            label.push_str(&operator_line(*operator_id, entry));
        }
    }
    graph.labels.push(label);
    if let Some(parent) = parent {
        graph.edges.push((id, parent));
    }
    for child in &node.children {
        walk_profile(child, Some(id), by_node, graph);
    }
}

/// The operators of each plan node the tree shows, in operator-id order
/// (the metrics map iterates in id order).
fn operators_by_node(profile: &QueryProfile) -> BTreeMap<u32, Vec<(OperatorId, &OperatorEntry)>> {
    let mut shown = BTreeSet::new();
    collect_ids(&profile.plan, &mut shown);
    let mut by_node: BTreeMap<u32, Vec<(OperatorId, &OperatorEntry)>> = BTreeMap::new();
    for (id, entry) in &profile.metrics.operators {
        if let Some(node) = entry.node
            && shown.contains(&node)
        {
            by_node.entry(node).or_default().push((*id, entry));
        }
    }
    by_node
}

fn collect_ids(node: &ProfileNode, shown: &mut BTreeSet<u32>) {
    shown.extend(node.id);
    for child in &node.children {
        collect_ids(child, shown);
    }
}

/// The one-line plan text of a node: its name and details.
fn plan_label(name: &str, detail: &str) -> String {
    if detail.is_empty() {
        name.to_owned()
    } else {
        format!("{name} {detail}")
    }
}

/// One operator with its counters, in raw units (nanoseconds, bytes, rows)
/// so a snapshot masks the volatile ones by name.
fn operator_line(id: OperatorId, entry: &OperatorEntry) -> String {
    let metrics = &entry.metrics;
    let pipelines = entry
        .pipelines
        .iter()
        .map(|p| format!("p{p}"))
        .collect::<Vec<_>>()
        .join(",");
    let mut line = format!(
        "#{} {} {} [{pipelines}]: in_rows={} in_batches={} out_rows={} out_batches={} \
         compute_ns={} max_compute_ns={} io_ns={} max_io_ns={} mem_wait_ns={} sink_wait_ns={} \
         morsels={} peak_bytes={} spills={} spilled_bytes={} spill_files={} \
         spill_write_bytes={} drivers={}",
        id.0,
        entry.name,
        entry.role.as_str(),
        metrics.rows_in,
        metrics.batches_in,
        metrics.rows_out,
        metrics.batches_out,
        ns(metrics.compute_time),
        ns(metrics.max_driver_compute_time),
        ns(metrics.blocked_on_io),
        ns(metrics.max_driver_blocked_on_io),
        ns(metrics.blocked_on_memory),
        ns(metrics.blocked_on_sink),
        metrics.morsels,
        metrics.peak_reserved,
        metrics.spills,
        metrics.spilled_bytes,
        metrics.spill_files,
        metrics.spill_write_bytes,
        metrics.drivers,
    );
    if !entry.extra.is_empty() {
        line.push(' ');
        line.push_str(
            &entry
                .extra
                .iter()
                .map(|(name, value)| match value {
                    Metric::Count(n) | Metric::Max(n) => format!("{name}={n}"),
                    Metric::Text(text) => format!("{name}={text}"),
                })
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
    line
}

fn ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Escapes `label` for a Mermaid quoted node label (`n0["..."]`).
fn escape_mermaid(label: &str) -> String {
    // Newlines become the two characters `\n` (a raw newline would split
    // the node across lines); every other replacement adds no newline.
    let flat = label.replace("\r\n", "\n").replace('\r', "\n");
    let mut out = String::with_capacity(flat.len());
    for c in flat.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("#quot;"),
            '&' => out.push_str("#amp;"),
            '<' => out.push_str("#lt;"),
            '>' => out.push_str("#gt;"),
            '|' => out.push_str("#124;"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out
}

/// Escapes `label` for a DOT quoted string (`label="..."`): Graphviz splits
/// `\n` into lines; `<`, `>` and `|` have no quoted-string escape, so they
/// become entities (they stay literal text, never structure).
fn escape_dot(label: &str) -> String {
    let flat = label.replace("\r\n", "\n").replace('\r', "\n");
    let mut out = String::with_capacity(flat.len());
    for c in flat.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '|' => out.push_str("&#124;"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out
}

fn emit_mermaid(graph: &Graph) -> String {
    let mut out = String::from("flowchart TD\n");
    for (id, label) in graph.labels.iter().enumerate() {
        let _ = writeln!(out, "    n{id}[\"{}\"]", escape_mermaid(label));
    }
    for (child, parent) in &graph.edges {
        let _ = writeln!(out, "    n{child} --> n{parent}");
    }
    out
}

fn emit_dot(graph: &Graph) -> String {
    let mut out = String::from("digraph {\n");
    for (id, label) in graph.labels.iter().enumerate() {
        let _ = writeln!(out, "    n{id} [label=\"{}\"];", escape_dot(label));
    }
    for (child, parent) in &graph.edges {
        let _ = writeln!(out, "    n{child} -> n{parent};");
    }
    out.push_str("}\n");
    out
}

impl DataFrame {
    /// The physical plan as a graph ([`GraphFormat::Mermaid`] or
    /// [`GraphFormat::Dot`]): the tree [`DataFrame::explain`] prints, with
    /// stable pre-order node ids (`n0`, `n1`, ...) and edges from each input
    /// to its parent. Every mode plans the same physical plan, so `mode`
    /// only names the plan; the graph renders that plan either way.
    ///
    /// ```
    /// use krishiv_session::{ExplainMode, GraphFormat, Session};
    ///
    /// let session = Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # session.register_batches("t", schema, vec![])?;
    /// let text = futures::executor::block_on(
    ///     session.table("t").explain_with(ExplainMode::Simple, GraphFormat::Mermaid),
    /// )?;
    /// assert!(text.starts_with("flowchart TD\n"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Analysis and planning errors, as [`DataFrame::explain`]; with
    /// [`GraphFormat::Json`], `INVALID_PARAMETER_VALUE` (a plan mode has no
    /// JSON form: JSON exists for `EXPLAIN VERBOSE` and `EXPLAIN ANALYZE`,
    /// see [`DataFrame::explain_verbose_with`] and
    /// [`DataFrame::explain_analyze_with`]).
    pub async fn explain_with(
        &self,
        mode: ExplainMode,
        format: GraphFormat,
    ) -> Result<String, KrishivError> {
        let render: fn(&ExplainNode) -> String = match format {
            GraphFormat::Mermaid => mermaid_of_plan,
            GraphFormat::Dot => dot_of_plan,
            GraphFormat::Json => return Err(no_json_for_mode(mode)),
        };
        let frame = self.clone();
        self.session
            .on_executor(async move {
                let analysis = frame.analysis().await?;
                let (_, physical) = frame
                    .session
                    .physical(&analysis.analyzed, &analysis.env)
                    .await?;
                Ok(render(physical.explain()))
            })
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chain of `depth` nodes, each over the next.
    fn chain(depth: usize) -> ExplainNode {
        let mut node = ExplainNode::leaf(format!("leaf{depth}"), "d");
        for i in (0..depth).rev() {
            node = ExplainNode::new(format!("node{i}"), String::new(), vec![node]);
        }
        node
    }

    #[test]
    fn graph_format_parses_names_and_rejects_the_rest() {
        assert_eq!(GraphFormat::parse("json"), Ok(GraphFormat::Json));
        assert_eq!(GraphFormat::parse("Mermaid"), Ok(GraphFormat::Mermaid));
        assert_eq!(GraphFormat::parse("  DOT  "), Ok(GraphFormat::Dot));
        assert_eq!(GraphFormat::Json.as_str(), "json");
        assert_eq!(GraphFormat::Mermaid.as_str(), "mermaid");
        assert_eq!(GraphFormat::Dot.as_str(), "dot");
        for bad in ["", "yaml", "flowchart", "mermaid;dot"] {
            assert!(GraphFormat::parse(bad).is_err(), "{bad:?} parses");
        }
    }

    #[test]
    fn unknown_format_names_the_allowed_values() {
        let error = GraphFormat::parse("yaml").unwrap_err();
        assert_eq!(
            error.error_class(),
            krishiv_common::error::ErrorClass::InvalidParameterValueString
        );
        let text = error.to_string();
        assert!(text.contains("INVALID_PARAMETER_VALUE"), "{text}");
        assert!(text.contains("json"), "{text}");
        assert!(text.contains("mermaid"), "{text}");
        assert!(text.contains("dot"), "{text}");
    }

    #[test]
    fn single_node_without_children_or_detail() {
        let node = ExplainNode::leaf("Scan", "");
        assert_eq!(mermaid_of_plan(&node), "flowchart TD\n    n0[\"Scan\"]\n");
        assert_eq!(
            dot_of_plan(&node),
            "digraph {\n    n0 [label=\"Scan\"];\n}\n"
        );
    }

    #[test]
    fn escapers_cover_every_listed_character() {
        assert_eq!(escape_mermaid("a\"b"), "a#quot;b");
        assert_eq!(
            escape_mermaid("a<b>c|d\\e\nf\rg\r\nh&i"),
            "a#lt;b#gt;c#124;d\\\\e\\nf\\ng\\nh#amp;i"
        );
        assert_eq!(escape_dot("a\"b"), "a\\\"b");
        assert_eq!(
            escape_dot("a<b>c|d\\e\nf\rg\r\nh&i"),
            "a&lt;b&gt;c&#124;d\\\\e\\nf\\ng\\nh&amp;i"
        );
    }

    #[test]
    fn labels_escape_quotes_newlines_angles_pipes_and_backslashes() {
        let node = ExplainNode::new(
            "We\"ird",
            "a<b>|c\\d\ne\r\nf\rg&h",
            vec![ExplainNode::leaf("kid", "")],
        );
        let mermaid = mermaid_of_plan(&node);
        assert!(
            mermaid.contains("We#quot;ird a#lt;b#gt;#124;c\\\\d\\ne\\nf\\ng#amp;h"),
            "{mermaid}"
        );
        let dot = dot_of_plan(&node);
        assert!(
            dot.contains("We\\\"ird a&lt;b&gt;&#124;c\\\\d\\ne\\nf\\ng&amp;h"),
            "{dot}"
        );
    }

    #[test]
    fn deep_chain_numbers_pre_order_with_edges_to_parents() {
        let text = mermaid_of_plan(&chain(4));
        for i in 0..=4 {
            assert!(text.contains(&format!("n{i}[")), "{text}");
        }
        assert!(text.contains("n4 --> n3"), "{text}");
        assert!(text.contains("n1 --> n0"), "{text}");
        let dot = dot_of_plan(&chain(4));
        assert!(dot.contains("n4 -> n3"), "{dot}");
        assert!(dot.contains("n1 -> n0"), "{dot}");
    }

    #[test]
    fn wide_union_orders_children_before_their_parent() {
        let node = ExplainNode::new(
            "Union",
            "",
            vec![
                ExplainNode::leaf("left", "a"),
                ExplainNode::leaf("middle", "b"),
                ExplainNode::leaf("right", "c"),
            ],
        );
        let text = mermaid_of_plan(&node);
        let ids: Vec<usize> = text
            .lines()
            .skip(1)
            .take_while(|l| l.contains('['))
            .map(|l| l.trim()[1..2].parse().unwrap_or_else(|e| panic!("{e}")))
            .collect();
        assert_eq!(ids, vec![0, 1, 2, 3], "{text}");
        assert!(text.contains("n1 --> n0"), "{text}");
        assert!(text.contains("n2 --> n0"), "{text}");
        assert!(text.contains("n3 --> n0"), "{text}");
    }

    #[test]
    fn node_ids_are_stable_and_follow_child_order() {
        let left = ExplainNode::new(
            "Join",
            "on",
            vec![ExplainNode::leaf("a", ""), ExplainNode::leaf("b", "")],
        );
        let right = ExplainNode::new(
            "Join",
            "on",
            vec![ExplainNode::leaf("b", ""), ExplainNode::leaf("a", "")],
        );
        assert_eq!(mermaid_of_plan(&left), mermaid_of_plan(&left));
        assert_ne!(mermaid_of_plan(&left), mermaid_of_plan(&right));
        assert_ne!(dot_of_plan(&left), dot_of_plan(&right));
    }

    #[test]
    fn profile_labels_hold_each_operators_counters() {
        use krishiv_exec::{Metric, OperatorEntry, OperatorMetrics, OperatorRole, QueryMetrics};

        let mut metrics = QueryMetrics::default();
        metrics.operators.insert(
            OperatorId(7),
            OperatorEntry {
                name: "MemoryScan".into(),
                node: Some(0),
                role: OperatorRole::Source,
                pipelines: vec![2],
                metrics: OperatorMetrics {
                    rows_out: 10,
                    batches_out: 2,
                    peak_reserved: 64,
                    ..OperatorMetrics::default()
                },
                extra: vec![(std::borrow::Cow::Borrowed("groups"), Metric::Count(3))],
            },
        );
        let profile = QueryProfile {
            plan: ProfileNode {
                id: Some(0),
                name: "Scan".to_owned(),
                detail: "t".to_owned(),
                children: Vec::new(),
            },
            metrics,
            detailed: false,
        };
        let text = mermaid_of_profile(&profile);
        assert!(text.contains("Scan t"), "{text}");
        assert!(text.contains("#7 MemoryScan source [p2]"), "{text}");
        assert!(text.contains("out_rows=10 out_batches=2"), "{text}");
        assert!(text.contains("peak_bytes=64"), "{text}");
        assert!(text.contains("groups=3"), "{text}");
        // No plan node holds the result sink: it prints under `Result:`,
        // as does an operator of a plan node the tree does not show (an
        // unmatched input).
        let mut sink_metrics = QueryMetrics::default();
        sink_metrics.operators.insert(
            OperatorId(0),
            OperatorEntry {
                name: "Result".into(),
                node: None,
                role: OperatorRole::Sink,
                pipelines: Vec::new(),
                metrics: OperatorMetrics {
                    rows_in: 10,
                    batches_in: 2,
                    ..OperatorMetrics::default()
                },
                extra: Vec::new(),
            },
        );
        sink_metrics.operators.insert(
            OperatorId(1),
            OperatorEntry {
                name: "Orphan".into(),
                node: Some(9),
                role: OperatorRole::Operator,
                pipelines: vec![0],
                metrics: OperatorMetrics::default(),
                extra: Vec::new(),
            },
        );
        let with_sink = QueryProfile {
            plan: ProfileNode {
                id: Some(0),
                name: "Scan".to_owned(),
                detail: String::new(),
                children: Vec::new(),
            },
            metrics: sink_metrics,
            detailed: false,
        };
        let text = dot_of_profile(&with_sink);
        assert!(text.contains("Result:"), "{text}");
        assert!(text.contains("#0 Result sink []"), "{text}");
        assert!(text.contains("in_rows=10 in_batches=2"), "{text}");
        assert!(text.contains("#1 Orphan operator [p0]"), "{text}");
    }

    #[test]
    fn profile_nodes_without_ids_hold_no_operators() {
        use krishiv_exec::QueryMetrics;

        let profile = QueryProfile {
            plan: ProfileNode {
                id: None,
                name: "Subquery".to_owned(),
                detail: String::new(),
                children: vec![ProfileNode {
                    id: Some(0),
                    name: "Scan".to_owned(),
                    detail: "t".to_owned(),
                    children: Vec::new(),
                }],
            },
            metrics: QueryMetrics::default(),
            detailed: true,
        };
        assert_eq!(
            mermaid_of_profile(&profile),
            "flowchart TD\n    n0[\"Subquery\"]\n    n1[\"Scan t\"]\n    n1 --> n0\n"
        );
    }

    #[test]
    fn profile_without_metrics_is_the_plan_tree() {
        use krishiv_exec::QueryMetrics;

        let profile = QueryProfile {
            plan: chain_profile(2),
            metrics: QueryMetrics::default(),
            detailed: true,
        };
        assert_eq!(
            mermaid_of_profile(&profile),
            "flowchart TD\n    n0[\"p0\"]\n    n1[\"p1\"]\n    n2[\"p2\"]\n    n1 --> n0\n    n2 --> n1\n"
        );
    }

    fn chain_profile(depth: u32) -> ProfileNode {
        let mut node = ProfileNode {
            id: Some(depth),
            name: format!("p{depth}"),
            detail: String::new(),
            children: Vec::new(),
        };
        for i in (0..depth).rev() {
            node = ProfileNode {
                id: Some(i),
                name: format!("p{i}"),
                detail: String::new(),
                children: vec![node],
            };
        }
        node
    }
}
