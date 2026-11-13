// SPDX-License-Identifier: Apache-2.0
//! The writers' column metrics modes (review F3): the table properties
//! `write.metadata.metrics.default` and `write.metadata.metrics.column.<col>`
//! say how a writer truncated string and binary bounds, or whether it wrote
//! bounds at all. Scan pruning takes an upper bound as exact only when it is
//! shorter than the column's truncation length, so that length must be the
//! table's, not a constant.
//!
//! Modes (Iceberg's `MetricsModes`): `none` and `counts` write no bounds, so any
//! bound found is not trusted; `truncate(n)` truncates to `n`; `full` does not
//! truncate. Without the property the default is `truncate(16)`. A value that
//! does not parse makes every string and binary upper bound of the column a
//! prefix (no assumption about the writer).

use std::collections::BTreeMap;

use super::prune::DEFAULT_TRUNCATE_LENGTH;
use super::types::SchemaField;

/// The table property of the default mode.
pub const METRICS_DEFAULT: &str = "write.metadata.metrics.default";
/// The prefix of a column's mode property (the column's dotted name follows).
pub const METRICS_COLUMN_PREFIX: &str = "write.metadata.metrics.column.";

/// A column's metrics mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricsMode {
    /// No metrics: bounds are not trusted.
    None,
    /// Counts only: bounds are not trusted.
    Counts,
    /// Bounds truncated to this many characters (strings) or bytes (binaries).
    Truncate(usize),
    /// Bounds not truncated.
    Full,
    /// A value that does not parse: string and binary upper bounds are prefixes.
    Unknown,
}

impl MetricsMode {
    /// The mode `text` names (case-insensitive, as Java's `MetricsModes.fromString`).
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let t = text.trim().to_ascii_lowercase();
        match t.as_str() {
            "none" => Self::None,
            "counts" => Self::Counts,
            "full" => Self::Full,
            _ => t
                .strip_prefix("truncate(")
                .and_then(|r| r.strip_suffix(')'))
                .and_then(|n| n.trim().parse::<usize>().ok())
                .filter(|n| *n > 0)
                .map_or(Self::Unknown, Self::Truncate),
        }
    }

    /// Whether bounds written under this mode may be used.
    #[must_use]
    pub fn trusts_bounds(self) -> bool {
        !matches!(self, Self::None | Self::Counts)
    }
}

/// The metrics mode of each column, by field id (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriterMetrics {
    default: MetricsMode,
    columns: BTreeMap<i32, MetricsMode>,
}

impl Default for WriterMetrics {
    fn default() -> Self {
        Self {
            default: MetricsMode::Truncate(DEFAULT_TRUNCATE_LENGTH),
            columns: BTreeMap::new(),
        }
    }
}

impl WriterMetrics {
    /// The modes the table `properties` set for the columns `fields` (a
    /// column property names the column by its dotted path).
    #[must_use]
    pub fn from_properties<'p>(
        properties: impl IntoIterator<Item = (&'p str, &'p str)>,
        fields: &[SchemaField],
    ) -> Self {
        let mut out = Self::default();
        for (k, v) in properties {
            if k == METRICS_DEFAULT {
                out.default = MetricsMode::parse(v);
            } else if let Some(col) = k.strip_prefix(METRICS_COLUMN_PREFIX)
                && let Some(f) = fields.iter().find(|f| f.path.0.join(".") == col)
            {
                out.columns.insert(f.id, MetricsMode::parse(v));
            }
        }
        out
    }

    /// The mode of the column with field id `id`.
    #[must_use]
    pub fn mode(&self, id: i32) -> MetricsMode {
        self.columns.get(&id).copied().unwrap_or(self.default)
    }
}
