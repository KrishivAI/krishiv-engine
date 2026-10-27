// SPDX-License-Identifier: Apache-2.0
//! The property-enforcement pass (ADR 0012, update of 2026-10-17; Spark's
//! `EnsureRequirements`): every node states the distribution it requires
//! of each input, every planned relation states the placement it provides,
//! and an [`Exchange`](PhysNode::Exchange) is inserted exactly where an
//! input's placement does not meet the requirement.
//!
//! The pass runs as each node is planned, bottom-up, for every worker
//! count including one (design P2-T171 §5). Placements are tracked by
//! attribute id, so they follow columns through projections, filters and
//! joins, and are dropped when a key column is.
//!
//! | Node | Requires of its input(s) | Provides |
//! |---|---|---|
//! | scan, union | — | [`Placement::Unknown`] |
//! | one row | — | [`Placement::Single`] |
//! | filter, replicate | — | its input's |
//! | project | — | its input's, under each name a plain column is projected as |
//! | grouped aggregate | [`Distribution::Clustered`] on the keys | its input's |
//! | global aggregate, limit, top-N | [`Distribution::Single`] (a limit or top-N first cuts each partition to `n` rows) | single |
//! | global sort | [`Distribution::Ordered`] on the keys | ranges of the keys |
//! | broadcast hash join, nested loop | build: [`Distribution::Broadcast`] | the probe's |
//! | partitioned hash join | both: hash on corresponding keys | the probe's; an inner join's build keys too |
//!
//! Satisfaction follows Spark: a hash placement on a subset of the
//! required keys clusters them; a single partition meets every
//! requirement except broadcast. A partitioned join also needs its two
//! inputs placed *alike*: hash keys at the same key positions, so that
//! rows with equal keys meet in the same partition. An input already
//! placed that way is kept and only the other one is exchanged, which is
//! how an aggregate or join on the same keys below avoids a second
//! shuffle (Spark's co-partitioning, ADR 0010's bucketed joins).
//!
//! Every hash exchange of a plan hashes with the same function, named in
//! the plan ([`HashScheme::KeyGroupV1`]: key-group ranges of the routing
//! hash `H_kg`), and the same partition count (chosen per query by the
//! task, I5), so equal hash keys mean equal placement.

use std::sync::Arc;

use krishiv_plan::AttrId;

use super::build::{Planned, Planner};
use super::explain::ExplainNode;
use crate::error::Result;
use crate::operators::SortKey;
use crate::physical::{ExchangeId, ExchangePlan, HashScheme, Partitioning, PhysNode};

/// A sort key of a range placement, by attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RangeKey {
    pub(crate) attr: AttrId,
    pub(crate) descending: bool,
    pub(crate) nulls_first: bool,
}

/// How a relation's rows are placed in partitions: what a planned
/// relation provides.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) enum Placement {
    /// No known relation between values and partitions (scans, unions).
    #[default]
    Unknown,
    /// All rows in one partition.
    Single,
    /// Every partition holds every row (a broadcast).
    Replicated,
    /// By the routing hash of key attributes. Each entry is one key list
    /// the placement is by: an inner join on `a = b` is placed by `[a]`
    /// and by `[b]` alike.
    Hash(Vec<Vec<AttrId>>),
    /// By ranges of sort keys, partitions in key order.
    Range(Vec<RangeKey>),
}

/// What a node requires of an input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Distribution<'a> {
    /// All rows in one partition.
    Single,
    /// Rows with equal values of these attributes in one partition.
    Clustered(&'a [AttrId]),
    /// Every row in every partition.
    Broadcast,
    /// Partitions in the order of these keys (a global sort).
    Ordered(&'a [RangeKey]),
}

impl Placement {
    /// Whether rows placed so meet `required` without an exchange.
    pub(crate) fn satisfies(&self, required: Distribution<'_>) -> bool {
        match required {
            Distribution::Single => *self == Self::Single,
            Distribution::Broadcast => *self == Self::Replicated,
            Distribution::Clustered(keys) => match self {
                Self::Single => true,
                Self::Hash(alts) => alts
                    .iter()
                    .any(|alt| !alt.is_empty() && alt.iter().all(|a| keys.contains(a))),
                Self::Range(by) => !by.is_empty() && by.iter().all(|k| keys.contains(&k.attr)),
                Self::Unknown | Self::Replicated => false,
            },
            Distribution::Ordered(keys) => match self {
                Self::Single => true,
                Self::Range(by) => by.as_slice() == keys,
                _ => false,
            },
        }
    }

    /// The placement as far as it can be stated over the attributes of
    /// `layout` (a node that drops a key column loses its placement).
    pub(crate) fn restrict(self, layout: &[AttrId]) -> Self {
        self.map_attrs(|a| layout.contains(&a).then_some(a))
    }

    /// The placement over the outputs of a projection. `pairs` lists
    /// `(source, output)` for each output that is a plain column (or an
    /// alias of one) of the input: a source projected under several names
    /// (`SELECT a, a AS k`) is placed by each of them. A hash key list
    /// becomes every list that takes one output per source key (never
    /// several outputs in one list, which would change the composite hash
    /// key) and is dropped if a key is not projected; a range follows the
    /// first output of each key. Widening casts are not plain columns, so
    /// they never map: their values can hash differently.
    pub(crate) fn project(self, pairs: &[(AttrId, AttrId)]) -> Self {
        /// Enough alternatives for any real plan; keeps a wide fan-out of
        /// aliases from growing combinatorially.
        const MAX_ALTERNATIVES: usize = 32;
        let outputs = |src: AttrId| {
            pairs
                .iter()
                .filter(move |(s, _)| *s == src)
                .map(|(_, o)| *o)
        };
        match self {
            Self::Hash(alts) => {
                let mut out: Vec<Vec<AttrId>> = Vec::new();
                'alts: for alt in alts {
                    let mut lists: Vec<Vec<AttrId>> = vec![Vec::new()];
                    for key in alt {
                        let choices: Vec<AttrId> = outputs(key).collect();
                        if choices.is_empty() {
                            continue 'alts;
                        }
                        lists = lists
                            .iter()
                            .flat_map(|l| {
                                choices.iter().map(move |c| {
                                    let mut l = l.clone();
                                    l.push(*c);
                                    l
                                })
                            })
                            .take(MAX_ALTERNATIVES)
                            .collect();
                    }
                    for l in lists {
                        if out.len() < MAX_ALTERNATIVES && !out.contains(&l) {
                            out.push(l);
                        }
                    }
                }
                if out.is_empty() {
                    Self::Unknown
                } else {
                    Self::Hash(out)
                }
            }
            Self::Range(keys) => keys
                .into_iter()
                .map(|k| outputs(k.attr).next().map(|attr| RangeKey { attr, ..k }))
                .collect::<Option<Vec<_>>>()
                .map_or(Self::Unknown, Self::Range),
            other => other,
        }
    }

    /// The placement with each attribute renamed by `f`; an attribute `f`
    /// drops drops the key lists that hold it.
    pub(crate) fn map_attrs(self, f: impl Fn(AttrId) -> Option<AttrId>) -> Self {
        match self {
            Self::Hash(alts) => {
                let alts: Vec<Vec<AttrId>> = alts
                    .into_iter()
                    .filter_map(|alt| alt.into_iter().map(&f).collect::<Option<Vec<_>>>())
                    .collect();
                if alts.is_empty() {
                    Self::Unknown
                } else {
                    Self::Hash(alts)
                }
            }
            Self::Range(keys) => keys
                .into_iter()
                .map(|k| f(k.attr).map(|attr| RangeKey { attr, ..k }))
                .collect::<Option<Vec<_>>>()
                .map_or(Self::Unknown, Self::Range),
            other => other,
        }
    }

    /// The key positions (into `keys`) of each of this placement's hash
    /// key lists that lie within `keys`.
    fn positions(&self, keys: &[AttrId]) -> Vec<Vec<usize>> {
        let Self::Hash(alts) = self else {
            return Vec::new();
        };
        alts.iter()
            .filter(|alt| !alt.is_empty())
            .filter_map(|alt| {
                alt.iter()
                    .map(|a| keys.iter().position(|k| k == a))
                    .collect::<Option<Vec<_>>>()
            })
            .collect()
    }

    /// Whether this is a hash placement by exactly `keys`.
    fn hashed_by(&self, keys: &[AttrId]) -> bool {
        matches!(self, Self::Hash(alts) if alts.iter().any(|alt| alt == keys))
    }

    /// The placement of an inner join's output: rows are placed by the
    /// probe's keys and, equal on matched rows, by the build's.
    pub(crate) fn union(self, other: Self) -> Self {
        match (self, other) {
            (Self::Hash(mut a), Self::Hash(b)) => {
                for alt in b {
                    if !a.contains(&alt) {
                        a.push(alt);
                    }
                }
                Self::Hash(a)
            }
            (a, _) => a,
        }
    }
}

/// The attribute keys of `keys`, columns of a relation with `layout`.
pub(crate) fn range_keys(keys: &[SortKey], layout: &[AttrId]) -> Result<Vec<RangeKey>> {
    keys.iter()
        .map(|k| {
            let attr = layout
                .get(k.column)
                .copied()
                .ok_or_else(|| super::build::internal("a sort key out of its input"))?;
            Ok(RangeKey {
                attr,
                descending: k.descending,
                nulls_first: k.nulls_first,
            })
        })
        .collect()
}

impl Planner<'_> {
    /// `input`, under an exchange if its placement does not meet
    /// `required`.
    pub(crate) fn require(
        &mut self,
        input: Planned,
        required: Distribution<'_>,
    ) -> Result<Planned> {
        if input.placement.satisfies(required) {
            return Ok(input);
        }
        let partitioning = match required {
            Distribution::Single => Partitioning::Single,
            Distribution::Broadcast => Partitioning::Broadcast,
            Distribution::Clustered(keys) => Partitioning::Hash {
                keys: keys
                    .iter()
                    .map(|a| input.column(*a))
                    .collect::<Result<_>>()?,
                scheme: HashScheme::KeyGroupV1,
            },
            Distribution::Ordered(keys) => Partitioning::Range {
                keys: keys
                    .iter()
                    .map(|k| {
                        Ok(SortKey {
                            column: input.column(k.attr)?,
                            descending: k.descending,
                            nulls_first: k.nulls_first,
                        })
                    })
                    .collect::<Result<_>>()?,
            },
        };
        self.exchange(input, partitioning)
    }

    /// The two inputs of a partitioned join placed alike on their keys
    /// (columns `probe_keys` of `probe` and `build_keys` of `build`, pairwise
    /// equal): an input already hash-placed on some of its keys is kept,
    /// and the other one is exchanged on the corresponding keys.
    pub(crate) fn co_partition(
        &mut self,
        probe: Planned,
        probe_keys: &[usize],
        build: Planned,
        build_keys: &[usize],
    ) -> Result<(Planned, Planned)> {
        let attrs = |p: &Planned, keys: &[usize]| -> Result<Vec<AttrId>> {
            keys.iter()
                .map(|&c| {
                    p.layout
                        .get(c)
                        .copied()
                        .ok_or_else(|| super::build::internal("a join key out of its input"))
                })
                .collect()
        };
        let (pa, ba) = (attrs(&probe, probe_keys)?, attrs(&build, build_keys)?);
        let pick = |keys: &[AttrId], at: &[usize]| -> Vec<AttrId> {
            at.iter().filter_map(|&i| keys.get(i).copied()).collect()
        };
        if probe.placement == Placement::Single && build.placement == Placement::Single {
            return Ok((probe, build));
        }
        let probe_at = probe.placement.positions(&pa);
        let build_at = build.placement.positions(&ba);
        if probe_at
            .iter()
            .any(|at| build.placement.hashed_by(&pick(&ba, at)))
        {
            return Ok((probe, build));
        }
        let hash = |p: &Planned, keys: &[AttrId]| -> Result<Partitioning> {
            Ok(Partitioning::Hash {
                keys: keys.iter().map(|a| p.column(*a)).collect::<Result<_>>()?,
                scheme: HashScheme::KeyGroupV1,
            })
        };
        if let Some(at) = probe_at.first() {
            let by = hash(&build, &pick(&ba, at))?;
            let build = self.exchange(build, by)?;
            return Ok((probe, build));
        }
        if let Some(at) = build_at.first() {
            let by = hash(&probe, &pick(&pa, at))?;
            let probe = self.exchange(probe, by)?;
            return Ok((probe, build));
        }
        let (pb, bb) = (hash(&probe, &pa)?, hash(&build, &ba)?);
        Ok((self.exchange(probe, pb)?, self.exchange(build, bb)?))
    }

    /// `input` under an exchange by `partitioning`.
    fn exchange(&mut self, input: Planned, partitioning: Partitioning) -> Result<Planned> {
        let id = ExchangeId(self.next_exchange);
        self.next_exchange = self
            .next_exchange
            .checked_add(1)
            .ok_or_else(|| super::build::internal("exchange id overflow"))?;
        let (title, detail, placement) = describe(&partitioning, &input.schema, &input.layout)?;
        Ok(Planned {
            node: PhysNode::Exchange(ExchangePlan {
                id,
                input: Box::new(input.node),
                partitioning,
            }),
            schema: Arc::clone(&input.schema),
            layout: input.layout,
            explain: ExplainNode::new(title, detail, vec![input.explain]),
            sites: input.sites,
            placement,
        })
    }
}

/// How `EXPLAIN` names an exchange by `partitioning` over an input of
/// `schema` whose columns are `layout`, and the placement it provides.
fn describe(
    partitioning: &Partitioning,
    schema: &arrow_schema::Schema,
    layout: &[AttrId],
) -> Result<(&'static str, String, Placement)> {
    let name = |c: usize| -> Result<String> {
        match (schema.fields().get(c), layout.get(c)) {
            (Some(f), Some(a)) => Ok(format!("{}#{}", f.name(), a.0)),
            _ => Err(super::build::internal("an exchange key out of its input")),
        }
    };
    Ok(match partitioning {
        Partitioning::Hash { keys, .. } => {
            let names = keys.iter().map(|&c| name(c)).collect::<Result<Vec<_>>>()?;
            let attrs =
                keys.iter()
                    .map(|&c| {
                        layout.get(c).copied().ok_or_else(|| {
                            super::build::internal("an exchange key out of its input")
                        })
                    })
                    .collect::<Result<_>>()?;
            (
                "Exchange",
                format!("hashpartitioning({})", names.join(", ")),
                Placement::Hash(vec![attrs]),
            )
        }
        Partitioning::Single => ("Exchange", "SinglePartition".to_owned(), Placement::Single),
        Partitioning::Broadcast => ("BroadcastExchange", String::new(), Placement::Replicated),
        Partitioning::RoundRobin => (
            "Exchange",
            "RoundRobinPartitioning".to_owned(),
            Placement::Unknown,
        ),
        Partitioning::Range { keys } => {
            let names = keys
                .iter()
                .map(|k| {
                    Ok(format!(
                        "{} {} NULLS {}",
                        name(k.column)?,
                        if k.descending { "DESC" } else { "ASC" },
                        if k.nulls_first { "FIRST" } else { "LAST" }
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            (
                "Exchange",
                format!("rangepartitioning({})", names.join(", ")),
                Placement::Range(range_keys(keys, layout)?),
            )
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(ids: &[u64]) -> Vec<AttrId> {
        ids.iter().map(|i| AttrId(*i)).collect()
    }

    fn asc(id: u64) -> RangeKey {
        RangeKey {
            attr: AttrId(id),
            descending: false,
            nulls_first: true,
        }
    }

    #[test]
    fn clustering_is_met_by_a_hash_on_a_subset_of_the_keys() {
        let keys = a(&[1, 2]);
        let d = Distribution::Clustered(&keys);
        assert!(Placement::Hash(vec![a(&[2])]).satisfies(d));
        assert!(Placement::Hash(vec![a(&[2, 1])]).satisfies(d));
        assert!(Placement::Hash(vec![a(&[3]), a(&[1])]).satisfies(d));
        assert!(!Placement::Hash(vec![a(&[1, 3])]).satisfies(d));
        assert!(!Placement::Hash(vec![Vec::new()]).satisfies(d));
        assert!(Placement::Single.satisfies(d));
        assert!(!Placement::Unknown.satisfies(d));
        assert!(!Placement::Replicated.satisfies(d));
        assert!(Placement::Range(vec![asc(1)]).satisfies(d));
        assert!(!Placement::Range(vec![asc(3)]).satisfies(d));
        assert!(!Placement::Range(Vec::new()).satisfies(d));
    }

    #[test]
    fn single_broadcast_and_order() {
        for p in [
            Placement::Unknown,
            Placement::Replicated,
            Placement::Hash(vec![a(&[1])]),
        ] {
            assert!(!p.satisfies(Distribution::Single), "{p:?}");
        }
        assert!(Placement::Single.satisfies(Distribution::Single));
        assert!(Placement::Replicated.satisfies(Distribution::Broadcast));
        assert!(!Placement::Single.satisfies(Distribution::Broadcast));
        let order = [asc(1), asc(2)];
        let d = Distribution::Ordered(&order);
        assert!(Placement::Single.satisfies(d));
        assert!(Placement::Range(order.to_vec()).satisfies(d));
        assert!(!Placement::Range(vec![asc(1)]).satisfies(d));
        let desc = RangeKey {
            descending: true,
            ..asc(1)
        };
        assert!(!Placement::Range(vec![desc, asc(2)]).satisfies(d));
        assert!(!Placement::Hash(vec![a(&[1, 2])]).satisfies(d));
    }

    #[test]
    fn placements_follow_their_columns() {
        let p = Placement::Hash(vec![a(&[1]), a(&[2, 3])]);
        assert_eq!(
            p.clone().restrict(&a(&[1, 2])),
            Placement::Hash(vec![a(&[1])])
        );
        assert_eq!(p.clone().restrict(&a(&[4])), Placement::Unknown);
        assert_eq!(
            p.map_attrs(|x| Some(AttrId(x.0 + 10))),
            Placement::Hash(vec![a(&[11]), a(&[12, 13])])
        );
        let r = Placement::Range(vec![asc(1), asc(2)]);
        assert_eq!(r.clone().restrict(&a(&[1, 2])), r);
        assert_eq!(r.restrict(&a(&[1])), Placement::Unknown);
        assert_eq!(Placement::Single.restrict(&[]), Placement::Single);
        assert_eq!(Placement::Replicated.restrict(&[]), Placement::Replicated);
    }

    #[test]
    fn a_projection_places_rows_by_every_name_of_a_key() {
        let at = |pairs: &[(u64, u64)]| -> Vec<(AttrId, AttrId)> {
            pairs.iter().map(|&(s, o)| (AttrId(s), AttrId(o))).collect()
        };
        let p = Placement::Hash(vec![a(&[1]), a(&[2, 3])]);
        // `a AS k1, a AS k2, a`: one list per name, never `[k1, k2]`.
        assert_eq!(
            p.clone().project(&at(&[(1, 11), (1, 12), (1, 1)])),
            Placement::Hash(vec![a(&[11]), a(&[12]), a(&[1])])
        );
        // A composite key: one output per key position, every combination.
        assert_eq!(
            p.clone().project(&at(&[(2, 20), (2, 21), (3, 30)])),
            Placement::Hash(vec![a(&[20, 30]), a(&[21, 30])])
        );
        // A key that is not projected as a plain column drops its list.
        assert_eq!(p.clone().project(&at(&[(2, 20)])), Placement::Unknown);
        assert_eq!(
            p.project(&at(&[(1, 11), (2, 20)])),
            Placement::Hash(vec![a(&[11])])
        );
        // The same list from two sources is kept once.
        assert_eq!(
            Placement::Hash(vec![a(&[1]), a(&[2])]).project(&at(&[(1, 7), (2, 7)])),
            Placement::Hash(vec![a(&[7])])
        );
        // A range follows the first name of each key.
        let r = Placement::Range(vec![asc(1), asc(2)]);
        assert_eq!(
            r.clone().project(&at(&[(1, 5), (2, 6), (2, 7)])),
            Placement::Range(vec![asc(5), asc(6)])
        );
        assert_eq!(r.project(&at(&[(1, 5)])), Placement::Unknown);
        assert_eq!(Placement::Single.project(&[]), Placement::Single);
        // Too many names stay bounded.
        let many: Vec<(AttrId, AttrId)> = (0..8)
            .flat_map(|k| (0..8).map(move |n| (AttrId(k), AttrId(100 + 10 * k + n))))
            .collect();
        let Placement::Hash(alts) =
            Placement::Hash(vec![a(&[0, 1, 2, 3, 4, 5, 6, 7])]).project(&many)
        else {
            panic!("a hash placement")
        };
        assert_eq!(alts.len(), 32);
        assert!(alts.iter().all(|l| l.len() == 8));
    }

    #[test]
    fn an_inner_join_is_placed_by_both_sides_keys() {
        let probe = Placement::Hash(vec![a(&[1])]);
        assert_eq!(
            probe.clone().union(Placement::Hash(vec![a(&[2]), a(&[1])])),
            Placement::Hash(vec![a(&[1]), a(&[2])])
        );
        assert_eq!(probe.clone().union(Placement::Single), probe);
        assert_eq!(
            Placement::Single.union(Placement::Single),
            Placement::Single
        );
    }

    #[test]
    fn hash_positions_name_the_join_keys() {
        let p = Placement::Hash(vec![a(&[7]), a(&[2, 1]), a(&[9])]);
        assert_eq!(p.positions(&a(&[1, 2, 7])), vec![vec![2], vec![1, 0]]);
        assert!(Placement::Single.positions(&a(&[1])).is_empty());
        assert!(p.hashed_by(&a(&[2, 1])));
        assert!(!p.hashed_by(&a(&[1, 2])));
        assert!(!Placement::Unknown.hashed_by(&a(&[1])));
    }

    #[test]
    fn exchanges_print_their_partitioning_and_place_rows() {
        use arrow_schema::{DataType, Field, Schema};
        let schema = Schema::new(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("y", DataType::Utf8, true),
        ]);
        let layout = a(&[5, 6]);
        let d = |p: &Partitioning| describe(p, &schema, &layout).unwrap();
        assert_eq!(
            d(&Partitioning::Hash {
                keys: vec![1, 0],
                scheme: HashScheme::KeyGroupV1
            }),
            (
                "Exchange",
                "hashpartitioning(y#6, x#5)".to_owned(),
                Placement::Hash(vec![a(&[6, 5])])
            )
        );
        assert_eq!(
            d(&Partitioning::RoundRobin),
            (
                "Exchange",
                "RoundRobinPartitioning".to_owned(),
                Placement::Unknown
            )
        );
        assert_eq!(
            d(&Partitioning::Broadcast),
            ("BroadcastExchange", String::new(), Placement::Replicated)
        );
        let (_, detail, placement) = d(&Partitioning::Range {
            keys: vec![SortKey::desc(1), SortKey::asc(0).nulls_first(false)],
        });
        assert_eq!(
            detail,
            "rangepartitioning(y#6 DESC NULLS LAST, x#5 ASC NULLS LAST)"
        );
        assert_eq!(
            placement,
            Placement::Range(vec![
                RangeKey {
                    attr: AttrId(6),
                    descending: true,
                    nulls_first: false
                },
                RangeKey {
                    nulls_first: false,
                    ..asc(5)
                }
            ])
        );
        for bad in [
            Partitioning::Hash {
                keys: vec![2],
                scheme: HashScheme::KeyGroupV1,
            },
            Partitioning::Range {
                keys: vec![SortKey::asc(2)],
            },
        ] {
            let e = describe(&bad, &schema, &layout).unwrap_err();
            assert_eq!(
                e.error_class(),
                krishiv_common::error::ErrorClass::InternalError,
                "{e}"
            );
            assert!(e.to_string().contains("out of its input"), "{e}");
        }
    }
}
