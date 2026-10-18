// SPDX-License-Identifier: Apache-2.0
//! Connect relations to IR relations, after Spark's
//! `SparkConnectPlanner.transformRelation`.
//!
//! Supported: `Read` (named tables and data sources), `SQL`, `Project`,
//! `Filter`, `Join`, `Sort`, `Limit`, `Aggregate`, `SubqueryAlias` and
//! `WithColumns`. The `plan_id` of a relation's `RelationCommon` becomes the
//! `Origin` of the node it translates to, as Spark tags the plan it builds
//! for the relation; `df["a"]` references name that id.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::secret::OptionValue;
use krishiv_plan::{
    Expr, ExprNode, Grouping, JoinCondition, JoinKind, Literal, LogicalPlan, Origin, PlanId,
    PlanNode, ReadSource, SqlArgs,
};

use super::errors::{
    assertion, invalid_enum, not_set, relation_not_set, unsupported, unsupported_field,
    unsupported_variant,
};
use super::names::parse_multipart_identifier;
use super::{Result, Translator};
use crate::proto::v4_2::{self as pb, Expression};
use crate::proto::v4_2::{
    Relation, aggregate::GroupType, join::JoinType, read::ReadType, relation::RelType,
};

impl Translator<'_> {
    /// Translates a relation and everything below it.
    pub(crate) fn relation(&mut self, rel: &Relation) -> Result<Arc<LogicalPlan>> {
        self.count()?;
        self.nested(|t| t.relation_inner(rel))
    }

    fn relation_inner(&mut self, rel: &Relation) -> Result<Arc<LogicalPlan>> {
        let Some(kind) = &rel.rel_type else {
            return Err(relation_not_set());
        };
        let node = match kind {
            RelType::Read(r) => self.read(r)?,
            RelType::Sql(s) => self.sql(s)?,
            RelType::Project(p) => {
                let Some(input) = &p.input else {
                    // Spark reads a missing input as `OneRowRelation`.
                    return Err(unsupported_field(
                        "Project",
                        "input",
                        "projections without an input relation",
                    ));
                };
                PlanNode::Project {
                    input: self.relation(input)?,
                    exprs: self.expressions(&p.expressions)?,
                }
            }
            RelType::Filter(f) => PlanNode::Filter {
                input: self.input("Filter", f.input.as_deref())?,
                predicate: self.required_expression(f.condition.as_ref())?,
            },
            RelType::Join(j) => self.join(j)?,
            RelType::Sort(s) => {
                if s.order.is_empty() {
                    return Err(assertion("'order' must be present and contain elements."));
                }
                PlanNode::Sort {
                    input: self.input("Sort", s.input.as_deref())?,
                    order: s
                        .order
                        .iter()
                        .map(|o| self.sort_order(o))
                        .collect::<Result<_>>()?,
                    global: s.is_global.unwrap_or(false),
                }
            }
            RelType::Limit(l) => PlanNode::Limit {
                input: self.input("Limit", l.input.as_deref())?,
                limit: Arc::new(Expr::literal(Literal::Integer(l.limit))),
            },
            RelType::Aggregate(a) => self.aggregate(a)?,
            RelType::SubqueryAlias(a) => {
                if !a.qualifier.is_empty() {
                    return Err(unsupported_field(
                        "SubqueryAlias",
                        "qualifier",
                        "relation aliases with a qualifier",
                    ));
                }
                PlanNode::SubqueryAlias {
                    input: self.input("SubqueryAlias", a.input.as_deref())?,
                    alias: a.alias.clone(),
                    columns: Vec::new(),
                }
            }
            RelType::WithColumns(w) => self.with_columns(w)?,
            other => {
                return Err(unsupported_variant(
                    "relation",
                    "spark.connect.Relation",
                    other.variant(),
                ));
            }
        };
        let origin = rel
            .common
            .as_ref()
            .and_then(|c| c.plan_id)
            .map_or(Origin::UNKNOWN, |id| Origin::plan(PlanId(id)));
        Ok(Arc::new(LogicalPlan::new(node).with_origin(origin)))
    }

    /// The input relation of `message`, which must be set.
    fn input(&mut self, message: &str, input: Option<&Relation>) -> Result<Arc<LogicalPlan>> {
        match input {
            Some(rel) => self.relation(rel),
            None => Err(assertion(format!("{message} needs an input relation"))),
        }
    }

    fn expressions(&mut self, list: &[Expression]) -> Result<Vec<Arc<Expr>>> {
        list.iter().map(|e| self.expression(e)).collect()
    }

    /// `Read`. Option values and paths are classified once, here, by
    /// [`OptionValue::from_user`] (ADR 0038): a credential or a signed URL
    /// never enters the IR as plain text.
    fn read(&mut self, r: &pb::Read) -> Result<PlanNode> {
        if r.is_streaming {
            return Err(unsupported_field("Read", "is_streaming", "streaming reads"));
        }
        let typed = |options: &BTreeMap<String, String>| -> BTreeMap<String, OptionValue> {
            options
                .iter()
                .map(|(k, v)| (k.clone(), OptionValue::from_user(k, v.as_str())))
                .collect()
        };
        let Some(read_type) = &r.read_type else {
            return Err(not_set("spark.connect.Read", "READ_TYPE_NOT_SET"));
        };
        Ok(PlanNode::Read(match read_type {
            ReadType::NamedTable(t) => ReadSource::Table {
                name: parse_multipart_identifier(&t.unparsed_identifier)?,
                options: typed(&t.options),
            },
            ReadType::DataSource(d) => {
                if d.schema.as_deref().is_some_and(|s| !s.is_empty()) {
                    return Err(unsupported_field(
                        "Read.DataSource",
                        "schema",
                        "reads with a user-specified schema",
                    ));
                }
                if !d.predicates.is_empty() {
                    return Err(unsupported_field(
                        "Read.DataSource",
                        "predicates",
                        "reads with predicates",
                    ));
                }
                ReadSource::Files {
                    format: d
                        .format
                        .clone()
                        .unwrap_or_else(|| self.conf.default_data_source.clone()),
                    paths: d
                        .paths
                        .iter()
                        .map(|p| OptionValue::from_user("path", p.as_str()))
                        .collect(),
                    options: typed(&d.options),
                }
            }
        }))
    }

    /// `SQL`. The parameters are, in Spark's order of precedence, the named
    /// expressions, the positional expressions, and the deprecated literal
    /// map and list.
    fn sql(&mut self, s: &pb::Sql) -> Result<PlanNode> {
        #[expect(
            deprecated,
            reason = "Spark still accepts the literal arguments of older clients"
        )]
        let (literal_map, literal_list) = (&s.args, &s.pos_args);
        let args = if !s.named_arguments.is_empty() {
            SqlArgs::Named(
                s.named_arguments
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.expression(v)?)))
                    .collect::<Result<_>>()?,
            )
        } else if !s.pos_arguments.is_empty() {
            SqlArgs::Positional(self.expressions(&s.pos_arguments)?)
        } else if !literal_map.is_empty() {
            SqlArgs::Named(
                literal_map
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.literal_expression(v)?)))
                    .collect::<Result<_>>()?,
            )
        } else if !literal_list.is_empty() {
            SqlArgs::Positional(
                literal_list
                    .iter()
                    .map(|v| self.literal_expression(v))
                    .collect::<Result<_>>()?,
            )
        } else {
            SqlArgs::None
        };
        Ok(PlanNode::Sql {
            query: s.query.clone(),
            args,
        })
    }

    fn literal_expression(&mut self, l: &pb::expression::Literal) -> Result<Arc<Expr>> {
        self.count()?;
        Ok(Arc::new(Expr::new(ExprNode::Literal(
            super::literal::literal(l)?,
        ))))
    }

    fn join(&mut self, j: &pb::Join) -> Result<PlanNode> {
        let (Some(left), Some(right)) = (&j.left, &j.right) else {
            return Err(assertion("Both join sides must be present"));
        };
        if j.join_condition.is_some() && !j.using_columns.is_empty() {
            return Err(
                krishiv_common::error::ConnectInvalidPlanUsingColumnsOrJoinConditionSetInJoin {}
                    .into(),
            );
        }
        if j.join_data_type.is_some() {
            return Err(unsupported_field(
                "Join",
                "join_data_type",
                "joinWith (typed joins)",
            ));
        }
        let kind = match JoinType::try_from(j.join_type) {
            Ok(JoinType::Inner) => JoinKind::Inner,
            Ok(JoinType::FullOuter) => JoinKind::FullOuter,
            Ok(JoinType::LeftOuter) => JoinKind::LeftOuter,
            Ok(JoinType::RightOuter) => JoinKind::RightOuter,
            Ok(JoinType::LeftAnti) => JoinKind::LeftAnti,
            Ok(JoinType::LeftSemi) => JoinKind::LeftSemi,
            Ok(JoinType::Cross) => JoinKind::Cross,
            Ok(JoinType::Unspecified) => {
                return Err(invalid_enum(
                    "spark.connect.Join.JoinType",
                    "JOIN_TYPE_UNSPECIFIED",
                    0,
                ));
            }
            Err(_) => {
                return Err(invalid_enum(
                    "spark.connect.Join.JoinType",
                    "UNRECOGNIZED",
                    j.join_type,
                ));
            }
        };
        let condition = match &j.join_condition {
            Some(c) => JoinCondition::On(self.expression(c)?),
            None if !j.using_columns.is_empty() => JoinCondition::Using(j.using_columns.clone()),
            None => JoinCondition::None,
        };
        Ok(PlanNode::Join {
            left: self.relation(left)?,
            right: self.relation(right)?,
            kind,
            condition,
        })
    }

    /// `Aggregate`. The output is the grouping expressions followed by the
    /// aggregate expressions, as Spark builds it.
    fn aggregate(&mut self, a: &pb::Aggregate) -> Result<PlanNode> {
        let Some(input) = &a.input else {
            return Err(krishiv_common::error::ConnectInvalidPlanAggregateNeedsPlanInput {}.into());
        };
        let input = self.relation(input)?;
        let keys = self.expressions(&a.grouping_expressions)?;
        let aggregates = self.expressions(&a.aggregate_expressions)?;
        let grouping = match GroupType::try_from(a.group_type) {
            Ok(GroupType::Groupby) => Grouping::GroupBy(keys.clone()),
            Ok(GroupType::Rollup) => Grouping::Rollup(keys.clone()),
            Ok(GroupType::Cube) => Grouping::Cube(keys.clone()),
            Ok(GroupType::GroupingSets) => Grouping::GroupingSets {
                exprs: keys.clone(),
                sets: a
                    .grouping_sets
                    .iter()
                    .map(|s| self.expressions(&s.grouping_set))
                    .collect::<Result<_>>()?,
            },
            Ok(GroupType::Pivot) => {
                return Err(unsupported("pivots (spark.connect.Aggregate.pivot)"));
            }
            Ok(GroupType::Unspecified) => {
                return Err(invalid_enum(
                    "spark.connect.Aggregate.GroupType",
                    "GROUP_TYPE_UNSPECIFIED",
                    0,
                ));
            }
            Err(_) => {
                return Err(invalid_enum(
                    "spark.connect.Aggregate.GroupType",
                    "UNRECOGNIZED",
                    a.group_type,
                ));
            }
        };
        Ok(PlanNode::Aggregate {
            input,
            grouping,
            aggregates: keys.into_iter().chain(aggregates).collect(),
        })
    }

    /// `WithColumns`: each alias names exactly one column.
    fn with_columns(&mut self, w: &pb::WithColumns) -> Result<PlanNode> {
        let mut columns = Vec::with_capacity(w.aliases.len());
        for alias in &w.aliases {
            let [name] = alias.name.as_slice() else {
                return Err(
                    krishiv_common::error::ConnectInvalidPlanWithColumnsRequireSingleNamePart {
                        got: format!("[{}]", alias.name.join(", ")),
                    }
                    .into(),
                );
            };
            if alias.metadata.as_deref().is_some_and(|m| !m.is_empty()) {
                return Err(unsupported_field(
                    "Expression.Alias",
                    "metadata",
                    "column aliases with metadata",
                ));
            }
            columns.push((
                name.clone(),
                self.required_expression(alias.expr.as_deref())?,
            ));
        }
        Ok(PlanNode::WithColumns {
            input: self.input("WithColumns", w.input.as_deref())?,
            columns,
        })
    }
}

#[cfg(test)]
mod tests;
