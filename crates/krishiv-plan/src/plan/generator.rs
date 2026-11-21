// SPDX-License-Identifier: Apache-2.0
//! What a [`super::PlanNode::Generate`] produces, and the column it reads.

use crate::{AttrId, Expr};

/// The attribute a resolved generator reads: its argument is always a
/// column of its input, which the generator does not carry to its output
/// (so an exploded array is not copied once per element).
#[must_use]
pub fn generator_arg(arg: &Expr) -> Option<AttrId> {
    match &arg.node {
        crate::ExprNode::Resolved(crate::ResolvedExpr::Column(c)) => Some(c.id),
        _ => None,
    }
}

/// What a [`super::PlanNode::Generate`] produces for each element of its argument
/// (Spark's `Explode`, `PosExplode` and `Inline` generators).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeneratorKind {
    /// `explode`: a row per array element (`col`), or per map entry (`key`,
    /// `value`).
    Explode,
    /// `posexplode`: as `explode` with the 0-based position first (`pos`).
    PosExplode,
    /// `inline`: a row per struct of an array of structs, one column per
    /// field.
    Inline,
    /// `variant_explode`: a row per field of a variant object or element of
    /// a variant array (`pos`, `key`, `value`); any other value, a variant
    /// null and SQL `NULL` give no rows.
    VariantExplode,
    /// A Python UDTF under `LATERAL` (Spark's `Generate(PythonUDTF)`): the
    /// node's `arg` is the call, a [`crate::FunctionId::PythonUdf`] of a
    /// table kind with the UDTF's arguments, not a column; the generated
    /// columns are its declared result.
    PythonUdtf,
}

impl GeneratorKind {
    /// The function name Spark shows for the generator (`explode`,
    /// `posexplode`, `inline`; `_outer` is added by the caller).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Explode => "explode",
            Self::PosExplode => "posexplode",
            Self::Inline => "inline",
            Self::VariantExplode => "variant_explode",
            Self::PythonUdtf => "python_udtf",
        }
    }

    /// The generator a function name calls (ASCII lower case), and whether
    /// it is the `_outer` form; `stack` is not one (it lowers to `inline`).
    #[must_use]
    pub fn of_name(name: &str) -> Option<(Self, bool)> {
        let (base, outer) = match name.strip_suffix("_outer") {
            Some(base) => (base, true),
            None => (name, false),
        };
        let kind = match base {
            "explode" => Self::Explode,
            "posexplode" => Self::PosExplode,
            "inline" => Self::Inline,
            "variant_explode" => Self::VariantExplode,
            _ => return None,
        };
        Some((kind, outer))
    }
}
