// SPDX-License-Identifier: Apache-2.0
//! Default collations the engine does not honour yet (ADR 0043 Decision 7, design
//! `P2-collations.md` section 3.2). The parser accepts the syntax; running such a statement would
//! ignore the default and type its strings as `UTF8_BINARY`, so it is refused instead.

use krishiv_common::error::{
    KrishivError, KrishivUnsupportedFeature, UnsupportedFeatureCollationsInMapKeys,
    UnsupportedFeatureObjectLevelCollations, UnsupportedFeatureSchemaLevelCollations,
};
use krishiv_sql::ast::{Statement, TableElement};
use krishiv_types::{Collation, DataType};

/// A view's `DEFAULT COLLATION`: `UTF8_BINARY` is the plain default and changes nothing; any other
/// needs the view body analysed under it (a later unit).
fn view_default(parts: &[&str]) -> Result<(), KrishivError> {
    let c = Collation::parse(&parts.join("."))?;
    if c.is_utf8_binary() {
        return Ok(());
    }
    Err(KrishivUnsupportedFeature {
        feature: "DEFAULT COLLATION on a view".to_owned(),
    }
    .into())
}

/// Fails for a default collation or collated map key the engine would silently ignore.
///
/// # Errors
/// `UNSUPPORTED_FEATURE.SCHEMA_LEVEL_COLLATIONS`, `.OBJECT_LEVEL_COLLATIONS` and
/// `.COLLATIONS_IN_MAP_KEYS` as Spark 4.2 raises them with its switches off;
/// `KRISHIV_UNSUPPORTED_FEATURE` for a view default; `COLLATION_INVALID_NAME` for an unknown name.
pub(crate) fn refuse_unhonoured(statement: &Statement) -> Result<(), KrishivError> {
    match statement {
        Statement::CreateNamespace(n) if n.collation.is_some() => {
            Err(UnsupportedFeatureSchemaLevelCollations {}.into())
        }
        Statement::CreateTable(t) => {
            if t.collation.is_some() {
                return Err(UnsupportedFeatureObjectLevelCollations {}.into());
            }
            for element in &t.elements {
                if let TableElement::Column(c) = element
                    && let Ok(dt) = c.data_type.to_data_type()
                    && dt.any_type(
                        &|x| matches!(x, DataType::Map(m) if m.key_type().contains_collation()),
                    )
                {
                    return Err(UnsupportedFeatureCollationsInMapKeys {}.into());
                }
            }
            Ok(())
        }
        Statement::CreateView(v) => match &v.collation {
            Some(c) => view_default(&[c.value.as_str()]),
            None => Ok(()),
        },
        Statement::CreatePermanentView(v) => match &v.collation {
            Some(c) => view_default(&c.parts.iter().map(|p| p.value.as_str()).collect::<Vec<_>>()),
            None => Ok(()),
        },
        _ => Ok(()),
    }
}
