// SPDX-License-Identifier: Apache-2.0
//! Default collations the engine does not honour yet (ADR 0043 Decision 7, design
//! `P2-collations.md` section 3.2). The parser accepts the syntax; running such a statement would
//! ignore the default and type its strings as `UTF8_BINARY`, so it is refused instead.

use krishiv_common::error::{
    KrishivError, UnsupportedFeatureCollationsInMapKeys, UnsupportedFeatureSchemaLevelCollations,
};
use krishiv_sql::ast::{AlterTableAction, Statement, TableElement};
use krishiv_types::DataType;

/// Whether `dt` holds a map whose key holds a collated string.
fn collated_map_key(dt: &DataType) -> bool {
    dt.any_type(&|x| matches!(x, DataType::Map(m) if m.key_type().contains_collation()))
}

/// Fails for a default collation or collated map key the engine would silently ignore.
///
/// # Errors
/// `UNSUPPORTED_FEATURE.SCHEMA_LEVEL_COLLATIONS`, `.OBJECT_LEVEL_COLLATIONS` and
/// `.COLLATIONS_IN_MAP_KEYS` as Spark 4.2 raises them with its switches off;
/// `KRISHIV_UNSUPPORTED_FEATURE` for a view default; `COLLATION_INVALID_NAME` for an unknown name.
///
/// With `schema_level` (`spark.sql.collation.schemaLevel.enabled`) a schema's default is stored and
/// shown (`DESCRIBE SCHEMA`); a table or view in such a schema is refused where it is created.
pub(crate) fn refuse_unhonoured(
    statement: &Statement,
    schema_level: bool,
) -> Result<(), KrishivError> {
    match statement {
        Statement::CreateNamespace(n) if n.collation.is_some() && !schema_level => {
            Err(UnsupportedFeatureSchemaLevelCollations {}.into())
        }
        Statement::CreateTable(t) => {
            for element in &t.elements {
                if let TableElement::Column(c) = element
                    && let Ok(dt) = c.data_type.to_data_type()
                    && collated_map_key(&dt)
                {
                    return Err(UnsupportedFeatureCollationsInMapKeys {}.into());
                }
            }
            Ok(())
        }
        Statement::AlterTable(a) => match &a.action {
            AlterTableAction::AddColumns(columns) => {
                for c in columns {
                    if let Ok(dt) = c.data_type.to_data_type()
                        && collated_map_key(&dt)
                    {
                        return Err(UnsupportedFeatureCollationsInMapKeys {}.into());
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        },
        // A permanent view's default is honoured (its text analyses under it): nothing to refuse.
        _ => Ok(()),
    }
}
