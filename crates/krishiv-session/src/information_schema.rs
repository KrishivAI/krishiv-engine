// SPDX-License-Identifier: Apache-2.0
//! `information_schema.schemata`, `.tables` and `.columns`: read-only views over the session
//! catalog (`spark_catalog`) for BI tools and SQL clients. Spark has no `information_schema`, so the
//! columns are the SQL standard's names. Rows follow the listing rule of `SHOW TABLES` /
//! `SHOW NAMESPACES` ([`crate::access`] `may_list`): a relation or schema the principal may not list
//! has no row. Temporary views are not listed (they belong to no schema); a declared catalog's own
//! `information_schema` is not built yet.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::planner::MemoryTable;
use krishiv_plan::analyzer::{AccessPolicy, Principal, Privilege, Securable, TableColumn};
use krishiv_types::DataType;
use krishiv_types::batch::Batch;

use crate::store::{CatalogStore, CatalogTable, InMemoryCatalogStore};
use crate::system::text;

/// The namespace the views live in.
pub(crate) const NAMESPACE: &str = "information_schema";

/// What lists the catalog for one query: the store, and the principal's listing rules.
#[derive(Clone)]
pub(crate) struct Source {
    pub(crate) store: Arc<InMemoryCatalogStore>,
    pub(crate) principal: Arc<Principal>,
    pub(crate) policy: Arc<dyn AccessPolicy>,
    pub(crate) grants: Option<Arc<crate::grants::GrantsPolicy>>,
}

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Source").finish_non_exhaustive()
    }
}

/// Whether `name` is one of the views.
pub(crate) fn is_view(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "schemata" | "tables" | "columns"
    )
}

/// The columns of the view `name`: `(column, nullable, integer)`.
fn layout(name: &str) -> Vec<(&'static str, bool, bool)> {
    match name {
        "schemata" => vec![
            ("catalog_name", false, false),
            ("schema_name", false, false),
        ],
        "tables" => vec![
            ("table_catalog", false, false),
            ("table_schema", false, false),
            ("table_name", false, false),
            ("table_type", false, false),
        ],
        _ => vec![
            ("table_catalog", false, false),
            ("table_schema", false, false),
            ("table_name", false, false),
            ("column_name", false, false),
            ("ordinal_position", false, true),
            ("data_type", false, false),
            ("is_nullable", false, false),
        ],
    }
}

/// The view's columns as the analyzer sees them.
pub(crate) fn columns(name: &str) -> Option<Vec<TableColumn>> {
    let name = name.to_ascii_lowercase();
    is_view(&name).then(|| {
        layout(&name)
            .into_iter()
            .map(|(column, nullable, integer)| TableColumn {
                name: column.to_owned(),
                data_type: if integer {
                    DataType::Integer
                } else {
                    DataType::String
                },
                nullable,
                char_varchar: None,
            })
            .collect()
    })
}

impl Source {
    fn listable(&self, securable: &Securable) -> Option<bool> {
        self.grants
            .as_ref()
            .and_then(|g| g.listable(&self.principal, securable))
    }

    async fn may_list(&self, securable: &Securable) -> bool {
        if let Some(listed) = self.listable(securable) {
            return listed;
        }
        krishiv_plan::analyzer::allowed(
            &*self.policy,
            &self.principal,
            securable,
            Privilege::Browse,
        )
        .await
    }

    /// The relations of the catalog the principal may list, with their namespace.
    async fn relations(&self) -> (Vec<String>, Vec<Arc<CatalogTable>>) {
        let mut schemas = Vec::new();
        let mut relations = Vec::new();
        for (namespace, _) in CatalogStore::namespaces(&*self.store) {
            let mut shown = Vec::new();
            for t in CatalogStore::tables_in(&*self.store, &namespace) {
                if self
                    .may_list(&crate::session::Session::relation_securable(&t))
                    .await
                {
                    shown.push(t);
                }
            }
            let namespace_securable =
                Securable::Namespace(vec![crate::catalog::CATALOG.to_owned(), namespace.clone()]);
            let mut listed = !shown.is_empty() || self.may_list(&namespace_securable).await;
            if !listed {
                for name in CatalogStore::functions(&*self.store).in_namespace(&namespace) {
                    let securable = Securable::Function(vec![
                        crate::catalog::CATALOG.to_owned(),
                        namespace.clone(),
                        name,
                    ]);
                    if self.may_list(&securable).await {
                        listed = true;
                        break;
                    }
                }
            }
            if listed {
                schemas.push(namespace);
            }
            relations.extend(shown);
        }
        schemas.sort();
        relations.sort_by(|a, b| (&a.namespace, &a.name).cmp(&(&b.namespace, &b.name)));
        (schemas, relations)
    }

    /// The view `name` with its rows now, or `None` for an unknown name.
    pub(crate) async fn provider(
        &self,
        name: &str,
    ) -> Result<Option<(Vec<TableColumn>, Arc<MemoryTable>)>, KrishivError> {
        let name = name.to_ascii_lowercase();
        let Some(table_columns) = columns(&name) else {
            return Ok(None);
        };
        let (schemas, relations) = self.relations().await;
        let catalog = crate::catalog::CATALOG;
        let values: Vec<ArrayRef> = match name.as_str() {
            "schemata" => vec![
                text(schemas.iter().map(|_| Some(catalog))),
                text(schemas.iter().map(|s| Some(s.as_str()))),
            ],
            "tables" => vec![
                text(relations.iter().map(|_| Some(catalog))),
                text(relations.iter().map(|t| Some(t.namespace.as_str()))),
                text(relations.iter().map(|t| Some(t.name.as_str()))),
                text(relations.iter().map(|t| {
                    Some(if t.view.is_some() {
                        "VIEW"
                    } else {
                        "BASE TABLE"
                    })
                })),
            ],
            _ => {
                let rows: Vec<(&CatalogTable, usize)> = relations
                    .iter()
                    .flat_map(|t| (0..t.columns.len()).map(move |i| (&**t, i)))
                    .collect();
                let types: Vec<String> = rows
                    .iter()
                    .map(|(t, i)| t.columns[*i].data_type.simple_string())
                    .collect();
                vec![
                    text(rows.iter().map(|_| Some(catalog))),
                    text(rows.iter().map(|(t, _)| Some(t.namespace.as_str()))),
                    text(rows.iter().map(|(t, _)| Some(t.name.as_str()))),
                    text(rows.iter().map(|(t, i)| Some(t.columns[*i].name.as_str()))),
                    Arc::new(Int32Array::from_iter(
                        rows.iter()
                            .map(|(_, i)| Some(i32::try_from(*i + 1).unwrap_or(i32::MAX))),
                    )),
                    text(types.iter().map(|s| Some(s.as_str()))),
                    text(
                        rows.iter()
                            .map(|(t, i)| Some(if t.columns[*i].nullable { "YES" } else { "NO" })),
                    ),
                ]
            }
        };
        let plain = Arc::new(Schema::new(
            layout(&name)
                .into_iter()
                .map(|(column, nullable, integer)| {
                    Field::new(
                        column,
                        if integer {
                            ArrowType::Int32
                        } else {
                            ArrowType::Utf8
                        },
                        nullable,
                    )
                })
                .collect::<Vec<_>>(),
        ));
        let schema = crate::carrier::canonical_schema(Arc::clone(&plain));
        let batch = RecordBatch::try_new(plain, values).map_err(|e| InternalError {
            message: format!("information_schema.{name}: {e}"),
        })?;
        let batch = crate::carrier::recarried(&batch, &schema)?;
        Ok(Some((
            table_columns,
            Arc::new(MemoryTable::new(schema, vec![Batch::new(batch)])),
        )))
    }
}
