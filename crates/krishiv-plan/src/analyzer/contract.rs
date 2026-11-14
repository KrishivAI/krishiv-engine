// SPDX-License-Identifier: Apache-2.0
//! The contract every catalog meets (design P3-lakehouse §2.1-2.3, P3-T020a): the tests the in-memory catalog, the
//! embedded SQLite catalog, the Hive metastore, Iceberg REST and Unity share, so a catalog differs from another in
//! what it supports and never in how it answers what it does. A catalog's own tests call these functions with a
//! catalog that is empty of the names used here; a failure is a `String` that names the case, never a panic.
//!
//! A write catalog runs [`write_contract`]; a read-only one (Unity in P3, and the read half of any catalog) runs
//! [`read_only_contract`]. Both run [`capabilities_agree`].

use std::future::Future;
use std::pin::Pin;

use super::{
    Catalog, CatalogWrite, ColumnPosition, CreateMode, NameForm, NamespaceChange, NamespaceInfo,
    TableChange, TableColumn, TableIdent, TableInfo, TableMeta,
};
use crate::TableVersion;
use krishiv_types::DataType;

/// The outcome of one case: `Err` names what went wrong.
pub type Check = Result<(), String>;

macro_rules! ensure {
    ($cond:expr, $($why:tt)+) => {
        if !$cond {
            return Err(format!($($why)+));
        }
    };
}

/// A future the cases return so they can be listed.
type Case<'a> = Pin<Box<dyn Future<Output = Check> + 'a>>;

/// The declared capabilities and the write half agree: `writes` is true exactly when there is something to call.
///
/// # Errors
/// The description of the disagreement.
pub async fn capabilities_agree(catalog: &dyn Catalog) -> Check {
    let writes = catalog.capabilities().writes;
    let has_write_half = catalog.as_write().is_some();
    ensure!(
        writes == has_write_half,
        "capabilities().writes is {writes} but as_write() is {}",
        if has_write_half { "Some" } else { "None" }
    );
    Ok(())
}

/// What a catalog that cannot write must answer: it has no write half, so no write method exists that could half-work,
/// and it reports an unknown table as absent, never as an error.
///
/// # Errors
/// The first case that fails.
pub async fn read_only_contract(catalog: &dyn Catalog) -> Check {
    capabilities_agree(catalog).await?;
    ensure!(
        catalog.as_write().is_none() && !catalog.capabilities().writes,
        "a read-only catalog has a write half"
    );
    let missing = TableIdent {
        catalog: catalog.name().to_owned(),
        namespace: vec!["contract_missing".to_owned()],
        name: "nothing".to_owned(),
    };
    let found = catalog
        .load_table(&missing, None, form())
        .await
        .map_err(|e| format!("loading an unknown table is an error, not absence: {e}"))?;
    ensure!(found.is_none(), "an unknown table was found");
    Ok(())
}

fn form() -> NameForm {
    NameForm {
        case_sensitive: false,
        qualified: true,
        temporary_only: false,
    }
}

fn names(meta: &TableMeta) -> Vec<String> {
    meta.columns.iter().map(|c| c.name.clone()).collect()
}

/// What a writable catalog must do, in a namespace of its own named `contract`: namespaces, tables, `alter_table`
/// (in order, all or none, against the version read), `rename_table`, `drop_table`, `drop_namespace`, with the epoch
/// bumped by every change and by nothing else.
///
/// The catalog must not already hold a namespace called `contract`. Cases a catalog declares it does not support
/// (`capabilities()`) are not run.
///
/// # Errors
/// The first case that fails, named.
pub async fn write_contract(catalog: &dyn Catalog) -> Check {
    capabilities_agree(catalog).await?;
    let write = catalog
        .as_write()
        .ok_or_else(|| "a write contract needs a catalog with a write half".to_owned())?;
    let cases: Vec<(&str, Case<'_>)> = vec![
        ("namespaces", Box::pin(namespaces(catalog, write))),
        ("tables", Box::pin(tables(catalog, write))),
        ("alter_table", Box::pin(alter_table(catalog, write))),
        ("rename_table", Box::pin(rename_table(catalog, write))),
        ("drop", Box::pin(drop(catalog, write))),
    ];
    for (name, case) in cases {
        case.await.map_err(|why| format!("{name}: {why}"))?;
    }
    Ok(())
}

fn ns() -> Vec<String> {
    vec!["contract".to_owned()]
}

fn id(catalog: &dyn Catalog, name: &str) -> TableIdent {
    TableIdent {
        catalog: catalog.name().to_owned(),
        namespace: ns(),
        name: name.to_owned(),
    }
}

fn info(catalog: &dyn Catalog, name: &str) -> TableInfo {
    TableInfo::new(
        id(catalog, name),
        vec![
            TableColumn::new("a", DataType::Integer, true),
            TableColumn::new("b", DataType::String, true),
        ],
    )
}

fn add(name: &str) -> TableChange {
    TableChange::AddColumn {
        path: vec![name.to_owned()],
        data_type: DataType::Long,
        declared: None,
        nullable: true,
        comment: None,
        position: None::<ColumnPosition>,
        default: None,
    }
}

async fn load(catalog: &dyn Catalog, name: &str) -> Result<Option<TableMeta>, String> {
    catalog
        .load_table(&id(catalog, name), None, form())
        .await
        .map_err(|e| format!("load_table({name}) failed: {e}"))
}

async fn must_load(catalog: &dyn Catalog, name: &str) -> Result<TableMeta, String> {
    load(catalog, name)
        .await?
        .ok_or_else(|| format!("table {name} is not there"))
}

fn version(meta: &TableMeta) -> Result<TableVersion, String> {
    meta.version.ok_or_else(|| {
        "a table of a writable catalog has a version (alter_table's guard)".to_owned()
    })
}

async fn namespaces(catalog: &dyn Catalog, write: &dyn CatalogWrite) -> Check {
    let before = catalog.epoch();
    write
        .create_namespace(&ns(), NamespaceInfo::default())
        .await
        .map_err(|e| format!("create_namespace failed: {e}"))?;
    ensure!(
        catalog.epoch() > before,
        "creating a namespace did not bump the epoch"
    );
    ensure!(
        write
            .create_namespace(&ns(), NamespaceInfo::default())
            .await
            .is_err(),
        "creating a namespace twice succeeded"
    );
    let epoch = catalog.epoch();
    write
        .alter_namespace(
            &ns(),
            &[NamespaceChange::SetProperty {
                key: "k".to_owned(),
                value: "v".to_owned(),
            }],
        )
        .await
        .map_err(|e| format!("alter_namespace failed: {e}"))?;
    ensure!(
        catalog.epoch() > epoch,
        "altering a namespace did not bump the epoch"
    );
    ensure!(
        write
            .alter_namespace(&["contract_nope".to_owned()], &[])
            .await
            .is_err(),
        "altering a namespace that does not exist succeeded"
    );
    Ok(())
}

async fn tables(catalog: &dyn Catalog, write: &dyn CatalogWrite) -> Check {
    let before = catalog.epoch();
    let created = write
        .create_table(info(catalog, "t"), CreateMode::ErrorIfExists)
        .await
        .map_err(|e| format!("create_table failed: {e}"))?;
    ensure!(created, "create_table reported nothing created");
    ensure!(
        catalog.epoch() > before,
        "creating a table did not bump the epoch"
    );
    let meta = must_load(catalog, "t").await?;
    ensure!(
        names(&meta) == ["a", "b"],
        "the loaded columns are {:?}, not [a, b]",
        names(&meta)
    );
    version(&meta)?;
    ensure!(
        load(catalog, "T").await?.is_some(),
        "a name is matched case-insensitively when the form says so"
    );
    let epoch = catalog.epoch();
    ensure!(
        write
            .create_table(info(catalog, "t"), CreateMode::ErrorIfExists)
            .await
            .is_err(),
        "creating a table that exists succeeded"
    );
    let ignored = write
        .create_table(info(catalog, "t"), CreateMode::IgnoreIfExists)
        .await
        .map_err(|e| format!("IF NOT EXISTS on a table that exists failed: {e}"))?;
    ensure!(
        !ignored,
        "IF NOT EXISTS on a table that exists reported a creation"
    );
    ensure!(
        catalog.epoch() == epoch,
        "a refused or ignored create bumped the epoch"
    );
    Ok(())
}

async fn alter_table(catalog: &dyn Catalog, write: &dyn CatalogWrite) -> Check {
    if !catalog.capabilities().alter_columns {
        return Ok(());
    }
    let table = id(catalog, "t");
    let before = must_load(catalog, "t").await?;
    let v0 = version(&before)?;
    let epoch = catalog.epoch();
    let after = write
        .alter_table(&table, v0, &[add("c")])
        .await
        .map_err(|e| format!("alter_table failed: {e}"))?;
    ensure!(
        names(&after) == ["a", "b", "c"],
        "the altered table has columns {:?}",
        names(&after)
    );
    ensure!(
        catalog.epoch() > epoch,
        "altering a table did not bump the epoch"
    );
    let v1 = version(&must_load(catalog, "t").await?)?;
    ensure!(v1 != v0, "altering a table did not change its version");
    // The version read is the guard: the old one is refused, and nothing changes.
    let epoch = catalog.epoch();
    ensure!(
        write.alter_table(&table, v0, &[add("late")]).await.is_err(),
        "alter_table accepted a version that is no longer current"
    );
    ensure!(catalog.epoch() == epoch, "a refused alter bumped the epoch");
    // All or none: a list whose last change fails leaves the table as it was.
    let current = must_load(catalog, "t").await?;
    ensure!(
        write
            .alter_table(&table, v1, &[add("d"), add("a")])
            .await
            .is_err(),
        "an alter_table whose last change is invalid succeeded"
    );
    let unchanged = must_load(catalog, "t").await?;
    ensure!(
        names(&unchanged) == names(&current) && unchanged.version == current.version,
        "a failed alter_table changed the table: {:?}",
        names(&unchanged)
    );
    ensure!(
        catalog.epoch() == epoch,
        "a failed alter_table bumped the epoch"
    );
    Ok(())
}

async fn rename_table(catalog: &dyn Catalog, write: &dyn CatalogWrite) -> Check {
    let before = must_load(catalog, "t").await?;
    write
        .create_table(info(catalog, "other"), CreateMode::ErrorIfExists)
        .await
        .map_err(|e| format!("create_table failed: {e}"))?;
    let epoch = catalog.epoch();
    ensure!(
        write
            .rename_table(&id(catalog, "t"), &id(catalog, "other"))
            .await
            .is_err(),
        "renaming onto a table that exists succeeded"
    );
    ensure!(
        write
            .rename_table(&id(catalog, "nope"), &id(catalog, "x"))
            .await
            .is_err(),
        "renaming a table that does not exist succeeded"
    );
    ensure!(
        catalog.epoch() == epoch,
        "a refused rename bumped the epoch"
    );
    write
        .rename_table(&id(catalog, "t"), &id(catalog, "renamed"))
        .await
        .map_err(|e| format!("rename_table failed: {e}"))?;
    ensure!(
        catalog.epoch() > epoch,
        "renaming a table did not bump the epoch"
    );
    ensure!(
        load(catalog, "t").await?.is_none(),
        "the old name is still there"
    );
    let after = must_load(catalog, "renamed").await?;
    ensure!(
        names(&after) == names(&before),
        "the renamed table has columns {:?}, not {:?}",
        names(&after),
        names(&before)
    );
    Ok(())
}

async fn drop(catalog: &dyn Catalog, write: &dyn CatalogWrite) -> Check {
    let epoch = catalog.epoch();
    ensure!(
        write
            .drop_table(&id(catalog, "renamed"), false)
            .await
            .map_err(|e| format!("drop_table failed: {e}"))?,
        "drop_table reported that a table that exists did not"
    );
    ensure!(
        catalog.epoch() > epoch,
        "dropping a table did not bump the epoch"
    );
    ensure!(
        load(catalog, "renamed").await?.is_none(),
        "a dropped table loads"
    );
    let epoch = catalog.epoch();
    ensure!(
        !write
            .drop_table(&id(catalog, "renamed"), false)
            .await
            .map_err(|e| format!("dropping a table twice failed: {e}"))?,
        "dropping a table that is gone reported a drop"
    );
    ensure!(
        catalog.epoch() == epoch,
        "dropping nothing bumped the epoch"
    );
    // A namespace that holds a table is not dropped without CASCADE; with it, the tables go too.
    ensure!(
        write.drop_namespace(&ns(), false).await.is_err(),
        "dropping a namespace that holds a table succeeded without CASCADE"
    );
    write
        .drop_namespace(&ns(), true)
        .await
        .map_err(|e| format!("drop_namespace with CASCADE failed: {e}"))?;
    ensure!(
        load(catalog, "other").await?.is_none(),
        "a table of a dropped namespace loads"
    );
    ensure!(
        write.drop_namespace(&ns(), true).await.is_err(),
        "dropping a namespace that is gone succeeded"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::MemoryCatalog;
    use crate::analyzer::testing::block_on;

    #[test]
    fn a_read_only_catalog_meets_the_read_only_contract() {
        let catalog = MemoryCatalog::new("c");
        block_on(read_only_contract(&catalog)).unwrap();
        block_on(capabilities_agree(&catalog)).unwrap();
    }

    #[test]
    fn the_write_contract_refuses_a_catalog_with_no_write_half() {
        let catalog = MemoryCatalog::new("c");
        let why = block_on(write_contract(&catalog)).unwrap_err();
        assert!(why.contains("needs a catalog with a write half"), "{why}");
    }
}
