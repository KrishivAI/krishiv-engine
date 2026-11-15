// SPDX-License-Identifier: Apache-2.0
//! The tables of a withheld explain (P3 PR-14d, design `P3-PR-14d-metadata-explain.md` D2): every
//! provider answers only its scan, so the optimizer and the planner see no statistics, no
//! row-group counts and no other hint taken from the data. The plan they choose is the one for
//! tables never analyzed, whatever the tables hold.

use std::sync::Arc;

use krishiv_exec::ScanPushdown;
use krishiv_exec::planner::{TableProvider, TableProviderResolver};
use krishiv_plan::Scan;

/// `tables` with every hint withheld.
pub(crate) fn resolver(tables: Arc<dyn TableProviderResolver>) -> Arc<dyn TableProviderResolver> {
    Arc::new(Withheld(tables))
}

#[derive(Debug)]
struct Withheld(Arc<dyn TableProviderResolver>);

impl TableProviderResolver for Withheld {
    fn provider(&self, scan: &Scan) -> krishiv_exec::Result<Arc<dyn TableProvider>> {
        Ok(Arc::new(ScanOnly(self.0.provider(scan)?)))
    }
}

/// A provider that delegates only [`TableProvider::scan`]: every other method answers the
/// trait's default (`None`, no write modes), so nothing but the scan reaches the planner.
#[derive(Debug)]
struct ScanOnly(Arc<dyn TableProvider>);

impl TableProvider for ScanOnly {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        self.0.scan()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_schema::{DataType, Field, Schema};
    use krishiv_exec::planner::MemoryTable;

    #[derive(Debug)]
    struct One(Arc<dyn TableProvider>);

    impl TableProviderResolver for One {
        fn provider(&self, _: &Scan) -> krishiv_exec::Result<Arc<dyn TableProvider>> {
            Ok(Arc::clone(&self.0))
        }
    }

    #[derive(Debug)]
    struct Units(Arc<dyn TableProvider>);

    impl TableProvider for Units {
        fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
            self.0.scan()
        }

        fn scan_units(&self) -> Option<u64> {
            Some(7)
        }
    }

    #[test]
    fn only_the_scan_reaches_the_planner() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let table: Arc<dyn TableProvider> = Arc::new(MemoryTable::new(schema, Vec::new()));
        let scan = Scan {
            source: krishiv_plan::ReadSource::Table {
                name: vec!["t".to_owned()],
                options: std::collections::BTreeMap::new(),
            },
            output: Vec::new(),
            version: None,
        };
        assert!(
            table.statistics().is_some(),
            "the table itself knows its row count"
        );
        let withheld = resolver(Arc::new(One(table)));
        let provider = withheld.provider(&scan).expect("a provider");
        assert!(provider.statistics().is_none());
        assert!(provider.scan_units().is_none());
        // A provider that knows its row groups: they are withheld too (review m1).
        let units = resolver(Arc::new(One(Arc::new(Units(Arc::clone(&provider))))));
        assert_eq!(Units(Arc::clone(&provider)).scan_units(), Some(7));
        assert!(
            units
                .provider(&scan)
                .expect("a provider")
                .scan_units()
                .is_none()
        );
        assert_eq!(
            provider
                .scan()
                .expect("a scan")
                .table_schema()
                .fields()
                .len(),
            1
        );
    }
}
