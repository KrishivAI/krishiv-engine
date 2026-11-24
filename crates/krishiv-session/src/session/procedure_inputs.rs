// SPDX-License-Identifier: Apache-2.0
//! The inputs a catalog procedure reads and writes tables through (`CALL`).

use std::sync::Arc;

use krishiv_exec::files::ObjectPaths;

use super::{LOCAL_STORE, Session, StoreId};

impl Session {
    /// What a catalog's procedure reads and writes tables through: the query's I/O, its path
    /// placement and its configuration snapshot (`CALL`, design P3-T027 PR-36e §3).
    pub(crate) fn procedure_inputs(&self) -> crate::catalog_registry::ScanInputs {
        let (config, _) = self.state.conf.query_snapshot();
        let stores = self.scoped_stores(&config);
        let io = self.scoped_io();
        let paths: Arc<dyn krishiv_exec::files::PathLocator> = Arc::new(ObjectPaths::new(
            StoreId::new(LOCAL_STORE),
            self.state.cwd.clone(),
            Arc::clone(&self.state.io),
            stores,
        ));
        let declared = self.declared_stores(&config);
        crate::catalog_registry::ScanInputs::new(io, paths, config, 1, declared)
    }
}
