// SPDX-License-Identifier: Apache-2.0
//! Test support for analyzer tests and benchmarks (feature `testing`): a
//! one-catalog resolver, a context builder (with the production built-in
//! functions) and a single-threaded `block_on`.

use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use super::{
    AllowAll, AnalysisContext, BuiltinFunctions, BuiltinTableFunctions, Catalog, CatalogResolver,
    MemoryCatalog, Principal, PrincipalKind, ResolutionConf, SparkProfile, VariableSnapshot,
};

/// One catalog, the current one, and the current namespace.
pub struct OneCatalog(pub Arc<dyn Catalog>, pub Vec<String>);

impl std::fmt::Debug for OneCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OneCatalog({}, {:?})", self.0.name(), self.1)
    }
}

impl CatalogResolver for OneCatalog {
    fn current_catalog(&self) -> &str {
        self.0.name()
    }

    fn current_namespace(&self) -> &[String] {
        &self.1
    }

    fn epoch(&self) -> u64 {
        0
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn Catalog>> {
        (name.eq_ignore_ascii_case(self.0.name())).then(|| Arc::clone(&self.0))
    }
}

/// A context over `catalog` (as `spark_catalog.default`), allowing all.
#[must_use]
pub fn context(catalog: MemoryCatalog, conf: ResolutionConf) -> AnalysisContext {
    AnalysisContext {
        profile: SparkProfile::default(),
        conf,
        principal: Arc::new(Principal::new("tester", PrincipalKind::User)),
        policy: Arc::new(AllowAll),
        catalogs: Arc::new(OneCatalog(Arc::new(catalog), vec!["default".to_owned()])),
        functions: Arc::new(BuiltinFunctions::new()),
        table_functions: Arc::new(BuiltinTableFunctions::new()),
        sql: None,
        files: None,
        variables: Arc::new(VariableSnapshot::default()),
    }
}

/// Runs a future that never waits on I/O to completion on this thread.
pub fn block_on<F: Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
    }
}
