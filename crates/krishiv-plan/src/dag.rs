// SPDX-License-Identifier: Apache-2.0
//! Equality, `Debug` and rewrites that work per distinct node.
//!
//! Plans and expressions share subtrees through `Arc`, so they are DAGs:
//! `p = p.join(p)` 30 times is 31 relations, but 2^31 - 1 as a tree. A
//! derived `PartialEq` or `Debug`, or a rewrite that visits every path,
//! would take time exponential in the depth of sharing. Here:
//!
//! - `PartialEq` returns early on the same `Arc` and remembers, for the
//!   outermost comparison running on the thread, the pairs of nodes found
//!   equal, so two DAGs of the same shape compare in time linear in their
//!   distinct nodes.
//! - `Debug` prints a node reached along several paths once, labelled
//!   `#n`, and `#n (shared, printed above)` at its later places, as the
//!   pretty-printer does.
//! - [`Memo`] makes `transform_up` / `transform_down` rewrite a shared node
//!   once and share the result.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Formatter};
use std::ops::ControlFlow;
use std::sync::Arc;
use std::thread::LocalKey;

use crate::walk::{Node, children, visit_distinct};

/// The rewrite of each node already rewritten, by address. It holds the
/// original `Arc` too, so no address is reused while the rewrite runs.
pub(crate) type Memo<T> = HashMap<*const T, (Arc<T>, Arc<T>)>;

type Pair = (*const (), *const ());

thread_local! {
    /// The node pairs found equal by the outermost comparison running on
    /// this thread; `None` when none runs.
    static EQUAL: RefCell<Option<HashSet<Pair>>> = const { RefCell::new(None) };
    /// The shared-node labels of the outermost `Debug` running on this
    /// thread.
    static SHOWN: RefCell<Option<Labels>> = const { RefCell::new(None) };
}

/// Clears a thread-local when the outermost call that set it returns.
struct Reset<T: 'static>(&'static LocalKey<RefCell<Option<T>>>);

impl<T> Drop for Reset<T> {
    fn drop(&mut self) {
        self.0.with_borrow_mut(|v| *v = None);
    }
}

/// `a == b` for two nodes of one kind, where `fields_eq` compares their
/// fields (and, through them, their children).
pub(crate) fn eq(a: Node<'_>, b: Node<'_>, fields_eq: impl FnOnce() -> bool) -> bool {
    let pair = (a.key(), b.key());
    if pair.0 == pair.1 {
        return true;
    }
    let (known, outermost) = EQUAL.with_borrow_mut(|memo| match memo {
        Some(set) => (set.contains(&pair), false),
        None => {
            *memo = Some(HashSet::new());
            (false, true)
        }
    });
    if known {
        return true;
    }
    let _reset = outermost.then(|| Reset(&EQUAL));
    let equal = fields_eq();
    if equal {
        EQUAL.with_borrow_mut(|memo| {
            if let Some(set) = memo {
                set.insert(pair);
            }
        });
    }
    equal
}

/// The label of every node reached along more than one path, numbered in
/// pre-order, and the labels printed so far.
struct Labels {
    ids: HashMap<*const (), usize>,
    printed: HashSet<*const ()>,
}

impl Labels {
    fn of(root: Node<'_>) -> Self {
        let mut order = Vec::new();
        let mut parents: HashMap<*const (), usize> = HashMap::new();
        let mut next = Vec::new();
        let _ = visit_distinct(root, |node| {
            order.push(node.key());
            children(node, &mut next);
            for child in next.drain(..) {
                *parents.entry(child.key()).or_default() += 1;
            }
            ControlFlow::Continue(())
        });
        let ids = order
            .into_iter()
            .filter(|k| parents.get(k).is_some_and(|&n| n > 1))
            .zip(1..)
            .collect();
        Self {
            ids,
            printed: HashSet::new(),
        }
    }
}

/// `Debug` of `node` named `name`: `body(f, name)` prints its fields, with
/// the name labelled when the node is shared; a shared node already printed
/// prints as a back reference.
pub(crate) fn debug(
    node: Node<'_>,
    f: &mut Formatter<'_>,
    name: &str,
    body: impl FnOnce(&mut Formatter<'_>, &str) -> fmt::Result,
) -> fmt::Result {
    let outermost = SHOWN.with_borrow_mut(|s| {
        let start = s.is_none();
        if start {
            *s = Some(Labels::of(node));
        }
        start
    });
    let _reset = outermost.then(|| Reset(&SHOWN));
    let label = SHOWN.with_borrow_mut(|s| {
        let labels = s.as_mut()?;
        let id = *labels.ids.get(&node.key())?;
        Some((id, labels.printed.insert(node.key())))
    });
    match label {
        None => body(f, name),
        Some((id, true)) => body(f, &format!("{name} #{id}")),
        Some((id, false)) => write!(f, "{name} #{id} (shared, printed above)"),
    }
}
