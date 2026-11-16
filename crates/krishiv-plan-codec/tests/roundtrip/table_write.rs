// SPDX-License-Identifier: Apache-2.0
//! `table_write` (P3 PR-37b): a corrupt `TableWrite` node is refused.

use super::*;

#[test]
fn a_corrupt_table_write_is_refused() {
    use pb::node::Kind;
    use pb::table_write_node::Request;
    let edit = |name: &str, f: &dyn Fn(&mut pb::TableWriteNode)| {
        let mut m = message(name);
        for node in &mut m.nodes {
            if let Some(Kind::TableWrite(t)) = &mut node.kind {
                f(t);
            }
        }
        m
    };
    corrupt(&edit("table_write", &|t| t.table = None), "a written table");
    corrupt(
        &edit("table_write", &|t| t.request = None),
        "a write request",
    );
    corrupt(
        &edit("table_write", &|t| on_write(t, |w| w.mode = 0)),
        "an unknown write mode",
    );
    // A filter belongs to an overwrite by filter only, and that mode needs one.
    corrupt(
        &edit("table_write_by_filter", &|t| {
            on_write(t, |w| {
                w.mode = pb::TableWriteMode::Append.into();
            });
        }),
        "a write filter on a mode that takes none",
    );
    corrupt(
        &edit("table_write_by_filter", &|t| {
            on_write(t, |w| w.filter = None);
        }),
        "an overwrite by filter without its filter",
    );
    corrupt(
        &edit("table_write_row_level", &|t| {
            if let Some(Request::RowLevel(r)) = &mut t.request {
                r.op = 0;
            }
        }),
        "an unknown row-level operation",
    );
}

fn on_write(t: &mut pb::TableWriteNode, f: impl Fn(&mut pb::TableWriteRequest)) {
    if let Some(pb::table_write_node::Request::Write(w)) = &mut t.request {
        f(w);
    }
}
