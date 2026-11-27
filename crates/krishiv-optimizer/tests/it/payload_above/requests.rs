// SPDX-License-Identifier: Apache-2.0
//! Key-fact requests (design P2b A5 (d), unit 2): where no fact proves the
//! dimension's key, the rule asks for a count of the base relation's columns
//! ([`Statistics::want_unique`]) only once every other gate passed, and fires
//! once the answer is a fact.

use std::sync::Mutex;

use super::*;

/// [`Small`]'s tables with the facts `keys`, recording every request.
#[derive(Debug, Default)]
struct Asking {
    keys: Mutex<Vec<(String, Vec<String>)>>,
    wanted: Mutex<Vec<(String, Vec<String>)>>,
}

impl Asking {
    fn wanted(&self) -> Vec<(String, Vec<String>)> {
        let mut w = self.wanted.lock().unwrap().clone();
        w.sort();
        w.dedup();
        w
    }

    /// Every request answered: the columns are unique.
    fn answer_all(&self) {
        let wanted = self.wanted();
        self.keys.lock().unwrap().extend(wanted);
        self.wanted.lock().unwrap().clear();
    }
}

impl Statistics for Asking {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        Small.table(scan)
    }

    fn unique_keys(&self, scan: &Scan) -> Vec<Vec<String>> {
        let Some(t) = table_name(scan) else {
            return Vec::new();
        };
        self.keys
            .lock()
            .unwrap()
            .iter()
            .filter(|(n, _)| n == t)
            .map(|(_, k)| k.clone())
            .collect()
    }

    fn want_unique(&self, scan: &Scan, columns: &[String]) {
        if let Some(t) = table_name(scan) {
            self.wanted
                .lock()
                .unwrap()
                .push((t.to_owned(), columns.to_vec()));
        }
    }
}

fn want(t: &str, cols: &[&str]) -> (String, Vec<String>) {
    (t.to_owned(), cols.iter().map(|c| (*c).to_owned()).collect())
}

#[test]
fn a_missing_key_is_requested_once_every_other_gate_passes() {
    let stats = Asking::default();
    let plan = resolve(PLAIN);
    let (on, notes) = optimized_alone(&plan, &stats, true);
    let (off, _) = optimized_alone(&plan, &Asking::default(), false);
    assert!(moved(&notes).is_empty(), "{notes:?}");
    assert!(on.structurally_eq(&off), "a request never changes the plan");
    assert_eq!(stats.wanted(), vec![want("d", &["dk"])]);
    assert!(
        notes
            .iter()
            .any(|n| n.contains("awaiting a key count of d [dk]")),
        "{notes:?}"
    );
}

#[test]
fn nothing_is_requested_when_another_gate_refuses() {
    for (sql, why) in [
        // Retention.
        (
            "SELECT dk, dname, sum(fv) FROM f, d WHERE fk = dk AND dp = 'a' GROUP BY dk, dname",
            "retention",
        ),
        // The overflow proof.
        (
            "SELECT dk, dname, sum(fl) FROM f, d WHERE fk = dk GROUP BY dk, dname",
            "overflow",
        ),
        // A call reading the dimension.
        (
            "SELECT dk, dname, sum(dek) FROM f, d WHERE fk = dk GROUP BY dk, dname",
            "reads the dimension",
        ),
        // The payload's width.
        (
            "SELECT dk, dek, sum(fv) FROM f, d WHERE fk = dk GROUP BY dk, dek",
            "payload",
        ),
    ] {
        let stats = Asking::default();
        let (_, notes) = optimized_alone(&resolve(sql), &stats, true);
        assert!(stats.wanted().is_empty(), "{sql}: {:?}", stats.wanted());
        assert!(notes.iter().any(|n| n.contains(why)), "{sql}: {notes:?}");
    }
}

#[test]
fn nothing_is_requested_with_the_switch_off() {
    let stats = Asking::default();
    let (_, notes) = optimized(&resolve(PLAIN), &stats, false);
    assert!(notes.is_empty(), "{notes:?}");
    assert!(stats.wanted().is_empty());
}

#[test]
fn an_answered_request_lets_the_rule_fire_on_the_next_optimization() {
    let stats = Asking::default();
    let plan = resolve(PLAIN);
    let _ = optimized(&plan, &stats, true);
    stats.answer_all();
    let (_, text, notes) = check(&plan, &stats, &dbs(false));
    assert_eq!(moved(&notes).len(), 1, "{notes:?}\n{text}");
    assert!(stats.wanted().is_empty(), "{:?}", stats.wanted());
}

/// The dimension with `e` absorbed (or `d ⋈ e` where join ordering put it
/// below): both relations' keys are requested together, and both answered
/// make the rule fire.
#[test]
fn a_dimension_joined_with_another_relation_requests_both_keys() {
    let sql = "SELECT dk, dname, ename, sum(fv) FROM f, d, e WHERE fk = dk AND dek = ek \
               GROUP BY dk, dname, ename";
    let stats = Asking::default();
    let plan = resolve(sql);
    let (_, notes) = optimized(&plan, &stats, true);
    assert!(moved(&notes).is_empty(), "{notes:?}");
    assert_eq!(
        stats.wanted(),
        vec![want("d", &["dk"]), want("e", &["ek"])],
        "{notes:?}"
    );
    stats.answer_all();
    let (_, text, notes) = check(&plan, &stats, &dbs(false));
    assert_eq!(moved(&notes).len(), 1, "{notes:?}\n{text}");
}

/// One relation's key already a fact: only the other is requested.
#[test]
fn only_the_missing_key_is_requested() {
    let sql = "SELECT dk, dname, ename, sum(fv) FROM f, d, e WHERE fk = dk AND dek = ek \
               GROUP BY dk, dname, ename";
    let stats = Asking::default();
    stats.keys.lock().unwrap().push(want("d", &["dk"]));
    let (_, notes) = optimized(&resolve(sql), &stats, true);
    assert_eq!(stats.wanted(), vec![want("e", &["ek"])], "{notes:?}");
}

#[test]
fn with_eager_aggregation_on_the_key_is_still_requested_and_the_rule_fires_next() {
    // Eager aggregation takes the aggregate while the key is not counted; its
    // probe makes the rule's request, so the next optimization moves the
    // payload and eager aggregation yields to it.
    let stats = Asking::default();
    let plan = resolve(PLAIN);
    let _ = optimized(&plan, &stats, true);
    assert_eq!(stats.wanted(), vec![want("d", &["dk"])]);
    stats.answer_all();
    let (_, notes) = optimized(&plan, &stats, true);
    assert_eq!(moved(&notes).len(), 1, "{notes:?}");
}
