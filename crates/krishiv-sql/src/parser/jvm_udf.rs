// SPDX-License-Identifier: Apache-2.0
//! The "did you mean" text of `CREATE FUNCTION name AS 'class' [USING JAR ...]` (P3-T081): a
//! function implemented by a JVM class cannot run here, and the error says what to use instead.
//!
//! The suggestion comes from the class's simple name (`org.apache.hadoop.hive.ql.udf.generic.
//! GenericUDFNvl` is `nvl`) or, failing that, from the name the statement gives the function, matched
//! against the built-in functions that Hive and Spark-community UDFs most often stand for. The
//! list leaves out functions whose results differ from Hive's (`hash` is Murmur3 here); a test
//! checks that each name has a reference page.

/// The built-in functions a JVM UDF is most often a copy of.
const BUILT_INS: &[&str] = &[
    "approx_count_distinct",
    "array_contains",
    "coalesce",
    "collect_list",
    "collect_set",
    "concat_ws",
    "date_add",
    "date_format",
    "date_sub",
    "datediff",
    "explode",
    "from_json",
    "from_unixtime",
    "from_utc_timestamp",
    "get_json_object",
    "hex",
    "if",
    "instr",
    "length",
    "locate",
    "lower",
    "map_keys",
    "map_values",
    "md5",
    "months_between",
    "nvl",
    "nvl2",
    "percentile",
    "pmod",
    "regexp_extract",
    "regexp_replace",
    "rlike",
    "sha2",
    "size",
    "split",
    "str_to_map",
    "substr",
    "to_date",
    "to_json",
    "unhex",
    "unix_timestamp",
    "upper",
    "xxhash64",
];

/// The prefixes Hive puts before a function's own name, longest first.
const PREFIXES: &[&str] = &[
    "genericudaf",
    "genericudtf",
    "genericudf",
    "udaf",
    "udtf",
    "udf",
];

/// The text after `Krishiv cannot load the class` in the error: what to use instead.
pub(crate) fn hint(function: &str, class_name: &str) -> String {
    let simple = class_name.rsplit(['.', '$']).next().unwrap_or(class_name);
    let mut key = normalise(simple);
    let mut hive_named = false;
    if let Some(rest) = PREFIXES.iter().find_map(|p| key.strip_prefix(p)) {
        key = rest.to_owned();
        hive_named = true;
    }
    if let Some(name) = built_in(&key) {
        if hive_named {
            return format!("Use the built-in function `{name}` instead.");
        }
        // A bare simple name such as `Hash` or `Split` may be any package's own class.
        return format!(
            "Krishiv has a built-in function `{name}`: if the class does the same, call it directly instead of registering a class."
        );
    }
    let own = function.rsplit('.').next().unwrap_or(function);
    if let Some(name) = built_in(&normalise(own)) {
        return format!(
            "Krishiv has a built-in function `{name}`: call it directly instead of registering a class."
        );
    }
    "Rewrite it as a built-in function, a SQL function (CREATE FUNCTION ... RETURN), a Rust UDF or a \
     Python UDF."
        .to_owned()
}

/// Lower case with underscores dropped, so `RegExpExtract` and `regexp_extract` meet.
fn normalise(name: &str) -> String {
    name.chars()
        .filter(|c| *c != '_')
        .flat_map(char::to_lowercase)
        .collect()
}

fn built_in(key: &str) -> Option<&'static str> {
    if key.is_empty() {
        return None;
    }
    BUILT_INS.iter().copied().find(|b| normalise(b) == key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hive_class_names_its_built_in() {
        for (class, built_in) in [
            ("org.apache.hadoop.hive.ql.udf.generic.GenericUDFNvl", "nvl"),
            (
                "org.apache.hadoop.hive.ql.udf.UDFRegExpExtract",
                "regexp_extract",
            ),
            (
                "org.apache.hadoop.hive.ql.udf.generic.GenericUDFConcatWS",
                "concat_ws",
            ),
            (
                "org.apache.hadoop.hive.ql.udf.UDFFromUnixTime",
                "from_unixtime",
            ),
            (
                "org.apache.hadoop.hive.ql.udf.generic.GenericUDFDateDiff",
                "datediff",
            ),
        ] {
            let text = hint("f", class);
            assert!(text.contains(&format!("`{built_in}`")), "{class}: {text}");
            assert!(text.starts_with("Use the built-in"), "{class}: {text}");
        }
    }

    #[test]
    fn a_bare_simple_name_is_only_a_suggestion() {
        for class in [
            "com.acme.Split",
            "com.example.Md5",
            "com.example.Outer$Upper",
        ] {
            let text = hint("f", class);
            assert!(
                text.contains("if the class does the same"),
                "{class}: {text}"
            );
            assert!(!text.starts_with("Use the built-in"), "{class}: {text}");
        }
        // Hive's `hash` is not Murmur3: never named as the replacement.
        assert!(
            !hint("f", "org.apache.hadoop.hive.ql.udf.generic.GenericUDFHash").contains("`hash`")
        );
    }

    #[test]
    fn the_function_name_is_the_second_guess() {
        let text = hint("db.nvl", "com.example.MyCoalescer");
        assert!(text.contains("`nvl`"), "{text}");
        assert!(text.contains("call it directly"), "{text}");
    }

    #[test]
    fn an_unknown_class_gets_the_general_advice() {
        let text = hint("my_fn", "com.example.MyFancyThing");
        assert!(text.contains("Rust UDF"), "{text}");
        // An empty simple name matches nothing.
        assert!(hint("my_fn", "com.example.").contains("Rust UDF"));
    }

    #[test]
    fn every_listed_built_in_has_a_reference_page() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/reference/sql/functions");
        if !dir.is_dir() {
            return; // a packaged copy of the crate has no docs tree
        }
        for name in BUILT_INS {
            assert!(
                dir.join(format!("{name}.md")).is_file(),
                "no page for {name}"
            );
        }
    }
}
