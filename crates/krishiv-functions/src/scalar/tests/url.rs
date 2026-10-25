// SPDX-License-Identifier: Apache-2.0
//! The URL bindings: `url_encode`, `url_decode`/`try_url_decode` and
//! `parse_url`/`try_parse_url`, checked against Spark 4.2's goldens, the
//! function-matrix fixtures and `java.net.URI` runs.

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use krishiv_common::error::ErrorClass;

use super::{
    BindArg, BindOptions, bind, bind_scalar, bind_with, class, col, lit, owned, strs, text,
};

/// The golden URL: userinfo, path, query and fragment.
const URL: &str = "http://userinfo@spark.apache.org/path?query=1#Ref";

fn url() -> ArrayRef {
    strs(&[Some(URL)])
}

fn part(p: &str) -> ArrayRef {
    strs(&[Some(p)])
}

fn legacy() -> BindOptions {
    BindOptions::new().with_ansi(false)
}

/// `parse_url(url, part)` over one row.
fn parse(url: &str, part: &str) -> Vec<Option<String>> {
    let (u, p) = (strs(&[Some(url)]), strs(&[Some(part)]));
    let f = bind("parse_url", &[&u, &p]);
    assert_eq!(f.return_type(), &DataType::Utf8);
    text(&f.invoke(&[col(&u), col(&p)], 1).unwrap())
}

#[test]
fn url_encode_follows_java_url_encoder() {
    // The goldens.
    assert_eq!(
        text(
            &bind("url_encode", &[&strs(&[Some("https://spark.apache.org")])])
                .invoke(&[col(&strs(&[Some("https://spark.apache.org")]))], 1)
                .unwrap()
        ),
        owned(&[Some("https%3A%2F%2Fspark.apache.org")])
    );
    let raw = strs(&[Some("inva lid://user:pass@host/file\\;param?query\\;p2")]);
    assert_eq!(
        text(&bind("url_encode", &[&raw]).invoke(&[col(&raw)], 1).unwrap()),
        owned(&[Some(
            "inva+lid%3A%2F%2Fuser%3Apass%40host%2Ffile%5C%3Bparam%3Fquery%5C%3Bp2"
        )])
    );
    // Unreserved bytes pass through, `~` and `+` do not.
    let kept = strs(&[Some("AZaz09.-*_")]);
    assert_eq!(
        text(
            &bind("url_encode", &[&kept])
                .invoke(&[col(&kept)], 1)
                .unwrap()
        ),
        owned(&[Some("AZaz09.-*_")])
    );
    let tilde = strs(&[Some("a~b+c")]);
    assert_eq!(
        text(
            &bind("url_encode", &[&tilde])
                .invoke(&[col(&tilde)], 1)
                .unwrap()
        ),
        owned(&[Some("a%7Eb%2Bc")])
    );
    // Non-ASCII encodes as uppercase hex of the UTF-8 bytes (the
    // fixture's trailing acute is `e` plus a combining mark).
    let uni = strs(&[Some("Ä€日本語😀e\u{301}")]);
    assert_eq!(
        text(&bind("url_encode", &[&uni]).invoke(&[col(&uni)], 1).unwrap()),
        owned(&[Some(
            "%C3%84%E2%82%AC%E6%97%A5%E6%9C%AC%E8%AA%9E%F0%9F%98%80e%CC%81"
        )])
    );
    let ws = strs(&[Some(" \t a\u{3000}\n ")]);
    assert_eq!(
        text(&bind("url_encode", &[&ws]).invoke(&[col(&ws)], 1).unwrap()),
        owned(&[Some("+%09+a%E3%80%80%0A+")])
    );
    // Empty and NULL.
    let empty = strs(&[Some("")]);
    assert_eq!(
        text(
            &bind("url_encode", &[&empty])
                .invoke(&[col(&empty)], 1)
                .unwrap()
        ),
        owned(&[Some("")])
    );
    let null = strs(&[None]);
    assert_eq!(
        text(
            &bind("url_encode", &[&null])
                .invoke(&[col(&null)], 1)
                .unwrap()
        ),
        owned(&[None])
    );
    // Names are case-insensitive, results are plain `Utf8`.
    let f = bind("URL_ENCODE", &[&empty]);
    assert_eq!(f.name(), "url_encode");
    assert_eq!(f.return_type(), &DataType::Utf8);
}

#[test]
fn url_encode_rejects_other_shapes() {
    for args in [
        vec![],
        vec![DataType::Int32],
        vec![DataType::Utf8, DataType::Utf8],
    ] {
        let bound: Vec<BindArg> = args.into_iter().map(BindArg::from).collect();
        let err = bind_scalar("url_encode", &bound, &BindOptions::new()).unwrap_err();
        assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
    }
    for name in ["url_decode", "try_url_decode"] {
        let err = bind_scalar(name, &[], &BindOptions::new()).unwrap_err();
        assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature, "{name}");
    }
}

#[test]
fn url_decode_follows_java_url_decoder() {
    let encoded = strs(&[Some("https%3A%2F%2Fspark.apache.org")]);
    let f = bind("url_decode", &[&encoded]);
    assert_eq!(f.return_type(), &DataType::Utf8);
    assert_eq!(
        text(&f.invoke(&[col(&encoded)], 1).unwrap()),
        owned(&[Some("https://spark.apache.org")])
    );
    // `+` reads as a space, `%2B` as a plus.
    let plus = strs(&[Some("a+b%2Bc")]);
    assert_eq!(
        text(
            &bind("url_decode", &[&plus])
                .invoke(&[col(&plus)], 1)
                .unwrap()
        ),
        owned(&[Some("a b+c")])
    );
    // Lowercase hex decodes too.
    let lower = strs(&[Some("a%2fb%2Fc")]);
    assert_eq!(
        text(
            &bind("url_decode", &[&lower])
                .invoke(&[col(&lower)], 1)
                .unwrap()
        ),
        owned(&[Some("a/b/c")])
    );
    // Bytes that are not valid UTF-8 after decoding read with replacement.
    let bad_utf8 = strs(&[Some("%FF%C3%28")]);
    assert_eq!(
        text(
            &bind("url_decode", &[&bad_utf8])
                .invoke(&[col(&bad_utf8)], 1)
                .unwrap()
        ),
        owned(&[Some("\u{FFFD}\u{FFFD}(")])
    );
    // Empty and NULL pass through.
    let empty = strs(&[Some("")]);
    assert_eq!(
        text(
            &bind("url_decode", &[&empty])
                .invoke(&[col(&empty)], 1)
                .unwrap()
        ),
        owned(&[Some("")])
    );
    let null = strs(&[None]);
    assert_eq!(
        text(
            &bind("url_decode", &[&null])
                .invoke(&[col(&null)], 1)
                .unwrap()
        ),
        owned(&[None])
    );
    // A NULL row among values decodes the rest.
    let mixed = strs(&[Some("a%20b"), None]);
    assert_eq!(
        text(
            &bind("url_decode", &[&mixed])
                .invoke(&[col(&mixed)], 2)
                .unwrap()
        ),
        owned(&[Some("a b"), None])
    );
}

#[test]
fn url_decode_rejects_malformed_escapes() {
    for bad in [
        "%",
        "%2",
        "a%",
        "%zz",
        "http%3A%2F%2spark.apache.org",
        "%-1",
    ] {
        let s = strs(&[Some(bad)]);
        let err = bind("url_decode", &[&s]).invoke(&[col(&s)], 1).unwrap_err();
        assert_eq!(class(&err), ErrorClass::CannotDecodeUrl, "{bad}");
        assert!(err.to_string().contains(bad), "{bad}: {err}");
        // `try_url_decode` is NULL wherever decoding fails, in any mode.
        for opts in [BindOptions::new(), legacy()] {
            let f = bind_with("try_url_decode", &[&s], &opts);
            assert_eq!(
                text(&f.invoke(&[col(&s)], 1).unwrap()),
                owned(&[None]),
                "{bad}"
            );
        }
    }
    // `try_url_decode` decodes what decodes.
    let good = strs(&[Some("https%3A%2F%2Fspark.apache.org")]);
    assert_eq!(
        text(
            &bind("try_url_decode", &[&good])
                .invoke(&[col(&good)], 1)
                .unwrap()
        ),
        owned(&[Some("https://spark.apache.org")])
    );
    let plain = strs(&[Some("inva lid://user:pass@host/file\\;param?query\\;p2")]);
    assert_eq!(
        text(
            &bind("TRY_URL_DECODE", &[&plain])
                .invoke(&[col(&plain)], 1)
                .unwrap()
        ),
        owned(&[Some("inva lid://user:pass@host/file\\;param?query\\;p2")])
    );
}

#[test]
fn parse_url_reads_every_part() {
    for (part, want) in [
        ("HOST", Some("spark.apache.org")),
        ("PATH", Some("/path")),
        ("QUERY", Some("query=1")),
        ("REF", Some("Ref")),
        ("PROTOCOL", Some("http")),
        ("FILE", Some("/path?query=1")),
        ("AUTHORITY", Some("userinfo@spark.apache.org")),
        ("USERINFO", Some("userinfo")),
    ] {
        assert_eq!(parse(URL, part), owned(&[want]), "{part}");
    }
    // With a port, a password, a query and a fragment.
    let full = "http://user:pass@h:8080/p?q=1#frag";
    for (part, want) in [
        ("HOST", Some("h")),
        ("PATH", Some("/p")),
        ("QUERY", Some("q=1")),
        ("REF", Some("frag")),
        ("PROTOCOL", Some("http")),
        ("FILE", Some("/p?q=1")),
        ("AUTHORITY", Some("user:pass@h:8080")),
        ("USERINFO", Some("user:pass")),
    ] {
        assert_eq!(parse(full, part), owned(&[want]), "{part}");
    }
    // An unknown (or lower-case) part is NULL, not an error.
    for part in [
        "",
        "abc",
        "FOO",
        "1",
        "host",
        "Host",
        "Ä€日本語😀é",
        " \t a\u{3000}\n ",
    ] {
        assert_eq!(parse(URL, part), owned(&[None]), "{part:?}");
    }
    // NULL arguments make NULL.
    let (u, p) = (url(), part("HOST"));
    let f = bind("parse_url", &[&u, &p]);
    assert_eq!(f.name(), "parse_url");
    let null = strs(&[None]);
    assert_eq!(
        text(&f.invoke(&[col(&null), col(&p)], 1).unwrap()),
        owned(&[None])
    );
    assert_eq!(
        text(&f.invoke(&[col(&u), col(&null)], 1).unwrap()),
        owned(&[None])
    );
}

#[test]
fn parse_url_of_partial_urls() {
    // Missing components are NULL; an empty path is an empty string.
    for (url, part, want) in [
        ("1", "HOST", None),
        ("abc", "HOST", None),
        ("", "HOST", None),
        ("", "PATH", Some("")),
        ("?query=1", "PATH", Some("")),
        ("?query=1", "QUERY", Some("query=1")),
        ("?query=1", "HOST", None),
        ("#frag", "REF", Some("frag")),
        ("#frag", "PATH", Some("")),
        ("mailto:user@example.com", "PROTOCOL", Some("mailto")),
        ("mailto:user@example.com", "PATH", None),
        ("mailto:user@example.com", "QUERY", None),
        ("foo:bar", "HOST", None),
        ("http://host", "PATH", Some("")),
        ("http://host", "FILE", Some("")),
        ("http://host", "QUERY", None),
        ("http://host?query", "FILE", Some("?query")),
        ("http://host#frag", "FILE", Some("")),
        ("//host/path?q#f", "HOST", Some("host")),
        ("//host/path?q#f", "PROTOCOL", None),
        ("///path", "PATH", Some("/path")),
        ("///path", "HOST", None),
        ("//?q", "QUERY", Some("q")),
        ("/just/a/path?q#f", "PATH", Some("/just/a/path")),
        ("relative/path", "PATH", Some("relative/path")),
        ("http://h/a%20b?c%3Dd#e%20f", "PATH", Some("/a%20b")),
        ("http://h/a%20b?c%3Dd#e%20f", "QUERY", Some("c%3Dd")),
        ("http://h/a%20b?c%3Dd#e%20f", "REF", Some("e%20f")),
        ("HTTP://Example.COM/Path", "PROTOCOL", Some("HTTP")),
        ("HTTP://Example.COM/Path", "HOST", Some("Example.COM")),
        ("HTTP://Example.COM/Path", "PATH", Some("/Path")),
        ("http://user@host/", "USERINFO", Some("user")),
        ("http://user:@host/", "USERINFO", Some("user:")),
        ("http://:pass@host/", "USERINFO", Some(":pass")),
        ("http://@host/", "USERINFO", Some("")),
        ("http://user:p%40ss@host/", "USERINFO", Some("user:p%40ss")),
        ("http://host:/path", "HOST", Some("host")),
        ("http://host:0/", "HOST", Some("host")),
        ("http://host:65536/", "HOST", Some("host")),
    ] {
        assert_eq!(parse(url, part), owned(&[want]), "{url:?} {part}");
    }
}

#[test]
fn parse_url_of_registry_authorities() {
    // A registry-based authority is kept, with a NULL host and no error.
    for url in [
        "http://host:abc/",
        "http://host:80:90/",
        "http://host:+80/",
        "http://host:99999999999999999999/",
        "http://a_b/",
        "http://a~b/",
        "http://1.2.3.4.5/",
        "http://-bad/",
        "http://a..b/",
        "https://h%41ost/path",
        "http://häst/",
        "ssh://git@github.com:org/repo.git",
        "http://user@/",
        "http://:80/",
        "http://user:p@ss@host/",
        "http://1.2.3/",
        "http://999.1.1.1/",
        "http://1234.1.1.1/",
    ] {
        assert_eq!(parse(url, "HOST"), owned(&[None]), "{url:?}");
        assert!(parse(url, "AUTHORITY")[0].is_some(), "{url:?}");
    }
    assert_eq!(
        parse("http://host:abc/", "AUTHORITY"),
        owned(&[Some("host:abc")])
    );
    assert_eq!(parse("http://123/", "HOST"), owned(&[Some("123")]));
    assert_eq!(parse("http://1.2.3.4/", "HOST"), owned(&[Some("1.2.3.4")]));
    assert_eq!(
        parse("http://123.1.1.1/", "HOST"),
        owned(&[Some("123.1.1.1")])
    );
    assert_eq!(parse("http://host./", "HOST"), owned(&[Some("host.")]));
    assert_eq!(parse("http://a.b.c/", "HOST"), owned(&[Some("a.b.c")]));
}

#[test]
fn parse_url_of_ipv6_hosts() {
    // The host keeps its brackets, as `URI.getHost` keeps them.
    for (url, want) in [
        ("http://[::1]/path", Some("[::1]")),
        ("http://[::1]:8080/path?q#f", Some("[::1]")),
        ("http://[2001:db8::1]/", Some("[2001:db8::1]")),
        ("http://[1:2:3:4:5:6:7:8]/", Some("[1:2:3:4:5:6:7:8]")),
        ("http://[::]/", Some("[::]")),
        ("http://[::ffff:1.2.3.4]/", Some("[::ffff:1.2.3.4]")),
        ("http://[fe80::1%25eth0]/", Some("[fe80::1%25eth0]")),
    ] {
        assert_eq!(parse(url, "HOST"), owned(&[want]), "{url:?}");
    }
    assert_eq!(
        parse("http://[::1]:8080/path?q#f", "AUTHORITY"),
        owned(&[Some("[::1]:8080")])
    );
}

#[test]
fn parse_url_rejects_invalid_urls_under_ansi() {
    // Every URL `java.net.URI` rejects raises `INVALID_URL` under ANSI
    // and is NULL otherwise; `try_parse_url` is NULL in both modes.
    for url in [
        " ",
        " \t a\u{3000}\n ",
        "inva lid://spark.apache.org/path?query=1",
        "http://host/path with space",
        "http://[::1",
        "http://]/",
        "http://h[/",
        "http://us er@host/",
        "http://ho st/",
        "http://h/a%",
        "http://h/a%2",
        "http://h/%zz",
        "http://h%zz/",
        "http://u%zz@h/",
        "http://h/?q=%zz",
        "http://h/#f%zz",
        "http://h/p|q",
        "http://h/p^q",
        "http://h/p`q",
        "http://h/p{q}",
        "http://h/p<q>",
        "a\u{3000}b",
        "http://h/\u{3000}",
        "http://h/\u{7f}",
        "1a:x",
        "a:",
        ":x",
        "//",
        "a://",
        "a#b#c",
        "#a#b",
        "http://[zzz]/",
        "http://[::1]junk/",
        "http://[v7.a]/",
        "http://[1.2.3.4]/",
        "http://[1::2::3]/",
        "http://[1:2:3:4:5:6:7]/",
        "http://[12345::]/",
        "http://[]/",
        "http://[fe80::1%]/",
        "http://[::1]:80x/",
        "http://[::1]x/",
        "http://[1:2:]/",
        "http://[::1.2.3]/",
        "http://[1:2:3:4:5:6:7:8::]/",
        "http://[::1%-]/",
        "a:b%zz",
    ] {
        let (u, p) = (strs(&[Some(url)]), part("HOST"));
        let err = bind("parse_url", &[&u, &p])
            .invoke(&[col(&u), col(&p)], 1)
            .unwrap_err();
        assert_eq!(class(&err), ErrorClass::InvalidUrl, "{url:?}");
        assert!(err.to_string().contains(url), "{url:?}: {err}");
        let f = bind_with("parse_url", &[&u, &p], &legacy());
        assert_eq!(
            text(&f.invoke(&[col(&u), col(&p)], 1).unwrap()),
            owned(&[None]),
            "{url:?}"
        );
        for opts in [BindOptions::new(), legacy()] {
            let f = bind_with("try_parse_url", &[&u, &p], &opts);
            assert_eq!(
                text(&f.invoke(&[col(&u), col(&p)], 1).unwrap()),
                owned(&[None]),
                "{url:?}"
            );
        }
    }
    // `try_parse_url` reads what parses.
    let (u, p) = (
        strs(&[Some("http://spark.apache.org/path?query=1")]),
        part("HOST"),
    );
    assert_eq!(
        text(
            &bind("try_parse_url", &[&u, &p])
                .invoke(&[col(&u), col(&p)], 1)
                .unwrap()
        ),
        owned(&[Some("spark.apache.org")])
    );
    // Whatever/userinfo-tolerant URLs stay NULL without raising.
    let odd = strs(&[Some("inva lid://spark.apache.org/path?query=1")]);
    let q = part("QUERY");
    assert_eq!(
        text(
            &bind("try_parse_url", &[&odd, &q])
                .invoke(&[col(&odd), col(&q)], 1)
                .unwrap()
        ),
        owned(&[None])
    );
}

#[test]
fn parse_url_reads_query_keys() {
    let keys = |url: &str, key: Option<&str>| {
        let (u, p) = (strs(&[Some(url)]), part("QUERY"));
        let f = bind("parse_url", &[&u, &p, &strs(&[key])]);
        text(
            &f.invoke(&[col(&u), col(&p), lit(&strs(&[key]))], 1)
                .unwrap(),
        )
    };
    assert_eq!(
        keys("http://spark.apache.org/path?query=1", Some("query")),
        owned(&[Some("1")])
    );
    // A missing key, an empty query, no query and a NULL key are NULL.
    for (url, key) in [
        ("http://spark.apache.org/path?query=1", Some("abc")),
        ("http://spark.apache.org/path?query=1", Some("")),
        ("http://spark.apache.org/path?query=1", Some("1")),
        ("http://spark.apache.org/path?query=1", Some("qu")),
        ("http://spark.apache.org/path?query=1", None),
        ("http://spark.apache.org/path", Some("query")),
        ("1", Some("query")),
        ("", Some("query")),
    ] {
        assert_eq!(keys(url, key), owned(&[None]), "{url:?} {key:?}");
    }
    // The first `key=value` pair starting the query or following a `&`.
    assert_eq!(keys("http://h/?a=1&a=2", Some("a")), owned(&[Some("1")]));
    assert_eq!(keys("http://h/?=v", Some("")), owned(&[Some("v")]));
    assert_eq!(keys("http://h/?k=", Some("k")), owned(&[Some("")]));
    assert_eq!(keys("http://h/?k", Some("k")), owned(&[None]));
    assert_eq!(keys("http://h/?a=1&b=2", Some("b")), owned(&[Some("2")]));
    // A key is only read for `QUERY`; any other part with a key is NULL,
    // even for a URL that would not parse.
    for (url, part) in [
        ("http://spark.apache.org/path?query=1", "HOST"),
        ("http://spark.apache.org/path?query=1", "host"),
        (" \t a\u{3000}\n ", "HOST"),
    ] {
        let (u, p, k) = (
            strs(&[Some(url)]),
            strs(&[Some(part)]),
            strs(&[Some("query")]),
        );
        let f = bind("parse_url", &[&u, &p, &k]);
        assert_eq!(
            text(&f.invoke(&[col(&u), col(&p), lit(&k)], 1).unwrap()),
            owned(&[None]),
            "{url:?} {part}"
        );
    }
    // With `QUERY` the URL still parses: an invalid one raises under ANSI.
    let (u, p, k) = (
        strs(&[Some(" \t a\u{3000}\n ")]),
        strs(&[Some("QUERY")]),
        strs(&[Some("query")]),
    );
    let f = bind("parse_url", &[&u, &p, &k]);
    assert_eq!(
        class(&f.invoke(&[col(&u), col(&p), lit(&k)], 1).unwrap_err()),
        ErrorClass::InvalidUrl
    );
}

#[test]
fn parse_url_key_is_a_java_regex() {
    // Spark's `ParseUrlEvaluator.getPattern` compiles `(&|^)key=([^&]*)`
    // with the key unquoted: metacharacters act as regex (checked against
    // Spark 4.2.0). Group 2 of the first match is read, so a key with
    // groups reads the key's first group.
    let keys = |url: &str, key: Option<&str>| {
        let (u, p) = (strs(&[Some(url)]), part("QUERY"));
        let f = bind("parse_url", &[&u, &p, &strs(&[key])]);
        text(
            &f.invoke(&[col(&u), col(&p), lit(&strs(&[key]))], 1)
                .unwrap(),
        )
    };
    assert_eq!(
        keys("http://h/?aXb=1&a.b=2", Some("a.b")),
        owned(&[Some("1")])
    );
    assert_eq!(keys("http://h/?aXb=1", Some("a.b")), owned(&[Some("1")]));
    assert_eq!(keys("http://h/?aab=1", Some("a*b")), owned(&[Some("1")]));
    assert_eq!(keys("http://h/?ab=1", Some("a+b")), owned(&[Some("1")]));
    assert_eq!(keys("http://h/?a=1", Some("^a")), owned(&[Some("1")]));
    assert_eq!(keys("http://h/?ab=1", Some("(a)(b)")), owned(&[Some("a")]));
    assert_eq!(keys("http://h/?b=1", Some("a|b")), owned(&[Some("1")]));
    // A regex that matches no `key=` pair is NULL, as a missing key is.
    assert_eq!(keys("http://h/?a=1", Some("a$")), owned(&[None]));
    assert_eq!(keys("http://h/?a=1", Some("b")), owned(&[None]));
    // An invalid pattern fails the query in both modes (Spark itself
    // throws `PatternSyntaxException`, which has no error class, also for
    // `try_parse_url`), as an untyped internal error. The URL still parses
    // first: an invalid URL raises `INVALID_URL` before the key compiles.
    for name in ["parse_url", "try_parse_url"] {
        for opts in [BindOptions::new(), legacy()] {
            let (u, p, k) = (
                strs(&[Some("http://h/?a=1")]),
                part("QUERY"),
                strs(&[Some("[")]),
            );
            let f = bind_with(name, &[&u, &p, &k], &opts);
            let err = f.invoke(&[col(&u), col(&p), lit(&k)], 1).unwrap_err();
            assert_eq!(class(&err), ErrorClass::InternalError, "{name}");
        }
    }
}

#[test]
fn parse_url_rejects_other_shapes() {
    let (u, p) = (url(), part("HOST"));
    for args in [
        vec![DataType::Utf8],
        vec![
            DataType::Utf8,
            DataType::Utf8,
            DataType::Utf8,
            DataType::Utf8,
        ],
        vec![DataType::Int32, DataType::Utf8],
        vec![DataType::Utf8, DataType::Int32],
    ] {
        let bound: Vec<BindArg> = args.into_iter().map(BindArg::from).collect();
        for name in ["parse_url", "try_parse_url"] {
            let err = bind_scalar(name, &bound, &BindOptions::new()).unwrap_err();
            assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature, "{name}");
        }
    }
    // Columns and literals mix.
    let f = bind("parse_url", &[&u, &p]);
    assert_eq!(
        text(&f.invoke(&[lit(&u), col(&p)], 1).unwrap()),
        owned(&[Some("spark.apache.org")])
    );
}
