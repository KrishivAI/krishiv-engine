# SPDX-License-Identifier: Apache-2.0
"""Regenerates inputs/types_*.sql, inputs/to_types.sql and inputs/proto2.sql (P3-T087, PR-34 step 2).

Wire bytes come from `protoc --encode` (textproto in this file); no expected value is produced here, the `.out`
files come only from the Spark runs. Usage: python3 gen_inputs_types.py   (needs libprotoc on PATH).
"""

import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
GOLD = "/golden/from_to_protobuf/fixtures"
T = f"{GOLD}/krishiv_golden_types.desc"
P = f"{GOLD}/krishiv_golden_p2.desc"
TP = "krishiv_golden_types.proto"
PP = "krishiv_golden_p2.proto"
TN = "krishiv.golden.types."
PN = "krishiv.golden.p2."


def enc(proto, msg, text):
    out = subprocess.run(
        ["protoc", f"--encode={msg}", proto],
        input=text.encode(),
        cwd=FIX,
        check=True,
        capture_output=True,
    ).stdout
    return "X'" + out.hex() + "'"


def fp(data, msg, desc, opts=None):
    o = ""
    if opts is not None:
        o = ", map(" + ", ".join(f"'{k}', '{v}'" for k, v in opts.items()) + ")"
    return f"from_protobuf({data}, '{msg}', '{desc}'{o})"


def tp(data, msg, desc, opts=None):
    o = ""
    if opts is not None:
        o = ", map(" + ", ".join(f"'{k}', '{v}'" for k, v in opts.items()) + ")"
    return f"to_protobuf({data}, '{msg}', '{desc}'{o})"


class File:
    def __init__(self, name):
        self.name = name
        self.lines = []

    def c(self, text):
        self.lines.append("-- " + text)

    def q(self, sql):
        self.lines.append(sql + ";")

    def save(self):
        with open(os.path.join(HERE, "inputs", self.name + ".sql"), "w") as f:
            f.write("\n".join(self.lines) + "\n")


def tenc(msg, text):
    return enc(TP, TN + msg, text)


def penc(msg, text):
    return enc(PP, PN + msg, text)


# ---------------------------------------------------------------- types_wkt
f = File("types_wkt")
f.c(
    "Timestamp and Duration map to TIMESTAMP and INTERVAL DAY TO SECOND; wrappers are structs unless unwrapped"
)
W = TN + "Wkt"
f.q(f"select {fp(chr(88) + chr(39) + chr(39), W, T)}")
for o in (
    {"unwrap.primitive.wrapper.types": "true"},
    {"unwrap.primitive.wrapper.types": "true", "upcast.unsigned.ints": "true"},
    {"convert.any.fields.to.json": "true"},
    {"enums.as.ints": "true", "retain.empty.message.types": "true"},
):
    f.q(f"select {fp(chr(88) + chr(39) + chr(39), W, T, o)}")
f.c("Timestamp values (session time zone is UTC; see ansi_tz for others)")
TS = TN + "TsOnly"
for txt in (
    "ts { seconds: 1 }",
    "ts { seconds: 1 nanos: 500000000 }",
    "ts { seconds: 1 nanos: 1500 }",
    "ts { seconds: 1 nanos: 999 }",
    "ts { }",
    "ts { seconds: -1 nanos: 999999999 }",
    "ts { seconds: -1 }",
    "ts { seconds: -62135596800 }",
    "ts { seconds: 253402300799 nanos: 999999999 }",
    "ts { seconds: 1 nanos: -1 }",
    "ts { seconds: 1 nanos: 2000000000 }",
    "ts { seconds: 9223372036855 }",
    "ts { seconds: 9223372036854 }",
):
    f.q(f"select {fp(tenc('TsOnly', txt), TS, T)}")
f.q(
    f"select {fp(tenc('TsOnly', 'ts { seconds: 1 nanos: 500000000 }'), TS, T, {'emit.default.values': 'true'})}"
)
f.q(f"select {fp(tenc('TsOnly', ''), TS, T, {'emit.default.values': 'true'})}")
f.c("Duration values")
DU = TN + "DurOnly"
for txt in (
    "dur { seconds: 1 }",
    "dur { seconds: 1 nanos: 500000000 }",
    "dur { seconds: -1 nanos: -500000000 }",
    "dur { seconds: 1 nanos: -500000000 }",
    "dur { seconds: -1 nanos: 500000000 }",
    "dur { seconds: 90061 nanos: 123456789 }",
    "dur { seconds: 1 nanos: 1500 }",
    "dur { }",
    "dur { seconds: 9223372036855 }",
    "dur { seconds: 315576000000 }",
):
    f.q(f"select {fp(tenc('DurOnly', txt), DU, T)}")
f.q(f"select typeof({fp(chr(88) + chr(39) + chr(39), DU, T)}.dur)")
f.q(f"select typeof({fp(chr(88) + chr(39) + chr(39), TS, T)}.ts)")
f.c("repeated and map of Timestamp, repeated wrappers")
f.q(
    f"select {fp(tenc('Wkt', 'tss { seconds: 1 } tss { seconds: 2 nanos: 1000 } tsmap { key: ' + chr(34) + 'k' + chr(34) + ' value { seconds: 3 } } i32vs { value: 5 } i32vs { }'), W, T)}"
)
f.c(
    "wrapper types with and without unwrap; value 0 inside a wrapper, wrapper present but empty, absent"
)
WR = (
    "i32v { value: 5 } u32v { value: 4294967295 } i64v { value: -7 } u64v { value: 18446744073709551615 } "
    "strv { value: 'hi' } binv { value: 'ab' } boolv { value: true } fltv { value: 1.5 } dblv { value: 2.5 }"
)
W0 = "i32v { } u32v { } i64v { } u64v { } strv { } binv { } boolv { } fltv { } dblv { }"
for nm, txt in (("values", WR), ("empty wrappers", W0), ("absent", "")):
    f.c(nm)
    for o in (
        None,
        {"unwrap.primitive.wrapper.types": "true"},
        {"unwrap.primitive.wrapper.types": "true", "upcast.unsigned.ints": "true"},
        {"unwrap.primitive.wrapper.types": "true", "emit.default.values": "true"},
        {"emit.default.values": "true"},
    ):
        f.q(f"select {fp(tenc('Wkt', txt), W, T, o)}")
f.c(
    "Empty (google.protobuf.Empty) field: dropped from the schema unless retain.empty.message.types"
)
f.q(f"select {fp(tenc('Wkt', 'empty { }'), W, T)}")
f.q(
    f"select {fp(tenc('Wkt', 'empty { }'), W, T, {'retain.empty.message.types': 'true'})}"
)
f.save()

# ---------------------------------------------------------------- types_any
f = File("types_any")
A = TN + "AnyOnly"
f.c(
    "Any: struct<type_url,value> by default, JSON string with convert.any.fields.to.json"
)
anyp = "event: 'click' details { [type.googleapis.com/krishiv.golden.types.Payload] { name: 'Mario' id: 100 } }"
f.q(f"select {fp(chr(88) + chr(39) + chr(39), A, T)}")
f.q(f"select {fp(tenc('AnyOnly', anyp), A, T)}")
J = {"convert.any.fields.to.json": "true"}
f.q(f"select {fp(chr(88) + chr(39) + chr(39), A, T, J)}")
f.q(f"select {fp(tenc('AnyOnly', anyp), A, T, J)}")
f.q(
    f"select {fp(tenc('AnyOnly', anyp), A, T, {'convert.any.fields.to.json': 'false'})}"
)
f.q(f"select {fp(tenc('AnyOnly', anyp), A, T, {'CONVERT.ANY.FIELDS.TO.JSON': 'TRUE'})}")
f.q(f"select {fp(tenc('AnyOnly', anyp), A, T, {'convert.any.fields.to.json': 'yes'})}")
f.q(
    f"select {fp(tenc('AnyOnly', anyp), A, T, {'convert.any.fields.to.json': 'true', 'enums.as.ints': 'true'})}"
)
f.q(
    f"select {fp(tenc('AnyOnly', anyp), A, T, {'convert.any.fields.to.json': 'true', 'emit.default.values': 'true'})}"
)
f.c("Any holding well-known types, an empty Any, many Any")
f.q(
    f"select {fp(tenc('AnyOnly', 'details { [type.googleapis.com/google.protobuf.Timestamp] { seconds: 1 nanos: 5000 } }'), A, T, J)}"
)
f.q(
    f"select {fp(tenc('AnyOnly', 'details { [type.googleapis.com/google.protobuf.Duration] { seconds: 1 } }'), A, T, J)}"
)
f.q(
    f"select {fp(tenc('AnyOnly', 'details { [type.googleapis.com/google.protobuf.Int32Value] { value: 3 } }'), A, T, J)}"
)
f.q(f"select {fp(tenc('AnyOnly', 'details { }'), A, T, J)}")
f.q(f"select {fp(tenc('AnyOnly', 'details { }'), A, T)}")
f.q(
    f"select {fp(tenc('AnyOnly', 'many { [type.googleapis.com/krishiv.golden.types.Payload] { name: ' + chr(39) + 'a' + chr(39) + ' } } many { [type.googleapis.com/krishiv.golden.types.Payload] { id: 2 } }'), A, T, J)}"
)
f.q(
    f"select {fp(tenc('AnyOnly', 'many { [type.googleapis.com/krishiv.golden.types.Payload] { name: ' + chr(39) + 'a' + chr(39) + ' } }'), A, T)}"
)
f.c("Any with an unresolvable type url: FAILFAST error vs PERMISSIVE NULL")
bad = enc(
    TP,
    TN + "AnyOnly",
    "details { type_url: 'type.googleapis.com/no.such.Type' value: 'x' }",
)
f.q(f"select {fp(bad, A, T, J)}")
f.q(f"select {fp(bad, A, T, dict(J, mode='PERMISSIVE'))}")
f.q(f"select {fp(bad, A, T)}")
f.c("Any with a url but no registered type, bare url form, malformed payload")
bad2 = enc(
    TP,
    TN + "AnyOnly",
    "details { type_url: 'krishiv.golden.types.Payload' value: 'x' }",
)
f.q(f"select {fp(bad2, A, T, J)}")
bad3 = enc(
    TP,
    TN + "AnyOnly",
    "details { type_url: 'type.googleapis.com/krishiv.golden.types.Payload' value: '\\377' }",
)
f.q(f"select {fp(bad3, A, T, J)}")
f.c("to_protobuf of an Any field from a struct<type_url,value>")
f.q(
    f"select hex({tp(chr(110) + 'amed_struct(' + chr(39) + 'event' + chr(39) + ', ' + chr(39) + 'e' + chr(39) + ', ' + chr(39) + 'details' + chr(39) + ', named_struct(' + chr(39) + 'type_url' + chr(39) + ', ' + chr(39) + 'type.googleapis.com/krishiv.golden.types.Payload' + chr(39) + ', ' + chr(39) + 'value' + chr(39) + ', X' + chr(39) + '0a024142' + chr(39) + '))', A, T)})"
)
f.q(
    f"select hex({tp('named_struct(' + chr(39) + 'details' + chr(39) + ', ' + chr(39) + 'x' + chr(39) + ')', A, T)})"
)
f.q(
    f"select hex({tp('named_struct(' + chr(39) + 'details' + chr(39) + ', ' + chr(39) + 'x' + chr(39) + ')', A, T, J)})"
)
f.save()

# ---------------------------------------------------------------- types_empty
f = File("types_empty")
f.c(
    "empty message types: dropped by default, kept as __dummy_field_in_empty_struct with retain.empty.message.types"
)
R = {"retain.empty.message.types": "true"}
H = TN + "HasEmpty"
E = TN + "Empty0"
OE = TN + "OnlyEmpty"
x = "X''"
for msg in (H, OE, E):
    f.q(f"select {fp(x, msg, T)}")
    f.q(f"select {fp(x, msg, T, R)}")
f.q(f"select typeof({fp(x, E, T)})")
f.q(f"select typeof({fp(x, E, T, R)})")
f.q(f"select typeof({fp(x, OE, T)}.e)")
hv = "name: 'n' e { } es { } es { } em { key: 'k' value { } }"
f.q(f"select {fp(tenc('HasEmpty', hv), H, T)}")
f.q(f"select {fp(tenc('HasEmpty', hv), H, T, R)}")
f.q(
    f"select {fp(tenc('HasEmpty', hv), H, T, dict(R, **{'emit.default.values': 'true'}))}"
)
f.q(f"select {fp(tenc('OnlyEmpty', 'e { }'), OE, T)}")
f.q(f"select {fp(tenc('OnlyEmpty', 'e { }'), OE, T, R)}")
f.q(
    f"select {fp(tenc('OnlyEmpty', 'e { }'), OE, T, {'retain.empty.message.types': 'maybe'})}"
)
f.q(
    f"select {fp(tenc('OnlyEmpty', ''), OE, T, {'retain.empty.message.types': 'true', 'mode': 'PERMISSIVE'})} is null"
)
f.c("to_protobuf into empty / retained-empty messages")
f.q(f"select hex({tp('named_struct()', E, T)})")
f.q(
    f"select hex({tp('named_struct(' + chr(39) + '__dummy_field_in_empty_struct' + chr(39) + ', ' + chr(39) + 'x' + chr(39) + ')', E, T)})"
)
f.q(
    f"select hex({tp('named_struct(' + chr(39) + 'e' + chr(39) + ', named_struct())', OE, T)})"
)
f.q(
    f"select hex({tp('named_struct(' + chr(39) + 'e' + chr(39) + ', named_struct(' + chr(39) + '__dummy_field_in_empty_struct' + chr(39) + ', ' + chr(39) + 'x' + chr(39) + '))', OE, T)})"
)
f.save()

# ---------------------------------------------------------------- types_recursion
f = File("types_recursion")
f.c(
    "recursive.fields.max.depth: -1 (default) rejects, 1 drops the recursive field, 2.. nests; 0 and >10 rejected"
)
Rc = TN + "Rec"
for d in (
    None,
    "-1",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "10",
    "11",
    "100",
    "abc",
    "1.5",
    " 2",
    "2 ",
    "+2",
    "02",
):
    o = None if d is None else {"recursive.fields.max.depth": d}
    f.q(f"select {fp(x, Rc, T, o)}")
f.c("mutual recursion MA <-> MB, a message that is only a recursive field")
for d in (None, "1", "2", "3"):
    o = None if d is None else {"recursive.fields.max.depth": d}
    f.q(f"select {fp(x, TN + 'MA', T, o)}")
    f.q(f"select {fp(x, TN + 'MB', T, o)}")
    f.q(f"select {fp(x, TN + 'OnlyRec', T, o)}")
    f.q(
        f"select {fp(x, TN + 'OnlyRec', T, dict(o or {}, **{'retain.empty.message.types': 'true'}))}"
    )
f.c("values truncate at the depth limit")
rv = (
    "name: 'a' child { name: 'b' child { name: 'c' child { name: 'd' } } } "
    "kids { name: 'k1' child { name: 'k2' } } byname { key: 'x' value { name: 'v' child { name: 'w' } } }"
)
for d in ("1", "2", "3", "4"):
    f.q(f"select {fp(tenc('Rec', rv), Rc, T, {'recursive.fields.max.depth': d})}")
f.q(f"select {fp(tenc('Rec', rv), Rc, T)}")
f.q(
    f"select {fp(tenc('Rec', rv), Rc, T, {'recursive.fields.max.depth': '3', 'emit.default.values': 'true'})}"
)
f.q(
    f"select {fp(tenc('MA', 'x: 1 b { y: 2 a { x: 3 b { y: 4 } } }'), TN + 'MA', T, {'recursive.fields.max.depth': '2'})}"
)
f.c("to_protobuf of a deeper struct than the proto allows is not limited by the option")
f.q(
    f"select hex({tp('named_struct(' + chr(39) + 'name' + chr(39) + ', ' + chr(39) + 'a' + chr(39) + ', ' + chr(39) + 'child' + chr(39) + ', named_struct(' + chr(39) + 'name' + chr(39) + ', ' + chr(39) + 'b' + chr(39) + '))', Rc, T, {'recursive.fields.max.depth': '1'})})"
)
f.c(
    "recursion through repeated and map: the container field is dropped, the others stay"
)
f.q(f"select {fp(x, Rc, T, {'recursive.fields.max.depth': '1'})}")
f.save()

# ---------------------------------------------------------------- types_kinds
f = File("types_kinds")
K = TN + "Kinds"
f.c("Kinds: type mapping of repeated/map/oneof/optional/nested enum/field names")
f.q(f"select {fp(x, K, T, {'recursive.fields.max.depth': '2'})}")
f.q(
    f"select {fp(x, K, T, {'recursive.fields.max.depth': '2', 'enums.as.ints': 'true', 'upcast.unsigned.ints': 'true'})}"
)
f.q(f"select {fp(x, K, T)}")
kv = (
    "shades: LIGHT shades: DARK shades: SHADE_UNSPECIFIED shade: DARK "
    "bins: 'a' bins: '' bins: '\\377' strs: 'x' strs: '' strs: '\\303\\251' "
    "flts: 1.5 flts: nan flts: -inf dbls: -0.0 dbls: inf bools: true bools: false "
    "s64s: -1 s64s: 2 u32s: 4294967295 u64s: 18446744073709551615 "
    "i2s { key: 1 value: 'one' } i2s { key: -1 value: '' } l2b { key: 5 value: 'zz' } "
    "b2e { key: true value: DARK } s2p { key: 'k' value { name: 'n' id: 3 } } u2u { key: 4294967295 value: 18446744073709551615 } "
    "p { name: 'oneof' } ps { name: 'p1' } ps { } single { } opayload { } oshade: SHADE_UNSPECIFIED ostr: '' "
    "packed_nums: 1 packed_nums: 2 unpacked_nums: 3 unpacked_nums: 4 nested { inside: INSIDE_ONE } "
    "CamelCase: 'c' snake_case: 's' json_name_field: 'j'"
)
f.q(f"select {fp(tenc('Kinds', kv), K, T, {'recursive.fields.max.depth': '2'})}")
f.q(
    f"select {fp(tenc('Kinds', kv), K, T, {'recursive.fields.max.depth': '2', 'enums.as.ints': 'true', 'upcast.unsigned.ints': 'true'})}"
)
f.q(
    f"select {fp(tenc('Kinds', kv), K, T, {'recursive.fields.max.depth': '2', 'emit.default.values': 'true'})}"
)
f.c("oneof: each member, none, two members on the wire (last wins)")
f.q(
    f"select {fp(tenc('Kinds', 'sh: DARK'), K, T, {'recursive.fields.max.depth': '2'})}"
)
f.q(
    f"select {fp(tenc('Kinds', 'raw: ' + chr(39) + 'ab' + chr(39)), K, T, {'recursive.fields.max.depth': '2'})}"
)
f.q(
    f"select {fp(tenc('Kinds', 'sh: SHADE_UNSPECIFIED'), K, T, {'recursive.fields.max.depth': '2'})}.sh"
)
f.q(
    f"select {fp('X' + chr(39) + '8a01030a0161' + '900102' + chr(39), K, T, {'recursive.fields.max.depth': '2'})}"
)
f.q(
    f"select {fp('X' + chr(39) + '900102' + '8a01030a0161' + chr(39), K, T, {'recursive.fields.max.depth': '2'})}"
)
f.c("proto3 optional: presence tracked, explicit empty/zero kept")
f.q(
    f"select {fp(tenc('Kinds', 'oshade: SHADE_UNSPECIFIED ostr: ' + chr(39) + chr(39)), K, T, {'recursive.fields.max.depth': '2'})}"
)
f.q(
    f"select {fp(tenc('Kinds', ''), K, T, {'recursive.fields.max.depth': '2', 'emit.default.values': 'true'})}"
)
f.c("enum numbers outside the declared set, negative enum numbers, aliases")
f.q(
    f"select {fp(tenc('Kinds', 'shade: 99'), K, T, {'recursive.fields.max.depth': '2'})}"
)
f.q(
    f"select {fp(tenc('Kinds', 'shade: 99'), K, T, {'recursive.fields.max.depth': '2', 'enums.as.ints': 'true'})}"
)
f.q(
    f"select {fp(tenc('Kinds', 'shades: 99 shades: LIGHT'), K, T, {'recursive.fields.max.depth': '2'})}"
)
f.q(
    f"select {fp(tenc('Kinds', 'shade: -5'), K, T, {'recursive.fields.max.depth': '2'})}"
)
f.q(
    f"select {fp(tenc('WeirdMsg', 'w: W_NEG ws: W_ONE ws: W_ONE_ALIAS ws: W_ZERO'), TN + 'WeirdMsg', T)}"
)
f.q(
    f"select {fp(tenc('WeirdMsg', 'w: W_NEG ws: W_ONE ws: W_ONE_ALIAS ws: W_ZERO'), TN + 'WeirdMsg', T, {'enums.as.ints': 'true'})}"
)
f.q(f"select {fp(tenc('WeirdMsg', 'w: 7'), TN + 'WeirdMsg', T)}")
f.c(
    "repeated: packed and unpacked encodings of the same field are both accepted, interleaved occurrences concatenate"
)
f.q(
    f"select {fp('X' + chr(39) + 'ca0102 0102'.replace(' ', '') + chr(39), K, T, {'recursive.fields.max.depth': '2'})}.packed_nums"
)
f.q(
    f"select {fp('X' + chr(39) + 'c80101' + 'ca01020304' + 'c80105' + chr(39), K, T, {'recursive.fields.max.depth': '2'})}.packed_nums"
)
f.q(
    f"select {fp('X' + chr(39) + 'd00101' + 'd201020304' + chr(39), K, T, {'recursive.fields.max.depth': '2'})}.unpacked_nums"
)
f.c("map: duplicate keys, entry without value, entry without key, map order")
f.q(
    f"select {fp(tenc('Kinds', 'i2s { key: 1 value: ' + chr(39) + 'a' + chr(39) + ' } i2s { key: 1 value: ' + chr(39) + 'b' + chr(39) + ' }'), K, T, {'recursive.fields.max.depth': '2'})}.i2s"
)
f.q(
    f"select map_keys({fp(tenc('Kinds', 'i2s { key: 3 value: ' + chr(39) + 'a' + chr(39) + ' } i2s { key: 1 value: ' + chr(39) + 'b' + chr(39) + ' }'), K, T, {'recursive.fields.max.depth': '2'})}.i2s)"
)
f.q(
    f"select {fp(tenc('Kinds', 'i2s { key: 1 }'), K, T, {'recursive.fields.max.depth': '2'})}.i2s"
)
f.q(
    f"select {fp(tenc('Kinds', 'i2s { value: ' + chr(39) + 'v' + chr(39) + ' }'), K, T, {'recursive.fields.max.depth': '2'})}.i2s"
)
f.q(
    f"select {fp(tenc('Kinds', 's2p { key: ' + chr(39) + 'k' + chr(39) + ' }'), K, T, {'recursive.fields.max.depth': '2'})}.s2p"
)
f.q(
    f"select {fp(tenc('Kinds', 's2p { key: ' + chr(39) + 'k' + chr(39) + ' }'), K, T, {'recursive.fields.max.depth': '2', 'emit.default.values': 'true'})}.s2p"
)
f.q(
    f"select {fp(tenc('Kinds', 's2rec { key: ' + chr(39) + 'k' + chr(39) + ' value { name: ' + chr(39) + 'r' + chr(39) + ' child { name: ' + chr(39) + 'q' + chr(39) + ' } } }'), K, T, {'recursive.fields.max.depth': '2'})}.s2rec"
)
f.c(
    "repeated message elements: an empty element message is a struct of NULLs, not NULL"
)
f.q(
    f"select {fp(tenc('Kinds', 'ps { } ps { name: ' + chr(39) + 'z' + chr(39) + ' }'), K, T, {'recursive.fields.max.depth': '2'})}.ps"
)
f.c("to_protobuf: repeated / map / oneof / optional / nested enum / enums")
Kd = {"recursive.fields.max.depth": "2"}


def ns(*a):
    return "named_struct(" + ", ".join(a) + ")"


def sq(s):
    return "'" + s + "'"


f.q(
    f"select hex({tp(ns(sq('shades'), 'array(' + sq('LIGHT') + ', ' + sq('DARK') + ')', sq('shade'), sq('DARK')), K, T)})"
)
f.q(f"select hex({tp(ns(sq('shades'), 'array(1, 2)'), K, T)})")
f.q(
    f"select hex({tp(ns(sq('shades'), 'array(1, 2)'), K, T, {'enums.as.ints': 'true'})})"
)
f.q(f"select hex({tp(ns(sq('shade'), '99'), K, T)})")
f.q(f"select hex({tp(ns(sq('shade'), sq('LIGHT')), K, T)})")
f.q(f"select hex({tp(ns(sq('shade'), sq('light')), K, T)})")
f.q(
    f"select hex({tp(ns(sq('bins'), 'array(X' + sq('ff') + ', X' + sq('') + ')', sq('strs'), 'array(' + sq('a') + ', ' + sq('') + ')'), K, T)})"
)
f.q(
    f"select hex({tp(ns(sq('flts'), 'array(1.5f, cast(' + sq('NaN') + ' as float))', sq('dbls'), 'array(-0.0d)', sq('bools'), 'array(true, false)'), K, T)})"
)
f.q(
    f"select hex({tp(ns(sq('s64s'), 'array(-1L, 2L)', sq('u32s'), 'array(-1)', sq('u64s'), 'array(-1L)'), K, T)})"
)
f.q(f"select hex({tp(ns(sq('u32s'), 'array(cast(4294967295 as bigint))'), K, T)})")
f.q(
    f"select hex({tp(ns(sq('u64s'), 'array(cast(18446744073709551615 as decimal(20,0)))'), K, T)})"
)
f.q(f"select hex({tp(ns(sq('u64s'), 'array(cast(-1 as decimal(20,0)))'), K, T)})")
f.q(
    f"select hex({tp(ns(sq('u2u'), 'map(cast(4294967295 as bigint), cast(18446744073709551615 as decimal(20,0)))'), K, T)})"
)
f.q(
    f"select hex({tp(ns(sq('i2s'), 'map(1, ' + sq('one') + ', -1, ' + sq('') + ')', sq('l2b'), 'map(5L, X' + sq('7a') + ')', sq('b2e'), 'map(true, ' + sq('DARK') + ')'), K, T)})"
)
f.q(
    f"select hex({tp(ns(sq('s2p'), 'map(' + sq('k') + ', named_struct(' + sq('name') + ', ' + sq('n') + ', ' + sq('id') + ', 3))'), K, T)})"
)
f.q(f"select hex({tp(ns(sq('p'), ns(sq('name'), sq('o'))), K, T)})")
f.q(
    f"select hex({tp(ns(sq('p'), ns(sq('name'), sq('o')), sq('sh'), sq('DARK')), K, T)})"
)
f.q(f"select hex({tp(ns(sq('sh'), sq('SHADE_UNSPECIFIED')), K, T)})")
f.q(
    f"select hex({tp(ns(sq('oshade'), sq('SHADE_UNSPECIFIED'), sq('ostr'), sq('')), K, T)})"
)
f.q(f"select hex({tp(ns(sq('nested'), ns(sq('inside'), sq('INSIDE_ONE'))), K, T)})")
f.q(f"select hex({tp(ns(sq('single'), ns()), K, T)})")
f.q(
    f"select hex({tp(ns(sq('ps'), 'array(named_struct(' + sq('name') + ', cast(null as string)), named_struct(' + sq('name') + ', ' + sq('z') + '))'), K, T)})"
)
f.q(f"select hex({tp(ns(sq('ps'), 'array(cast(null as struct<name:string>))'), K, T)})")
f.q(f"select hex({tp(ns(sq('CAMELCASE'), sq('c'), sq('Snake_Case'), sq('s')), K, T)})")
f.q(f"select hex({tp(ns(sq('customJson'), sq('j')), K, T)})")
f.q(f"select hex({tp(ns(sq('json_name_field'), sq('j')), K, T)})")
f.c(
    "to_protobuf field names match case-insensitively by default, exactly with spark.sql.caseSensitive=true"
)
f.q(
    f"select hex({tp(ns(sq('I32'), '1'), TN.replace('types.', '') + 'Scalars', GOLD + '/krishiv_golden.desc')})"
)
f.save()

# ---------------------------------------------------------------- types_wkt_to  (to_protobuf of wkt)
f = File("to_types")
f.c("to_protobuf to Timestamp / Duration / wrappers / Empty / Any fields")
S = lambda s: "'" + s + "'"
f.q(
    f"select hex({tp(ns(S('ts'), 'timestamp' + S('2024-01-02 03:04:05.678901')), W, T)})"
)
f.q(
    f"select hex({tp(ns(S('ts'), 'timestamp' + S('1969-12-31 23:59:59.999999')), W, T)})"
)
f.q(f"select hex({tp(ns(S('ts'), 'timestamp' + S('1969-12-31 23:59:59.0005')), W, T)})")
f.q(f"select hex({tp(ns(S('ts'), 'timestamp' + S('1970-01-01 00:00:00')), W, T)})")
f.q(f"select hex({tp(ns(S('ts'), 'timestamp' + S('0001-01-01 00:00:00')), W, T)})")
f.q(f"select hex({tp(ns(S('ts'), 'timestamp_ntz' + S('2024-01-02 03:04:05')), W, T)})")
f.q(f"select hex({tp(ns(S('ts'), 'date' + S('2024-01-02')), W, T)})")
f.q(f"select hex({tp(ns(S('ts'), '1L'), W, T)})")
f.q(f"select hex({tp(ns(S('ts'), ns(S('seconds'), '5L', S('nanos'), '7')), W, T)})")
f.q(f"select hex({tp(ns(S('ts'), ns(S('seconds'), '5L')), W, T)})")
f.q(
    f"select hex({tp(ns(S('tss'), 'array(timestamp' + S('1970-01-01 00:00:01') + ', timestamp' + S('1970-01-01 00:00:02.5') + ')'), W, T)})"
)
f.q(
    f"select hex({tp(ns(S('tsmap'), 'map(' + S('k') + ', timestamp' + S('1970-01-01 00:00:01') + ')'), W, T)})"
)
f.q(
    f"select hex({tp(ns(S('dur'), 'interval ' + S('1 02:03:04.567891') + ' day to second'), W, T)})"
)
f.q(
    f"select hex({tp(ns(S('dur'), 'interval ' + S('-1 02:03:04.567891') + ' day to second'), W, T)})"
)
f.q(
    f"select hex({tp(ns(S('dur'), 'interval ' + S('0 00:00:00.0005') + ' day to second'), W, T)})"
)
f.q(
    f"select hex({tp(ns(S('dur'), 'interval ' + S('-0 00:00:00.5') + ' day to second'), W, T)})"
)
f.q(f"select hex({tp(ns(S('dur'), 'interval ' + S('1') + ' day'), W, T)})")
f.q(f"select hex({tp(ns(S('dur'), 'interval ' + S('3') + ' hour'), W, T)})")
f.q(f"select hex({tp(ns(S('dur'), 'interval ' + S('90') + ' second'), W, T)})")
f.q(f"select hex({tp(ns(S('dur'), 'interval ' + S('1-2') + ' year to month'), W, T)})")
f.q(
    f"select hex({tp(ns(S('dur'), 'interval ' + S('1') + ' day' + ' + interval ' + S('1') + ' month'), W, T)})"
)
f.q(f"select hex({tp(ns(S('dur'), 'cast(null as interval day to second)'), W, T)})")
f.c(
    "wrapper types accept the primitive; a struct{value} also works; mismatched widths do not"
)
f.q(
    f"select hex({tp(ns(S('i32v'), '5', S('u32v'), '-1', S('i64v'), '-7L', S('u64v'), '-1L', S('strv'), S('hi'), S('binv'), 'X' + S('6162'), S('boolv'), 'true', S('fltv'), '1.5f', S('dblv'), '2.5d'), W, T)})"
)
f.q(
    f"select hex({tp(ns(S('i32v'), '0', S('strv'), S(''), S('boolv'), 'false', S('dblv'), '0.0d'), W, T)})"
)
f.q(f"select hex({tp(ns(S('i32v'), ns(S('value'), '5')), W, T)})")
f.q(f"select hex({tp(ns(S('i32v'), '5L'), W, T)})")
f.q(f"select hex({tp(ns(S('i64v'), '5'), W, T)})")
f.q(f"select hex({tp(ns(S('u32v'), 'cast(5 as bigint)'), W, T)})")
f.q(f"select hex({tp(ns(S('u64v'), 'cast(5 as decimal(20,0))'), W, T)})")
f.q(f"select hex({tp(ns(S('i32v'), 'cast(null as int)'), W, T)})")
f.q(f"select hex({tp(ns(S('i32vs'), 'array(1, 2)'), W, T)})")
f.q(f"select hex({tp(ns(S('empty'), ns()), W, T)})")
f.q(f"select hex({tp(ns(S('empty'), ns()), W, T, R)})")
f.save()

# ---------------------------------------------------------------- proto2
f = File("proto2")
P2 = PN + "P2"
f.c(
    "proto2: required, defaults, group, extensions (spark.sql.function.protobufExtensions.enabled)"
)
f.q(f"select {fp(x, P2, P)}")
f.q(f"select {fp(x, P2, P, {'mode': 'PERMISSIVE'})}")
f.q(f"select {fp(penc('P2', 'req: 1'), P2, P)}")
f.q(f"select {fp(penc('P2', 'req: 1'), P2, P, {'emit.default.values': 'true'})}")
f.q(
    f"select {fp(penc('P2', 'req: 0 opt: ' + chr(39) + chr(39) + ' n: 0 hue: HUE_A flag: false d: 0'), P2, P)}"
)
f.q(
    f"select {fp(penc('P2', 'req: 1 Grp { x: 5 } item { label: ' + chr(39) + 'l' + chr(39) + ' } r: 1 r: 2'), P2, P)}"
)
f.q(f"select {fp(penc('P2', 'req: 1 Grp { }'), P2, P)}")
f.q(f"select typeof({fp(x, P2, P, {'mode': 'PERMISSIVE'})}.grp)")
f.q(f"select {fp('X' + chr(39) + '08012807' + chr(39), P2, P)}")
f.q(
    f"select {fp('X' + chr(39) + '08012807' + chr(39), P2, P, {'enums.as.ints': 'true'})}"
)
f.c("proto2 unknown enum number: preserved as unknown field, not an enum value")
f.q(f"select {fp('X' + chr(39) + '0801' + '2807' + chr(39), P2, P)}")
f.c("proto2 message nesting and a repeated message with a missing required field")
f.q(f"select {fp(x, PN + 'P2Nest', P)}")
f.q(f"select {fp(penc('P2Nest', 'p { req: 1 }'), PN + 'P2Nest', P)}")
f.q(f"select {fp(penc('P2Nest', 'ps { req: 1 } ps { req: 2 }'), PN + 'P2Nest', P)}")
f.q(f"select {fp('X' + chr(39) + '1200' + chr(39), PN + 'P2Nest', P)}")
f.q(
    f"select {fp('X' + chr(39) + '1200' + chr(39), PN + 'P2Nest', P, {'mode': 'PERMISSIVE'})}"
)
f.q(
    f"select {fp(penc('P2Required', 'a: ' + chr(39) + 'x' + chr(39) + ' b { }'), PN + 'P2Required', P)}"
)
f.q(f"select {fp('X' + chr(39) + '0a0178' + chr(39), PN + 'P2Required', P)}")
f.q(
    f"select {fp('X' + chr(39) + '0a0178' + chr(39), PN + 'P2Required', P, {'mode': 'PERMISSIVE'})}"
)
f.c(
    "extensions: not in the schema by default; with the SQL conf on, extension fields are appended (4.2 source)"
)
ext = "req: 1 [krishiv.golden.p2.ext_i]: 5 [krishiv.golden.p2.ext_s]: 'e' [krishiv.golden.p2.ext_item] { label: 'x' } [krishiv.golden.p2.ext_r]: 1 [krishiv.golden.p2.ext_r]: 2"
f.q(f"select {fp(penc('P2', ext), P2, P)}")
f.q("set spark.sql.function.protobufExtensions.enabled=true")
f.q(f"select {fp(x, P2, P, {'mode': 'PERMISSIVE'})}")
f.q(f"select {fp(penc('P2', ext), P2, P)}")
f.q(f"select {fp(penc('P2', ext), P2, P, {'emit.default.values': 'true'})}")
f.q(
    f"select hex({tp(ns(S('req'), '1', S('ext_i'), '5', S('ext_s'), S('e'), S('ext_item'), ns(S('label'), S('x')), S('ext_r'), 'array(1, 2)'), P2, P)})"
)
f.q("set spark.sql.function.protobufExtensions.enabled=false")
f.q(f"select hex({tp(ns(S('req'), '1', S('ext_i'), '5'), P2, P)})")
f.q(f"select hex({tp(ns(S('req'), '1'), P2, P)})")
f.c("to_protobuf proto2: required fields missing / NULL; defaults are not written")
f.q(f"select hex({tp(ns(S('opt'), S('x')), P2, P)})")
f.q(f"select hex({tp(ns(S('req'), 'cast(null as int)'), P2, P)})")
f.q(
    f"select hex({tp(ns(S('req'), '0', S('opt'), S('dflt'), S('hue'), S('HUE_C'), S('flag'), 'true', S('d'), 'cast(' + S('Infinity') + ' as double)'), P2, P)})"
)
f.q(f"select hex({tp(ns(S('req'), '1', S('grp'), ns(S('x'), '5')), P2, P)})")
f.q(f"select hex({tp(ns(S('a'), S('x'), S('b'), ns()), PN + 'P2Required', P)})")
f.q(
    f"select hex({tp(ns(S('a'), S('x'), S('b'), 'cast(null as struct<label:string>)'), PN + 'P2Required', P)})"
)
f.q(f"select hex({tp(ns(S('a'), S('x')), PN + 'P2Required', P)})")
f.q(
    f"select hex({tp(ns(S('a'), 'cast(null as string)', S('b'), ns()), PN + 'P2Required', P)})"
)
f.c(
    "case-insensitive field resolution and ambiguity (CaseClash has fields abc and ABC)"
)
CC = PN + "CaseClash"
f.q(
    f"select {fp(penc('CaseClash', 'abc: ' + chr(39) + '1' + chr(39) + ' ABC: ' + chr(39) + '2' + chr(39)), CC, P)}"
)
f.q(f"select hex({tp(ns(S('abc'), S('1')), CC, P)})")
f.q(f"select hex({tp(ns(S('ABC'), S('2')), CC, P)})")
f.q(f"select hex({tp(ns(S('Abc'), S('3')), CC, P)})")
f.q("set spark.sql.caseSensitive=true")
f.q(f"select hex({tp(ns(S('Abc'), S('3')), CC, P)})")
f.q(f"select hex({tp(ns(S('abc'), S('1'), S('ABC'), S('2')), CC, P)})")
f.q(
    f"select hex({tp(ns(S('I32'), '1'), PN.replace('p2.', '') + 'Scalars', GOLD + '/krishiv_golden.desc')})"
)
f.q("set spark.sql.caseSensitive=false")
f.q(
    f"select hex({tp(ns(S('I32'), '1'), PN.replace('p2.', '') + 'Scalars', GOLD + '/krishiv_golden.desc')})"
)
f.save()

# ---------------------------------------------------------------- types_tz
f = File("types_tz")
f.c(
    "session time zone only changes how a decoded Timestamp renders / how a local literal is encoded; Duration never depends on it"
)
for tz in ("UTC", "America/Los_Angeles", "Asia/Kolkata"):
    f.q(f"set spark.sql.session.timeZone={tz}")
    for sec in (1710053999, 1710054000, 1730613599, 1730613600, 0):
        f.q(f"select {fp(tenc('TsOnly', f'ts {{ seconds: {sec} }}'), TS, T)}")
    f.q(f"select hex({tp(ns(S('ts'), 'timestamp' + S('2024-03-10 02:30:00')), W, T)})")
    f.q(f"select hex({tp(ns(S('ts'), 'timestamp' + S('2024-11-03 01:30:00')), W, T)})")
    f.q(
        f"select hex({tp(ns(S('ts'), 'timestamp' + S('2024-01-02 03:04:05') + ' + interval ' + S('1') + ' hour'), W, T)})"
    )
    f.q(f"select {fp(tenc('DurOnly', 'dur { seconds: 3600 }'), DU, T)}")
f.q("set spark.sql.session.timeZone=UTC")
f.c("ANSI off: no effect on either function")
f.q("set spark.sql.ansi.enabled=false")
f.q(f"select {fp(tenc('TsOnly', 'ts { seconds: 9223372036855 }'), TS, T)}")
f.q(f"select {fp(tenc('TsOnly', 'ts { seconds: 1 }'), TS, T)}")
f.q(f"select hex({tp(ns(S('shade'), S('99')), K, T)})")
f.q(f"select hex({tp(ns(S('shade'), '99'), K, T)})")
f.q(f"select hex({tp(ns(S('shades'), 'array(' + S('LIGHT') + ')'), K, T)})")
f.q(f"select hex({tp(ns(S('i32v'), 'cast(' + S('x') + ' as int)'), W, T)})")
f.q("set spark.sql.ansi.enabled=true")
f.save()
