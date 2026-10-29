#!/usr/bin/env python3
"""Generate the Spark WKB fixture (P2-T181): how Spark 4.2 reads and writes well-known binary.

Each case is a WKB byte string built below. A REAL Spark 4.2 run (a local `pyspark==4.2.0` install, which
ships the Spark distribution) evaluates, for both spatial types,

    hex(ST_AsBinary(ST_GeomFromWKB(X'..')))  and  hex(ST_AsBinary(ST_GeomFromWKB(X'..'), 'XDR'))
    hex(ST_AsBinary(ST_GeogFromWKB(X'..')))  and  hex(ST_AsBinary(ST_GeogFromWKB(X'..'), 'XDR'))

and the fixture records the two encodings or Spark's error condition and message parameters. Nothing is
transcribed by hand. Output is deterministic and contains no local paths.

Usage (from the repo root, with the pyspark 4.2.0 package on the path and Java 17 or 21):

    PYTHONPATH=<dir containing pyspark> python3 tests/scripts/gen_wkb_goldens.py \
        --out tests/fixtures/spark-4.2.0/wkb-goldens.json
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_wkb_goldens.py"
NAN = float("nan")
INF = float("inf")


def le(t: int, *coords: float) -> bytes:
    return struct.pack("<BI", 1, t) + b"".join(struct.pack("<d", c) for c in coords)


def be(t: int, *coords: float) -> bytes:
    return struct.pack(">BI", 0, t) + b"".join(struct.pack(">d", c) for c in coords)


def u32(n: int) -> bytes:
    return struct.pack("<I", n)


def d(*cs: float) -> bytes:
    return b"".join(struct.pack("<d", c) for c in cs)


P = le(1, 1, 2)
SQUARE = d(0, 0, 1, 0, 1, 1, 0, 0)


def nest(n: int) -> bytes:
    b = P
    for _ in range(n):
        b = le(7) + u32(1) + b
    return b


CASES: dict[str, bytes] = {
    # headers and byte orders
    "empty": b"",
    "one_byte": b"\x01",
    "four_bytes": le(1)[:4],
    "byte_order_2": b"\x02" + u32(1) + d(1, 2),
    "byte_order_6f": b"\x6f",
    "point": P,
    "point_xdr": be(1, 1, 2),
    "point_trailing_byte": P + b"\x00",
    "point_truncated_x": le(1),
    "point_truncated_y": P[:-1],
    # type codes
    "type_0": le(0, 1, 2),
    "type_8": le(8, 1, 2),
    "type_17": le(17, 1, 2),
    "type_4001": le(4001, 1, 2),
    "ewkb_z_flag": le(0x80000001, 1, 2, 3),
    "ewkb_srid_flag": struct.pack("<BII", 1, 0x20000001, 4326) + d(1, 2),
    "ewkb_z_flag_xdr": struct.pack(">BI", 0, 0x80000001) + struct.pack(">dd", 1, 2),
    "point_z": le(1001, 1, 2, 3),
    "point_m": le(2001, 1, 2, 3),
    "point_zm": le(3001, 1, 2, 3, 4),
    "point_z_xdr": struct.pack(">BI", 0, 1001) + struct.pack(">ddd", 1, 2, 3),
    # ordinates
    "point_empty": le(1, NAN, NAN),
    "point_nan_y": le(1, 1, NAN),
    "point_nan_payload": struct.pack("<BIQQ", 1, 1, 0x7FF0000000000001, 0xFFF8000000000000),
    "point_inf_x": le(1, INF, 2),
    "point_inf_y": le(1, 1, INF),
    "point_ninf_y": le(1, 1, -INF),
    "point_z_inf": le(1001, 1, 2, INF),
    "point_m_nan": le(2001, 1, 2, NAN),
    "point_z_empty_xy": le(1001, NAN, NAN, 1),
    # line strings
    "line_empty": le(2) + u32(0),
    "line_one_point": le(2) + u32(1) + d(1, 2),
    "line": le(2) + u32(2) + d(1, 2, 3, 4),
    "line_count_minus_1": le(2) + u32(0xFFFFFFFF) + d(1, 2),
    "line_count_min": le(2) + u32(0x80000000),
    "line_count_exceeds": le(2) + u32(2) + d(1, 2, 3),
    "line_count_exceeds_one": le(2) + u32(1) + d(1),
    "line_no_count": le(2),
    "line_short_count": le(2) + b"\x02\x00",
    "line_nan_point": le(2) + u32(2) + d(NAN, NAN, 1, 2),
    "line_nan_y": le(2) + u32(2) + d(1, NAN, 1, 2),
    "line_one_nan_point": le(2) + u32(1) + d(NAN, NAN),
    "line_z_nan_z": le(1002) + u32(2) + d(1, 2, NAN, 3, 4, 5),
    "line_m_nan_m": le(2002) + u32(2) + d(1, 2, NAN, 3, 4, 5),
    # polygons
    "polygon_empty": le(3) + u32(0),
    "polygon": le(3) + u32(1) + u32(4) + SQUARE,
    "polygon_no_count": le(3),
    "polygon_rings_minus_1": le(3) + u32(0xFFFFFFFF),
    "polygon_rings_exceed": le(3) + u32(2),
    "polygon_ring_exceeds_one": le(3) + u32(1),
    "polygon_ring_points_exceed": le(3) + u32(1) + u32(4),
    "polygon_ring_points_minus_1": le(3) + u32(1) + u32(0xFFFFFFFF),
    "polygon_ring_points_short": le(3) + u32(1) + u32(3) + d(1),
    "polygon_second_ring_missing": le(3) + u32(2) + u32(4) + SQUARE,
    "ring_empty": le(3) + u32(1) + u32(0),
    "ring_two_points": le(3) + u32(1) + u32(2) + d(0, 0, 0, 0),
    "ring_three_points": le(3) + u32(1) + u32(3) + d(0, 0, 1, 0, 0, 0),
    "ring_open": le(3) + u32(1) + u32(4) + d(0, 0, 1, 0, 1, 1, 0, 1),
    "ring_nan": le(3) + u32(1) + u32(4) + d(NAN, NAN, 1, 0, 1, 1, NAN, NAN),
    "ring_negative_zero": le(3) + u32(1) + u32(4) + d(0.0, 0, 1, 0, 1, 1, -0.0, 0),
    "ring_z_open_in_z": le(1003) + u32(1) + u32(4) + d(0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 5),
    "ring_m_open_in_m": le(2003) + u32(1) + u32(4) + d(0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 5),
    "polygon_second_ring_short": le(3) + u32(2) + u32(4) + SQUARE + u32(3) + d(0, 0, 1, 0, 0, 0),
    # collections
    "multipoint_empty": le(4) + u32(0),
    "multipoint": le(4) + u32(1) + P,
    "multipoint_xdr_child": le(4) + u32(1) + be(1, 1, 2),
    "multipoint_empty_point": le(4) + u32(1) + le(1, NAN, NAN),
    "multipoint_line_child": le(4) + u32(1) + le(2) + u32(2) + d(1, 2, 3, 4),
    "multipoint_z_2d_child": le(1004) + u32(1) + P,
    "multipoint_z": le(1004) + u32(1) + le(1001, 1, 2, 3),
    "multipoint_count_minus_1": le(4) + u32(0xFFFFFFFF),
    "multipoint_count_2_of_1": le(4) + u32(2) + P,
    "multipoint_count_4_of_1": le(4) + u32(4) + P,
    "multipoint_count_5_of_1": le(4) + u32(5) + P,
    "multipoint_child_trailing": le(4) + u32(1) + P + b"\x00\x00",
    "multiline": le(5) + u32(1) + le(2) + u32(2) + d(1, 2, 3, 4),
    "multiline_point_child": le(5) + u32(1) + P,
    "multiline_count_2_of_1": le(5) + u32(2) + le(2) + u32(0),
    "multipolygon": le(6) + u32(1) + le(3) + u32(1) + u32(4) + SQUARE,
    "multipolygon_point_child": le(6) + u32(1) + P,
    "multipolygon_count_2_of_1": le(6) + u32(2) + le(3) + u32(0),
    "collection_empty": le(7) + u32(0),
    "collection": le(7) + u32(1) + P,
    "collection_nested": le(7) + u32(1) + le(7) + u32(1) + P,
    "collection_count_2_of_1": le(7) + u32(2) + P,
    "collection_count_4_of_1": le(7) + u32(4) + P,
    "collection_count_5_of_1": le(7) + u32(5) + P,
    "collection_count_minus_1": le(7) + u32(0xFFFFFFFF),
    "collection_bad_child": le(7) + u32(1) + b"\x05",
    "collection_bad_nested_type": le(7) + u32(1) + le(7) + u32(1) + le(9, 1, 2),
    "collection_short_line": le(7) + u32(1) + le(2) + u32(1) + d(1, 2),
    "collection_truncated_line": le(7) + u32(1) + le(2) + u32(2) + d(1, 2, 3),
    "collection_z_2d_child": le(1007) + u32(1) + P,
    "collection_zm_child": le(7) + u32(1) + le(3001, 1, 2, 3, 4),
    "collection_nested_200": nest(200),
    # geography bounds
    "lon_200_lat_100": le(1, 200, 100),
    "lat_95": le(1, 10, 95),
    "lon_180_lat_90": le(1, 180, 90),
    "lon_minus_180_lat_minus_90": le(1, -180, -90),
    "lon_just_over_180": le(1, 180.0000001, 0),
    "lat_minus_90_5": le(1, 0, -90.5),
    "lon_nan_lat_1": le(1, NAN, 1),
    "z_1e300": le(1001, 1, 2, 1e300),
    "m_inf": le(2001, 1, 2, INF),
    "lon_out_and_m_inf": le(2001, 200, 0, INF),
    "line_second_lon_out": le(2) + u32(2) + d(1, 2, 200, 4),
    "line_second_lat_out": le(2) + u32(2) + d(1, 2, 3, 100),
}


def outcome(spark, sql: str) -> dict:
    try:
        row = spark.sql(sql).collect()[0]
        return {"ndr": row[0], "xdr": row[1]}
    except Exception as e:
        get_condition = getattr(e, "getCondition", None)
        get_params = getattr(e, "getMessageParameters", None)
        cond = get_condition() if callable(get_condition) else None
        if cond is None or not callable(get_params):
            return {"error_class": "UNCLASSIFIED"}
        params = get_params() or {}
        return {"error_class": cond, "message_parameters": dict(sorted(params.items()))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/wkb-goldens.json")
    args = ap.parse_args()

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
    spark.sparkContext.setLogLevel("OFF")
    cases = []
    for name, wkb in CASES.items():
        h = wkb.hex().upper()
        case = {"name": name, "wkb": h}
        for kind, fn in (("geometry", "ST_GeomFromWKB"), ("geography", "ST_GeogFromWKB")):
            sql = f"SELECT hex(ST_AsBinary({fn}(X'{h}'))), hex(ST_AsBinary({fn}(X'{h}'), 'XDR'))"
            case[kind] = outcome(spark, sql)
        cases.append(case)
    doc = {
        "format_version": FORMAT_VERSION,
        "spark_version": spark.version,
        "generator": GENERATOR,
        "notes": "Each case is evaluated as ST_AsBinary(ST_GeomFromWKB(wkb)) and ST_GeogFromWKB, in NDR "
        "and XDR; errors carry Spark's condition and message parameters.",
        "cases": cases,
    }
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases against Spark {spark.version} -> {args.out}")


if __name__ == "__main__":
    main()
