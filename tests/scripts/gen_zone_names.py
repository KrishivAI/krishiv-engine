#!/usr/bin/env python3
"""Generate the zone display names Spark prints for `date_format(ts, 'z')` and `'zzzz'` (kernel `zone_name`).

Runs a REAL Spark (never from memory) in a throw-away container. For every region id the JVM knows it sets
`spark.sql.session.timeZone` to the id and formats one standard-time and one daylight-time instant of 2024
(chosen by `ZoneRules.isDaylightSavings`, as `java.time` chooses the name). A zone with no daylight time in
2024 takes its daylight instant from its historical transitions; only a zone that never had daylight time
repeats the standard names. Usage (from repo root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[2]' /work/gen_zone_names.py \
        > tests/fixtures/spark-4.2.0/zone-names.json

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import json
import sys

from pyspark.sql import SparkSession

DAY = 86400


def main() -> None:
    spark = SparkSession.builder.appName("zone-names").getOrCreate()
    jvm = spark._jvm
    ids = sorted(str(i) for i in jvm.java.time.ZoneId.getAvailableZoneIds())
    # 1st and 15th of every month of 2024 (epoch seconds).
    base = 1704067200
    cuts = [base + d * DAY for d in range(0, 366, 15)]
    zones = {}
    for zid in ids:
        rules = jvm.java.time.ZoneId.of(zid).getRules()
        std = dst = None
        for t in cuts:
            if rules.isDaylightSavings(jvm.java.time.Instant.ofEpochSecond(t)):
                dst = dst or t
            else:
                std = std or t
        if dst is None:
            # No daylight time in 2024: look through the zone's historical transitions. (A zone that is
            # "daylight" all of 2024, as Namibia is in Java's rearguard data, keeps 2024 for both slots.)
            for tr in rules.getTransitions():
                inst = tr.getInstant()
                for t in (inst.getEpochSecond() - 1, inst.getEpochSecond()):
                    if rules.isDaylightSavings(jvm.java.time.Instant.ofEpochSecond(t)):
                        dst = dst or t
        std = std or dst
        dst = dst or std
        spark.conf.set("spark.sql.session.timeZone", zid)
        row = spark.sql(
            f"select date_format(timestamp_seconds({std}), 'z'), date_format(timestamp_seconds({std}), 'zzzz'),"
            f" date_format(timestamp_seconds({dst}), 'z'), date_format(timestamp_seconds({dst}), 'zzzz')"
        ).first()
        zones[zid] = {"std_at": std, "dst_at": dst, "names": list(row)}
    json.dump({"spark_version": spark.version, "zones": zones}, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
