A zero-byte file in a directory no longer fails a read.

CSV, JSON and text files compressed with Zstandard (`.zst`, `.zstd`) and files named `.gzip` are read, as Spark 4.1 and later read them, and the 4.0 and later profiles match compression suffixes in any case. The 3.5 and 4.0 profiles keep their releases' suffixes (`krishiv.io.text.compressionSuffixes`).
