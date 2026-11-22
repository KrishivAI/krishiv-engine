---
description: "Read and write Parquet, CSV and other files on local disk and on S3, Google Cloud Storage and Azure Blob Storage."
---

# Data sources

Krishiv reads and writes files from your own disk (below) and, from 0.2.1, from cloud object stores: Amazon S3 (`s3://`, `s3a://`),
Google Cloud Storage (`gs://`) and Azure Blob Storage (`az://`, `abfs://`, `abfss://`). The wheels from PyPI include the
object-store support; the 0.2.0 wheels did not, and failed with `KRISHIV_UNSUPPORTED_FEATURE` on these URLs.

```python
import krishiv

spark = krishiv.session()
spark.conf.set("fs.s3a.endpoint.region", "eu-west-1")
df = spark.read.parquet("s3://my-bucket/events/")
df.write.mode("overwrite").parquet("s3://my-bucket/summary/")
```

## Read and write local files

A CSV file with a header row, read with its column types inferred, summed by group and written back as Parquet:

```python
import krishiv

spark = krishiv.session()
df = spark.read.csv("trips.csv", header=True, inferSchema=True)
summary = df.groupBy("region").sum("amount")
summary.write.mode("overwrite").parquet("summary/")
print(sorted(tuple(r) for r in spark.read.parquet("summary/").collect()))
# [('east', 15.5), ('west', 20.0)]
```

The output above is for a `trips.csv` of `region,amount` followed by the rows `east,10.0`, `west,20.0` and `east,5.5`.

`spark.read.json` and `spark.read.parquet` work the same way, and `df.write` has `parquet`, `csv`
and `json`, with `.mode(...)` and `.option(key, value)` such as `.option("header", True)` for CSV.

The save modes are Spark's:

| Mode | When the path already exists |
|---|---|
| `error`, `errorifexists` (the default) | the write fails with `PATH_ALREADY_EXISTS` |
| `ignore` | nothing is written and no error is raised |
| `append` | new files are added next to the existing ones |
| `overwrite` | everything under the path is replaced; a failed overwrite keeps the old data |

A write creates a directory, not a single file: `part-*` data files, a `_SUCCESS` marker and a `_krishiv_log/`
directory that records each commit (see [known differences](../concepts/known-differences.md#writers-leave-a-_krishiv_log-directory)).
To write a new directory without it, set `krishiv.sql.files.commitProtocol` to `NONE` on the session
(`spark.conf.set("krishiv.sql.files.commitProtocol", "NONE")`); the data reads back the same. A directory that already
has a `_krishiv_log/` keeps using it.

## Credentials and options

Set the options on the session with `spark.conf.set`, or for one read or write with `.option(key, value)`. A per-read
option overrides the session value, and two reads of one bucket with different credentials never share a connection.
Keys match without regard to case. An option value is never printed: errors and `EXPLAIN` name the key only.

When an option is not set, Krishiv falls back to the standard environment variables of each cloud (`AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `GOOGLE_APPLICATION_CREDENTIALS`, `AZURE_STORAGE_ACCOUNT_NAME`, and so on).

### Amazon S3 and S3-compatible stores

| Option | Meaning |
|---|---|
| `fs.s3a.access.key`, `fs.s3a.secret.key` | access key id and secret |
| `fs.s3a.session.token` | session token, for temporary credentials |
| `fs.s3a.endpoint.region` (or `fs.s3a.region`) | region; the default is `us-east-1` |
| `fs.s3a.endpoint` | endpoint URL for MinIO, LocalStack, Ceph and similar; `http://` is allowed |
| `fs.s3a.path.style.access` | `true` forces `https://host/bucket/key` requests; already the default for `http://` and `localhost` endpoints |

### Google Cloud Storage

| Option | Meaning |
|---|---|
| `fs.gs.auth.service.account.json.keyfile` | path to a service-account JSON file |
| `fs.gs.auth.service.account.json` | the service-account JSON itself |
| `fs.gs.storage.root.url` (or `fs.gs.endpoint`) | endpoint URL for an emulator |

### Azure Blob Storage

| Option | Meaning |
|---|---|
| `fs.azure.account.key.<account>.dfs.core.windows.net` | storage account key (the `.blob.` form works too) |
| `fs.azure.sas.<container>.<account>.dfs.core.windows.net` | shared access signature for one container |
| `fs.azure.account.name` | account name, when the URL is `az://container/path` and does not carry it |
| `fs.azure.endpoint` | endpoint URL for Azurite (`http://127.0.0.1:10000/<account>`) |

### Hugging Face datasets (`hf://`)

A Krishiv extension, not in Spark: `hf://datasets/<org>/<name>[@revision]/<path>` reads one named file through the
Hugging Face `resolve` endpoint (`main` unless you write `@revision`).

```python
spark.read.parquet("hf://datasets/cornell-movie-review-data/rotten_tomatoes/train.parquet").show()
```

| Option | Meaning |
|---|---|
| `fs.hf.token` (or the `HF_TOKEN` environment variable) | access token, sent as a bearer header; needed for private datasets. `HF_TOKEN` is used only for the default host: with `fs.hf.endpoint` set, only a per-read `fs.hf.token` is sent |
| `fs.hf.endpoint` | host to use instead of `https://huggingface.co` |

Name the file: a directory, a glob and Hugging Face's generated Parquet branch (`@~parquet/...`) are not supported yet.

### `http://` and `https://`

Spark has no working `http://` or `https://` file system. When Krishiv has to infer the schema of such a location, it
refuses it with `FAILED_READ_FILE.UNSUPPORTED_FILE_SYSTEM`, the error Spark gives. A read that gives its own schema is not
refused by that check and goes to the HTTP object store, which needs a server that supports range requests; do not rely on
it. On the server, `http://` and `https://` locations also need an entry in `allowedPrefixes`.
Download the file, or use an object store or `hf://`.

## Try it without a cloud account

Run MinIO locally, create a bucket as a directory of its data folder, and point `fs.s3a.endpoint` at it:

```bash
mkdir -p data/my-bucket
docker run --rm -p 127.0.0.1:9000:9000 -v "$PWD/data:/data" quay.io/minio/minio server /data
```

```python
for key, value in {
    "fs.s3a.endpoint": "http://127.0.0.1:9000",
    "fs.s3a.access.key": "minioadmin",
    "fs.s3a.secret.key": "minioadmin",
}.items():
    spark.conf.set(key, value)
spark.sql("select 1 as id, 'a' as kind").write.parquet("s3://my-bucket/events/")
spark.read.parquet("s3://my-bucket/events/").show()
```

## On a shared server

In server mode with authentication on, reads and writes by path are confined to the prefixes in
`krishiv.io.objectStore.allowedPrefixes` and `krishiv.io.local.allowedRoots`; see the
[configuration reference](../reference/config/index.md).
