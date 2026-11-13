# Run SQL from the command line

Use `krishiv sql` to run one or more statements and print the result. This page shows the three ways to give it SQL and the output formats. The output below was captured from a real run.

## Give it SQL

Pass statements with `-e`, separated by `;`:

```bash
krishiv sql -e "SELECT 1 AS id, 'a' AS name UNION ALL SELECT 2, 'b'"
```

Read them from a file with `-f`, or from standard input with `-f -`. With neither `-e` nor `-f`, the command reads standard input.

## Pick an output format

The default is an aligned table:

```text
+----+------+
| id | name |
+----+------+
| 1  | a    |
| 2  | b    |
+----+------+
2 rows
```

Add `-o csv` for CSV with a header row:

```text
id,name
1,a
2,b
```

Add `-o json` for one JSON object per row:

```text
{"id":1,"name":"a"}
{"id":2,"name":"b"}
```

The other formats are `tsv`, `jsonl`, `markdown` and `line`.

## Use it from a script or an agent

When standard output is not a terminal and you give no `--format`, the command prints the compact `krishiv.agent-result` table, and errors as one JSON object each. Pass `--agent` to ask for this on a terminal.
