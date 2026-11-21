// SPDX-License-Identifier: Apache-2.0

// Smoke run of the Apache Spark Connect Go client against a Spark Connect server
// (Krishiv, or real Spark for a baseline). Report-only: it always writes its
// results file and exits 0 unless the file cannot be written.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"reflect"
	"strings"
	"time"

	"github.com/apache/spark-connect-go/v40/spark/sql"
)

type query struct {
	Name    string     `json:"name"`
	SQL     string     `json:"sql"`
	Columns []string   `json:"columns,omitempty"`
	Rows    [][]string `json:"rows,omitempty"`
	Error   string     `json:"error,omitempty"`
}

type result struct {
	Name   string `json:"name"`
	Status string `json:"status"` // pass | fail | error
	Detail string `json:"detail,omitempty"`
	Millis int64  `json:"ms"`
}

type report struct {
	Client  string   `json:"client"`
	Version string   `json:"version"`
	Remote  string   `json:"remote"`
	Results []result `json:"results"`
}

// render is the shared text form of a value: NULL for null, fmt's %v otherwise
// (the Scala client's String.valueOf agrees for the types the queries return).
func render(v any) string {
	if v == nil {
		return "NULL"
	}
	return fmt.Sprintf("%v", v)
}

func runQuery(ctx context.Context, spark sql.SparkSession, q query) (string, string) {
	df, err := spark.Sql(ctx, q.SQL)
	var rows [][]string
	var cols []string
	if err == nil {
		got, cerr := df.Collect(ctx)
		err = cerr
		for _, r := range got {
			cols = r.FieldNames()
			line := make([]string, 0, r.Len())
			for _, v := range r.Values() {
				line = append(line, render(v))
			}
			rows = append(rows, line)
		}
	}
	if q.Error != "" {
		if err == nil {
			return "fail", "expected error " + q.Error + ", the query succeeded"
		}
		if !strings.Contains(err.Error(), q.Error) {
			return "fail", "expected error " + q.Error + ", got: " + firstLine(err.Error())
		}
		return "pass", ""
	}
	if err != nil {
		return "error", firstLine(err.Error())
	}
	if len(rows) == 0 && len(q.Rows) == 0 {
		return "pass", ""
	}
	if !reflect.DeepEqual(rows, q.Rows) {
		return "fail", fmt.Sprintf("rows: want %v, got %v", q.Rows, rows)
	}
	if len(q.Columns) > 0 && !reflect.DeepEqual(cols, q.Columns) {
		return "fail", fmt.Sprintf("columns: want %v, got %v", q.Columns, cols)
	}
	return "pass", ""
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if len(s) > 300 {
		s = s[:300]
	}
	return s
}

// apiChecks are client calls that are not a plain SQL string.
func apiChecks(ctx context.Context, spark sql.SparkSession) []result {
	type check struct {
		name string
		run  func() (string, string)
	}
	checks := []check{
		{"df_count", func() (string, string) {
			df, err := spark.Sql(ctx, "SELECT id FROM range(25)")
			if err != nil {
				return "error", firstLine(err.Error())
			}
			n, err := df.Count(ctx)
			if err != nil {
				return "error", firstLine(err.Error())
			}
			if n != 25 {
				return "fail", fmt.Sprintf("count: want 25, got %d", n)
			}
			return "pass", ""
		}},
		{"df_schema_names", func() (string, string) {
			df, err := spark.Sql(ctx, "SELECT 1 AS a, 'x' AS b")
			if err != nil {
				return "error", firstLine(err.Error())
			}
			schema, err := df.Schema(ctx)
			if err != nil {
				return "error", firstLine(err.Error())
			}
			var names []string
			for _, f := range schema.Fields {
				names = append(names, f.Name)
			}
			if !reflect.DeepEqual(names, []string{"a", "b"}) {
				return "fail", fmt.Sprintf("field names: %v", names)
			}
			return "pass", ""
		}},
		{"config_round_trip", func() (string, string) {
			cfg := spark.Config()
			if err := cfg.Set(ctx, "spark.sql.shuffle.partitions", "7"); err != nil {
				return "error", firstLine(err.Error())
			}
			got, err := cfg.Get(ctx, "spark.sql.shuffle.partitions")
			if err != nil {
				return "error", firstLine(err.Error())
			}
			if got != "7" {
				return "fail", "config value: " + got
			}
			return "pass", ""
		}},
		{"values_collect", func() (string, string) {
			df, err := spark.Sql(ctx, "SELECT * FROM VALUES (1, 'a'), (2, 'b') AS t(i, s)")
			if err != nil {
				return "error", firstLine(err.Error())
			}
			rows, err := df.Collect(ctx)
			if err != nil {
				return "error", firstLine(err.Error())
			}
			if len(rows) != 2 {
				return "fail", fmt.Sprintf("rows: %d", len(rows))
			}
			return "pass", ""
		}},
	}
	var out []result
	for _, c := range checks {
		start := time.Now()
		status, detail := c.run()
		out = append(out, result{Name: "api_" + c.name, Status: status, Detail: detail, Millis: time.Since(start).Milliseconds()})
	}
	return out
}

func main() {
	remote := flag.String("remote", "sc://localhost:15002", "Spark Connect address")
	queries := flag.String("queries", "queries.json", "shared query file")
	out := flag.String("out", "go.json", "results file")
	flag.Parse()
	ctx := context.Background()

	rep := report{Client: "go", Version: "spark-connect-go v40", Remote: *remote}
	raw, err := os.ReadFile(*queries)
	if err != nil {
		fmt.Fprintln(os.Stderr, "queries:", err)
		os.Exit(2)
	}
	var qs []query
	if err := json.Unmarshal(raw, &qs); err != nil {
		fmt.Fprintln(os.Stderr, "queries:", err)
		os.Exit(2)
	}

	spark, err := sql.NewSessionBuilder().Remote(*remote).Build(ctx)
	if err != nil {
		rep.Results = append(rep.Results, result{Name: "connect", Status: "error", Detail: firstLine(err.Error())})
	} else {
		defer func() { _ = spark.Stop() }()
		for _, q := range qs {
			start := time.Now()
			status, detail := runQuery(ctx, spark, q)
			rep.Results = append(rep.Results, result{Name: q.Name, Status: status, Detail: detail, Millis: time.Since(start).Milliseconds()})
		}
		rep.Results = append(rep.Results, apiChecks(ctx, spark)...)
	}

	body, _ := json.MarshalIndent(rep, "", "  ")
	if err := os.WriteFile(*out, body, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write:", err)
		os.Exit(2)
	}
	pass := 0
	for _, r := range rep.Results {
		if r.Status == "pass" {
			pass++
		}
	}
	fmt.Printf("go client: %d/%d pass\n", pass, len(rep.Results))
}
