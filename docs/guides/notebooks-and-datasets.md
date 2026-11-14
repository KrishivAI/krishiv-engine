---
description: "Sample notebooks and datasets to try Krishiv on."
---

# Notebooks and sample data

Three notebooks in [`docs/notebooks`](https://github.com/KrishivAI/krishiv-engine/tree/main/docs/notebooks) walk through the quickstart's ideas on generated order data:

1. `01-first-queries`: SQL over Parquet, filters and counts.
2. `02-dataframes-and-joins`: the same query as SQL and as DataFrame calls, and the Arrow result.
3. `03-explain-and-check-compat`: reading a plan with `explain(analyze=True)`, and where to scan a job with `krishiv check-compat` (a shell command, shown there as text).

Install, then open them with any Jupyter:

```bash
pip install "krishiv[pyarrow]" jupyter
jupyter lab docs/notebooks
```

Each notebook generates its own data with a small script, so nothing is downloaded and no data file is stored in the repository. To make the files yourself:

```bash
python docs/datasets/generate.py ./data            # orders.parquet (200,000 rows) and customers.parquet, about 5 MB
python docs/datasets/generate.py ./data --orders 5000000   # a larger set
```

The data is deterministic: the same arguments always give the same rows. Notebooks are committed with their outputs stripped. CI runs each one headless against the built wheel on changes to the Python package or the engine; the link check and the generated migration guide are checked on every change.
