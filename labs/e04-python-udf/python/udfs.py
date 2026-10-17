"""UDF bodies. cheap = Int64 x+1 ; expensive = Utf8 -> Int64 count of digit runs (regex)."""
import re
import ctypes
import pyarrow as pa
import pyarrow.compute as pc

PAT = re.compile(r"\d+")


# ---- scalar (row-at-a-time)
def scalar_cheap(x):
    return x + 1


def scalar_expensive(s):
    return len(PAT.findall(s))


# ---- pandas (Series -> Series)
def pandas_cheap(s):
    return s + 1


def pandas_expensive(s):
    return s.str.count(r"\d+")


# ---- arrow (pa.Array -> pa.Array)
def arrow_cheap(a):
    return pc.add(a, 1)


def arrow_expensive(a):
    return pc.count_substring_regex(a, r"\d+").cast(pa.int64())


def crash_udf(a):
    """Arrow UDF that segfaults on a poison batch (first value == -1)."""
    if a[0].as_py() == -1:
        ctypes.string_at(0)
    return pc.add(a, 1)



def make_batch_fn(kind, body):
    """Return fn(arrow_like) -> pa.Array for (kind, body)."""
    if kind == "crash":
        return lambda w: crash_udf(pa.array(w))
    f = {"scalar": (scalar_cheap, scalar_expensive),
         "pandas": (pandas_cheap, pandas_expensive),
         "arrow": (arrow_cheap, arrow_expensive)}[kind][0 if body == "cheap" else 1]
    if kind == "scalar":
        def run(w):
            a = pa.array(w)
            return pa.array([f(v) for v in a.to_pylist()], type=pa.int64())
    elif kind == "pandas":
        def run(w):
            a = pa.array(w)
            r = f(a.to_pandas())
            return pa.Array.from_pandas(r).cast(pa.int64())
    else:
        def run(w):
            return f(pa.array(w))
    return run
