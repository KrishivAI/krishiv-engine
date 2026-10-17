"""Option C: code run inside a PEP 734 subinterpreter (concurrent.interpreters, Python 3.14).
pyarrow / numpy / pandas cannot be imported in subinterpreters (single-phase init), so only a
pure-Python scalar UDF over raw Arrow buffers (shared as memoryviews) is possible."""
CODE = r'''
import re, ctypes
PAT = re.compile(r"\d+")
def cheap(inp, out):
    inp = inp.cast("q"); out = out.cast("q")
    for k in range(len(inp)):
        out[k] = inp[k] + 1
def expensive(offs, data, out):
    offs = offs.cast("i"); out = out.cast("q")
    for k in range(len(out)):
        s = data[offs[k]:offs[k+1]].tobytes().decode()
        out[k] = len(PAT.findall(s))
def crash(inp, out):
    inp = inp.cast("q"); out = out.cast("q")
    if inp[0] == -1:
        ctypes.string_at(0)
    for k in range(len(inp)):
        out[k] = inp[k] + 1
'''
def create():
    from concurrent import interpreters
    i = interpreters.create()
    i.exec(CODE)
    return i
