"""Option D worker: reads Arrow IPC from shared memory (memfd), runs UDF, writes Arrow IPC back to shm.
Control: u64 LE length on stdin -> u64 LE reply length on stdout (2**64-1 = error)."""
import sys, os, mmap, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow as pa
import udfs

fd, region = int(sys.argv[1]), int(sys.argv[2])
kind, body = sys.argv[3], sys.argv[4]
mm = mmap.mmap(fd, 2 * region)
mv = memoryview(mm)
out_buf = pa.py_buffer(mv[region:2 * region])
fn = udfs.make_batch_fn(kind, body)
rin, wout = sys.stdin.buffer, sys.stdout.buffer
ERR = struct.pack("<Q", 2**64 - 1)

class Wrap:  # lets make_batch_fn's pa.array(w) accept an already-built array
    def __init__(self, a): self.a = a
    def __arrow_c_array__(self, requested_schema=None): return self.a.__arrow_c_array__(requested_schema)

wout.write(b"R"); wout.flush()
while True:
    hdr = rin.read(8)
    if len(hdr) < 8:
        break
    n = struct.unpack("<Q", hdr)[0]
    try:
        reader = pa.ipc.open_stream(pa.py_buffer(mv[:n]))
        batch = reader.read_next_batch()
        res = fn(Wrap(batch.column(0)))
        sink = pa.FixedSizeBufferWriter(out_buf)
        w = pa.ipc.new_stream(sink, pa.schema([("r", pa.int64())]))
        w.write_batch(pa.record_batch([res], names=["r"]))
        w.close()
        wout.write(struct.pack("<Q", sink.tell()))
    except Exception as e:
        sys.stderr.write(f"worker error: {e!r}\n"); sys.stderr.flush()
        wout.write(ERR)
    wout.flush()
