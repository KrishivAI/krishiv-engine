//! The four execution options behind a common `Runner` trait.
use crate::bridge::{import_array, ArrowExport};
use crate::{check, Batch};
use arrow::array::{Array, ArrayRef, Int64Array, StringArray};
use arrow::datatypes::{Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use memmap2::MmapMut;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::io::{Cursor, Read, Write};
use std::os::fd::FromRawFd;
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Arc;

#[derive(Clone)]
pub struct Spec {
    pub option: String,
    pub kind: String,
    pub body: String,
    pub python: String,
    pub worker: String,
}

pub trait Runner: Send {
    fn run(&mut self, col: &ArrayRef) -> Result<ArrayRef, String>;
    fn pids(&self) -> Vec<u32> {
        vec![]
    }
    fn pin_thread(&self) -> Option<Box<dyn std::any::Any>> {
        None
    }
    fn recover(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn shutdown(&mut self) {}
}

pub fn make(spec: &Spec) -> Result<Box<dyn Runner>, String> {
    match spec.option.as_str() {
        "A" | "B" => Ok(Box::new(PyRunner::new(spec).map_err(|e| e.to_string())?)),
        "C" => Ok(Box::new(SubRunner::new(spec)?)),
        "D" => Ok(Box::new(ProcRunner::new(spec)?)),
        o => Err(format!("unknown option {o}")),
    }
}

// ---------------------------------------------------------------- A / B: in-process
pub struct PyRunner {
    f: Py<PyAny>,
}
impl PyRunner {
    fn new(spec: &Spec) -> PyResult<Self> {
        Python::attach(|py| {
            let m = py.import("udfs")?;
            let f = m.call_method1("make_batch_fn", (&spec.kind, &spec.body))?;
            Ok(Self { f: f.unbind() })
        })
    }
}
impl Runner for PyRunner {
    fn run(&mut self, col: &ArrayRef) -> Result<ArrayRef, String> {
        Python::attach(|py| -> PyResult<ArrayRef> {
            let w = Bound::new(py, ArrowExport::new(col))?;
            let r = self.f.bind(py).call1((w,))?;
            import_array(&r)
        })
        .map_err(|e| e.to_string())
    }
    fn pin_thread(&self) -> Option<Box<dyn std::any::Any>> {
        PyThreadPin::new()
    }
}

/// Keeps a worker thread's PyThreadState alive (detached) for the thread's lifetime. Without it,
/// every `Python::attach` creates and destroys a thread state (PyGILState_Ensure/Release at depth 0),
/// which costs ~1.8 ms/call on free-threaded 3.14t (measured) and ~tens of us on the GIL build.
pub struct PyThreadPin {
    state: ffi::PyGILState_STATE,
    saved: *mut ffi::PyThreadState,
}
impl PyThreadPin {
    pub fn new() -> Option<Box<dyn std::any::Any>> {
        if NO_PIN.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        // SAFETY: standard Ensure + SaveThread pair, undone in Drop on the same thread.
        unsafe {
            let state = ffi::PyGILState_Ensure();
            let saved = ffi::PyEval_SaveThread();
            Some(Box::new(PyThreadPin { state, saved }))
        }
    }
}
impl Drop for PyThreadPin {
    fn drop(&mut self) {
        // SAFETY: restores the thread state saved in `new` on the same thread.
        unsafe {
            ffi::PyEval_RestoreThread(self.saved);
            ffi::PyGILState_Release(self.state);
        }
    }
}
pub static NO_PIN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

// ---------------------------------------------------------------- C: subinterpreters
pub struct SubRunner {
    interp: Py<PyAny>,
    kind: String,
    body: String,
}
impl SubRunner {
    fn new(spec: &Spec) -> Result<Self, String> {
        if spec.kind != "scalar" && spec.kind != "crash" {
            return Err(format!(
                "pyarrow/numpy/pandas cannot be imported in subinterpreters ('module pyarrow.lib does not support loading in subinterpreters'); only pure-Python scalar UDFs are possible, kind={} unsupported",
                spec.kind
            ));
        }
        Python::attach(|py| -> PyResult<Self> {
            let i = py.import("subi")?.call_method0("create")?;
            Ok(Self { interp: i.unbind(), kind: spec.kind.clone(), body: spec.body.clone() })
        })
        .map_err(|e| e.to_string())
    }
}
fn mview<'py>(py: Python<'py>, p: *const u8, bytes: usize, write: bool) -> PyResult<Bound<'py, PyAny>> {
    let flags = if write { ffi::PyBUF_WRITE } else { ffi::PyBUF_READ };
    // SAFETY: memory outlives the call (inputs are owned by the pool/caller, output Vec by this frame);
    // the views are dropped inside the subinterpreter before the call returns.
    unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyMemoryView_FromMemory(p as *mut _, bytes as isize, flags)) }
}
impl Runner for SubRunner {
    fn run(&mut self, col: &ArrayRef) -> Result<ArrayRef, String> {
        let n = col.len();
        let mut out = vec![0i64; n];
        Python::attach(|py| -> PyResult<()> {
            let kw = PyDict::new(py);
            let code;
            if self.body == "expensive" {
                let s = col.as_any().downcast_ref::<StringArray>().ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("need utf8"))?;
                let offs = s.value_offsets();
                kw.set_item("offs", mview(py, offs.as_ptr() as *const u8, offs.len() * 4, false)?)?;
                kw.set_item("data", mview(py, s.value_data().as_ptr(), s.value_data().len(), false)?)?;
                code = "expensive(offs, data, out); offs=data=out=None";
            } else {
                let a = col.as_any().downcast_ref::<Int64Array>().ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("need int64"))?;
                kw.set_item("inp", mview(py, a.values().as_ptr() as *const u8, n * 8, false)?)?;
                code = if self.kind == "crash" { "crash(inp, out); inp=out=None" } else { "cheap(inp, out); inp=out=None" };
            }
            kw.set_item("out", mview(py, out.as_mut_ptr() as *const u8, n * 8, true)?)?;
            let i = self.interp.bind(py);
            i.call_method("prepare_main", (), Some(&kw))?;
            i.call_method1("exec", (code,))?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
        Ok(Arc::new(Int64Array::from(out)))
    }
    fn pin_thread(&self) -> Option<Box<dyn std::any::Any>> {
        PyThreadPin::new()
    }
    fn shutdown(&mut self) {
        Python::attach(|py| {
            let _ = self.interp.bind(py).call_method0("close");
        });
    }
}

// ---------------------------------------------------------------- D: process per worker, Arrow IPC over memfd
const REGION: usize = 8 << 20;
pub struct ProcRunner {
    spec: Spec,
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: ChildStdout,
    mm: MmapMut,
    dead: Option<String>,
}
fn spawn(spec: &Spec) -> Result<(Child, ChildStdin, ChildStdout, MmapMut), String> {
    // SAFETY: plain syscalls; the fd is wrapped in a File exactly once.
    let (file, fd) = unsafe {
        let fd = libc::memfd_create(c"e04-shm".as_ptr(), 0);
        if fd < 0 || libc::ftruncate(fd, (2 * REGION) as i64) != 0 {
            return Err("memfd_create/ftruncate failed".into());
        }
        (std::fs::File::from_raw_fd(fd), fd)
    };
    let mm = unsafe { MmapMut::map_mut(&file) }.map_err(|e| e.to_string())?;
    let mut cmd = Command::new(&spec.python);
    cmd.arg(&spec.worker).arg(fd.to_string()).arg(REGION.to_string()).arg(&spec.kind).arg(&spec.body);
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
    // SAFETY: only async-signal-safe fcntl between fork and exec; keeps the memfd open in the child.
    unsafe {
        cmd.pre_exec(move || {
            libc::fcntl(fd, libc::F_SETFD, 0);
            Ok(())
        });
    }
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let stdin = child.stdin.take().ok_or("no stdin")?;
    let mut stdout = child.stdout.take().ok_or("no stdout")?;
    let mut rdy = [0u8; 1];
    stdout.read_exact(&mut rdy).map_err(|e| format!("worker failed to start: {e}"))?;
    drop(file);
    Ok((child, stdin, stdout, mm))
}
impl ProcRunner {
    fn new(spec: &Spec) -> Result<Self, String> {
        let (child, stdin, stdout, mm) = spawn(spec)?;
        Ok(Self { spec: spec.clone(), child, stdin: Some(stdin), stdout, mm, dead: None })
    }
    fn died(&mut self, why: &str) -> String {
        let st = self.child.wait().map(|s| {
            use std::os::unix::process::ExitStatusExt;
            match s.signal() {
                Some(sig) => format!("killed by signal {sig}"),
                None => format!("exit {:?}", s.code()),
            }
        });
        let m = format!("worker died ({why}): {}", st.unwrap_or_default());
        self.dead = Some(m.clone());
        m
    }
}
impl Runner for ProcRunner {
    fn run(&mut self, col: &ArrayRef) -> Result<ArrayRef, String> {
        if let Some(d) = &self.dead {
            return Err(d.clone());
        }
        let schema = Arc::new(Schema::new(vec![Field::new("x", col.data_type().clone(), false)]));
        let rb = RecordBatch::try_new(schema.clone(), vec![col.clone()]).map_err(|e| e.to_string())?;
        let n = {
            let mut cur = Cursor::new(&mut self.mm[..REGION]);
            let mut w = StreamWriter::try_new(&mut cur, &schema).map_err(|e| e.to_string())?;
            w.write(&rb).map_err(|e| e.to_string())?;
            w.finish().map_err(|e| e.to_string())?;
            drop(w);
            cur.position()
        };
        let io = |s: &mut Self| -> std::io::Result<u64> {
            let si = s.stdin.as_mut().ok_or(std::io::ErrorKind::BrokenPipe)?;
            si.write_all(&n.to_le_bytes())?;
            si.flush()?;
            let mut h = [0u8; 8];
            s.stdout.read_exact(&mut h)?;
            Ok(u64::from_le_bytes(h))
        };
        let len = match io(self) {
            Ok(l) => l,
            Err(e) => return Err(self.died(&e.to_string())),
        };
        if len == u64::MAX {
            return Err("udf error in worker".into());
        }
        let bytes = &self.mm[REGION..REGION + len as usize];
        let mut rd = StreamReader::try_new(Cursor::new(bytes), None).map_err(|e| e.to_string())?;
        let b = rd.next().ok_or("no batch")?.map_err(|e| e.to_string())?;
        Ok(b.column(0).clone())
    }
    fn pids(&self) -> Vec<u32> {
        vec![self.child.id()]
    }
    fn recover(&mut self) -> Result<(), String> {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let (c, i, o, m) = spawn(&self.spec)?;
        self.child = c;
        self.stdin = Some(i);
        self.stdout = o;
        self.mm = m;
        self.dead = None;
        Ok(())
    }
    fn shutdown(&mut self) {
        self.stdin = None;
        let _ = self.child.wait();
    }
}

// ---------------------------------------------------------------- selftest
pub fn selftest(spec: &Spec, pool: &[Batch]) {
    let b = &pool[0];
    if spec.option == "A" || spec.option == "B" {
        Python::attach(|py| {
            let gil = py.import("sys").and_then(|s| s.call_method0("_is_gil_enabled")).and_then(|x| x.extract::<bool>()).ok();
            let ver: String = Python::version_str().to_string();
            let ints: ArrayRef = Arc::new(Int64Array::from_iter_values(0..8192));
            let w = Bound::new(py, ArrowExport::new(&ints)).unwrap();
            let pa = py.import("pyarrow").unwrap();
            let arr = pa.call_method1("array", (w,)).unwrap();
            let addr: usize = arr.call_method0("buffers").unwrap().get_item(1).unwrap().getattr("address").unwrap().extract().unwrap();
            let rust_addr = ints.as_any().downcast_ref::<Int64Array>().unwrap().values().as_ptr() as usize;
            // result direction: pyarrow array -> Rust, pointer identity
            let ra = py.import("pyarrow.compute").unwrap().call_method1("add", (arr.clone(), 1)).unwrap();
            let rbuf: usize = ra.call_method0("buffers").unwrap().get_item(1).unwrap().getattr("address").unwrap().extract().unwrap();
            let back = import_array(&ra);
            let back_ptr = back.as_ref().ok().and_then(|a| a.as_any().downcast_ref::<Int64Array>().map(|x| x.values().as_ptr() as usize)).unwrap_or(0);
            println!(
                "{}",
                serde_json::json!({"python":ver,"gil_enabled":gil,"rust_to_py_zero_copy":addr==rust_addr,"py_to_rust_zero_copy":rbuf==back_ptr && back_ptr!=0})
            );
        });
    }
    let mut r = match make(spec) {
        Ok(r) => r,
        Err(e) => return println!("{}", serde_json::json!({"unsupported":e})),
    };
    match r.run(&b.col).and_then(|o| check(&o, b).map(|_| o)) {
        Ok(o) => println!("{}", serde_json::json!({"option":spec.option,"kind":spec.kind,"roundtrip_ok":true,"len":o.len()})),
        Err(e) => println!("{}", serde_json::json!({"roundtrip_ok":false,"error":e})),
    }
    r.shutdown();
}
