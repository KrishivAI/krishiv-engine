// The repo-level .cargo/config.toml sets PYO3_BUILD_EXTENSION_MODULE=1 (krishiv-python is an extension
// module), which stops pyo3 from linking libpython. This lab *embeds* Python, so link it explicitly.
fn main() {
    let cfg = pyo3_build_config::get();
    if let Some(dir) = cfg.lib_dir() {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
    if let Some(name) = cfg.lib_name() {
        println!("cargo:rustc-link-lib={name}");
    }
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
}
