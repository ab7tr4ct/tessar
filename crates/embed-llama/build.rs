//! Builds llama.cpp (the `third_party/llama.cpp` submodule) as static libraries with
//! CMake, then the C++ shim and the cxx bridge, and links them together.
//!
//! On macOS the Metal backend is on and its shader source is embedded in
//! the binary (`GGML_METAL_EMBED_LIBRARY`), so nothing has to sit next to
//! the executable. Elsewhere only the CPU backend is built.

use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source = manifest.join("../../third_party/llama.cpp");
    if !source.join("CMakeLists.txt").is_file() {
        panic!(
            "{} is empty: run `git submodule update --init` first",
            source.display()
        );
    }
    let macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");

    let mut cmake = cmake::Config::new(&source);
    cmake
        // ggml without optimizations is far too slow even for tests.
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        // Only the library: no common utils, tools, server, tests or examples.
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_APP", "OFF")
        .define("LLAMA_OPENSSL", "OFF")
        // ggml's own thread pool; OpenMP would add a runtime to link.
        .define("GGML_OPENMP", "OFF")
        .define("GGML_METAL", if macos { "ON" } else { "OFF" })
        .define("GGML_METAL_EMBED_LIBRARY", if macos { "ON" } else { "OFF" });
    let out = cmake.build();

    let lib = [out.join("lib"), out.join("lib64")]
        .into_iter()
        .find(|dir| dir.join(static_lib("llama")).is_file())
        .expect("llama.cpp installed no libllama");
    println!("cargo:rustc-link-search=native={}", lib.display());
    for name in link_order(&lib) {
        println!("cargo:rustc-link-lib=static={name}");
    }
    if macos {
        for framework in ["Foundation", "Metal", "MetalKit", "Accelerate"] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
    }
    // The C++ standard library comes with `cxx` (through `link-cplusplus`).

    cxx_build::bridge("src/lib.rs")
        .file("src/shim.cc")
        .include(out.join("include"))
        .std("c++17")
        .flag_if_supported("-Wall")
        .compile("tessar-embed-llama-shim");

    for path in ["build.rs", "src/lib.rs", "src/shim.h", "src/shim.cc"] {
        println!("cargo:rerun-if-changed={path}");
    }
    for path in [
        "CMakeLists.txt",
        "include",
        "src",
        "ggml/CMakeLists.txt",
        "ggml/include",
        "ggml/src",
    ] {
        println!("cargo:rerun-if-changed=../../third_party/llama.cpp/{path}");
    }
}

/// The static libraries llama.cpp installed, in an order a one-pass
/// linker accepts: `llama`, then `ggml` and the backends, then
/// `ggml-base`, which they all use.
fn link_order(dir: &Path) -> Vec<String> {
    let mut backends: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().into_string().ok()?;
            let stem = name.strip_prefix("lib")?.strip_suffix(".a")?.to_owned();
            (stem.starts_with("ggml-") && stem != "ggml-base").then_some(stem)
        })
        .collect();
    backends.sort();
    let mut order = vec!["llama".to_owned(), "ggml".to_owned()];
    order.extend(backends);
    order.push("ggml-base".to_owned());
    for name in &order {
        assert!(
            dir.join(static_lib(name)).is_file(),
            "llama.cpp installed no {}",
            static_lib(name)
        );
    }
    order
}

fn static_lib(name: &str) -> String {
    format!("lib{name}.a")
}
