//! build.rs — компиляция CUDA-ядер через nvcc
//!
//! Собирает .cu файлы из csrc/ и линкует их с Rust-бинарником.
//! Это позволяет использовать один компилятор (cargo build) для всего.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let cuda_files = vec![
        "csrc/cuda_view.cu",
    ];

    let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let has_cuda = src_dir.join("csrc/cuda_view.cu").exists();

    if has_cuda {
        let nvcc = which_nvcc();
        if let Some(nvcc_path) = nvcc {
            println!("cargo:rerun-if-changed=csrc/");
            println!("cargo:rerun-if-changed=build.rs");

            let mut nvcc_cmd = Command::new(nvcc_path);
            nvcc_cmd
                .arg("--compiler-options")
                .arg("-fPIC")
                .arg("--shared")
                .arg("-o")
                .arg(out_dir.join("librllm_cuda.so"));

            for f in &cuda_files {
                let path = src_dir.join(f);
                if path.exists() {
                    nvcc_cmd.arg(path);
                }
            }

            if cfg!(target_os = "windows") {
                nvcc_cmd
                    .arg("--compiler-options")
                    .arg("/MD")
                    .arg("-o")
                    .arg(out_dir.join("rllm_cuda.dll"));

                for f in &cuda_files {
                    let path = src_dir.join(f);
                    if path.exists() {
                        nvcc_cmd.arg(path);
                    }
                }
            }

            let _ = nvcc_cmd.status();
            println!("cargo:rustc-link-search=native={}", out_dir.display());
        }
    }
}

fn which_nvcc() -> Option<String> {
    if let Ok(path) = env::var("CUDA_HOME") {
        let nvcc = PathBuf::from(path).join("bin").join("nvcc");
        if nvcc.exists() {
            return Some(nvcc.to_string_lossy().to_string());
        }
    }
    Command::new("which")
        .arg("nvcc")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
}
