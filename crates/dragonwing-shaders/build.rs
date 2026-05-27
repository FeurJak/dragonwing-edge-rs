//! Build script: compile every `glsl/*.comp` into SPIR-V.
//!
//! # Two-track build policy
//!
//! There are two ways a contributor can produce the `.spv` blobs that
//! [`lib.rs`] re-exports via `include_bytes!`:
//!
//! 1. **`glslangValidator` on PATH** — invoked by this build script. The
//!    output goes into `OUT_DIR` and `lib.rs` includes from there. This is
//!    the development happy path: edit a `.comp`, `cargo build`, the
//!    shader is recompiled.
//!
//! 2. **No `glslangValidator` on PATH** — this build script copies the
//!    checked-in `spv/*.spv` blobs into `OUT_DIR` so `include_bytes!` still
//!    works. Use this on contributors' machines that don't want to install
//!    the LunarG SDK, and on CI runners that should reproduce releases.
//!
//! The checked-in blobs in `spv/` are the source of truth for releases.
//! When a `.comp` is changed, the developer must:
//!
//! ```text
//! cargo build -p dragonwing-shaders
//! cp $(find target -name 'OPNAME.spv' -path '*dragonwing-shaders*') \
//!    crates/dragonwing-shaders/spv/OPNAME.spv
//! git add crates/dragonwing-shaders/spv/OPNAME.spv
//! ```
//!
//! and commit. A CI check (later task) can diff freshly-compiled output
//! against `spv/` to prevent drift.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let glsl_dir = manifest_dir.join("glsl");
    let spv_fallback_dir = manifest_dir.join("spv");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=glsl");
    println!("cargo:rerun-if-changed=spv");
    println!("cargo:rerun-if-env-changed=DRAGONWING_FORCE_PREBUILT_SPV");

    let force_prebuilt = env::var("DRAGONWING_FORCE_PREBUILT_SPV").is_ok();
    let glslang = if force_prebuilt {
        None
    } else {
        which("glslangValidator")
    };

    // Enumerate every .comp source.
    let sources: Vec<PathBuf> = match fs::read_dir(&glsl_dir) {
        Ok(it) => it
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "comp"))
            .collect(),
        Err(_) => {
            // No glsl/ directory means the crate hasn't been set up yet;
            // emit a clear error rather than producing an empty crate.
            panic!(
                "dragonwing-shaders: missing glsl/ directory at {}",
                glsl_dir.display()
            );
        }
    };

    if sources.is_empty() {
        panic!(
            "dragonwing-shaders: no .comp shaders found in {}",
            glsl_dir.display()
        );
    }

    for src in &sources {
        let stem = src.file_stem().unwrap().to_string_lossy().to_string();
        let dst = out_dir.join(format!("{stem}.spv"));

        if let Some(ref tool) = glslang {
            compile_with_glslang(tool, src, &dst);
        } else {
            // Fallback: copy checked-in SPIR-V.
            let prebuilt = spv_fallback_dir.join(format!("{stem}.spv"));
            if !prebuilt.exists() {
                panic!(
                    "dragonwing-shaders: glslangValidator not found AND no prebuilt {} \
                     exists. Either install glslang (`brew install glslang`) or commit \
                     a prebuilt SPIR-V at that path.",
                    prebuilt.display()
                );
            }
            fs::copy(&prebuilt, &dst).unwrap_or_else(|e| {
                panic!(
                    "dragonwing-shaders: failed to copy {} -> {}: {e}",
                    prebuilt.display(),
                    dst.display()
                )
            });
        }
    }
}

fn which(cmd: &str) -> Option<String> {
    let path = env::var_os("PATH")?;
    for dir in env::split_paths(&path) {
        let full = dir.join(cmd);
        if full.is_file() {
            return Some(full.to_string_lossy().into_owned());
        }
    }
    None
}

fn compile_with_glslang(tool: &str, src: &Path, dst: &Path) {
    // `glslangValidator -V` emits SPIR-V (Vulkan semantics). `-o` sets the
    // output path. We require Vulkan 1.1 semantics for subgroup ops.
    let status = Command::new(tool)
        .arg("-V")
        .arg("--target-env")
        .arg("vulkan1.1")
        .arg("-o")
        .arg(dst)
        .arg(src)
        .status()
        .unwrap_or_else(|e| {
            panic!(
                "dragonwing-shaders: failed to spawn glslangValidator ({tool}): {e}"
            )
        });
    if !status.success() {
        panic!(
            "dragonwing-shaders: glslangValidator failed on {}",
            src.display()
        );
    }
}
