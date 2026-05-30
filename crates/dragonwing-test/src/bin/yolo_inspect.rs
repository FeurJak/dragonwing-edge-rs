//! Inspect a YOLOv8 ONNX model: print input/output shapes, op counts, and
//! the result of compiling it through dragonwing-onnx.
//!
//! Usage:
//!     yolo_inspect <path/to/model.onnx>
//!
//! No Vulkan device required; pure host-side analysis.

use std::collections::HashMap;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: yolo_inspect <path/to/model.onnx>");
        std::process::exit(2);
    });
    println!("Loading {path}...");
    let model = match dragonwing_onnx::load_model(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR loading: {e}");
            std::process::exit(1);
        }
    };

    println!("\n=== Inputs ===");
    for inp in &model.inputs {
        println!("  {} shape={:?}", inp.name, inp.shape);
    }
    println!("\n=== Outputs ===");
    for out in &model.outputs {
        println!("  {} shape={:?}", out.name, out.shape);
    }
    println!("\n=== Node count: {} ===", model.nodes.len());
    let mut by_type: HashMap<&str, usize> = HashMap::new();
    for n in &model.nodes {
        *by_type.entry(n.op_type.as_str()).or_insert(0) += 1;
    }
    let mut v: Vec<_> = by_type.into_iter().collect();
    v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (k, n) in &v {
        println!("  {k}: {n}");
    }

    // Try to compile to F32 NHWC.
    println!("\n=== Validating with dragonwing-onnx (Dtype::F32) ===");
    let report = dragonwing_onnx::validate_model(&model, dragonwing_core::Dtype::F32);
    println!(
        "  supported: {} / {} nodes",
        report.supported.len(),
        report.supported.len() + report.unsupported.len()
    );
    if !report.unsupported.is_empty() {
        println!("  unsupported ops (first 10):");
        for (name, reason) in report.unsupported.iter().take(10) {
            println!("    {name}: {reason}");
        }
    }
}
