//! Hexagon DSP probe.
//!
//! On Qualcomm SoCs the DSP is reachable from userspace through the
//! "FastRPC" character devices: `/dev/adsprpc-smd` (audio DSP) and
//! `/dev/cdsprpc-smd` (compute DSP). On a mainline-kernel image these nodes
//! are typically absent, so this probe simply reports which (if any) exist.
//! Loading the Hexagon SDK on top is deferred until a later task.

use dragonwing_core::capabilities::DspInfo;
use std::path::Path;

const CANDIDATES: &[&str] = &[
    "/dev/adsprpc-smd",
    "/dev/cdsprpc-smd",
    "/dev/fastrpc-adsp",
    "/dev/fastrpc-cdsp",
];

#[must_use]
pub fn probe() -> DspInfo {
    let fastrpc_nodes: Vec<String> = CANDIDATES
        .iter()
        .filter(|p| Path::new(p).exists())
        .map(|s| (*s).to_string())
        .collect();
    DspInfo {
        present: !fastrpc_nodes.is_empty(),
        fastrpc_nodes,
    }
}
