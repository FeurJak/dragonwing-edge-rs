//! Memory probe — parses /proc/meminfo.

use dragonwing_core::capabilities::MemoryInfo;
use std::fs;

#[must_use]
pub fn probe() -> MemoryInfo {
    let text = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let mut info = MemoryInfo::default();
    for line in text.lines() {
        // Format: `Key:    12345 kB`
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let value_kib = rest
            .split_whitespace()
            .next()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        match key {
            "MemTotal" => info.total_kib = value_kib,
            "MemAvailable" => info.available_kib = value_kib,
            "MemFree" => info.free_kib = value_kib,
            _ => {}
        }
    }
    info
}
