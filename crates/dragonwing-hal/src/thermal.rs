//! Thermal probe — enumerates /sys/class/thermal/thermal_zone*.

use dragonwing_core::capabilities::ThermalZone;
use std::fs;

#[must_use]
pub fn probe() -> Vec<ThermalZone> {
    let Ok(dir) = fs::read_dir("/sys/class/thermal") else {
        return Vec::new();
    };
    let mut zones = Vec::new();
    for entry in dir.flatten() {
        let name = entry.file_name();
        let s = name.to_string_lossy();
        let Some(rest) = s.strip_prefix("thermal_zone") else {
            continue;
        };
        let Ok(index) = rest.parse::<u32>() else {
            continue;
        };
        let base = entry.path();
        let type_name = fs::read_to_string(base.join("type"))
            .unwrap_or_default()
            .trim()
            .to_string();
        let temp_mc = fs::read_to_string(base.join("temp"))
            .ok()
            .and_then(|s| s.trim().parse::<i64>().ok())
            .unwrap_or(0);
        zones.push(ThermalZone {
            index,
            type_name,
            temp_mc,
        });
    }
    zones.sort_by_key(|z| z.index);
    zones
}
