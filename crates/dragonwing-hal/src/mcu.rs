//! STM32U585 link probe.
//!
//! On the Arduino UNO Q the A53 talks to the STM32U585 microcontroller over
//! a USB CDC-ACM (serial) link. We enumerate likely TTY candidates and the
//! USB devicetree. The actual STM32-side framework lives outside this repo
//! (see `DragonWing-rs`); this probe is informational only.

use dragonwing_core::capabilities::{McuLinkInfo, UsbDevice};
use std::fs;
use std::path::{Path, PathBuf};

#[must_use]
pub fn probe() -> McuLinkInfo {
    McuLinkInfo {
        candidate_tty: enumerate_tty(),
        usb_devices: enumerate_usb(),
    }
}

fn enumerate_tty() -> Vec<String> {
    let Ok(entries) = fs::read_dir("/dev") else {
        return Vec::new();
    };
    let mut ttys = Vec::new();
    for e in entries.flatten() {
        let name = e.file_name();
        let s = name.to_string_lossy();
        // Likely USB-serial bridges, not the legacy 8250 ttyS* or virtual
        // ttyN consoles.
        if s.starts_with("ttyACM")
            || s.starts_with("ttyUSB")
            || s.starts_with("ttyHS")
            || s.starts_with("ttyGS")
        {
            ttys.push(format!("/dev/{s}"));
        }
    }
    ttys.sort();
    ttys
}

fn enumerate_usb() -> Vec<UsbDevice> {
    // /sys/bus/usb/devices/<busaddr>/{idVendor,idProduct,product}
    let Ok(entries) = fs::read_dir("/sys/bus/usb/devices") else {
        return Vec::new();
    };
    let mut devices = Vec::new();
    for e in entries.flatten() {
        let path: PathBuf = e.path();
        // Filter out interface entries (they contain ':').
        let name = e.file_name();
        let s = name.to_string_lossy();
        if s.contains(':') {
            continue;
        }
        let vendor_id = read_hex_u16(&path.join("idVendor"));
        let product_id = read_hex_u16(&path.join("idProduct"));
        if vendor_id == 0 && product_id == 0 {
            // usb root hubs etc. — skip.
            continue;
        }
        let description = read_trim(&path.join("product"));
        devices.push(UsbDevice {
            bus_address: s.into_owned(),
            vendor_id,
            product_id,
            description,
        });
    }
    devices.sort_by(|a, b| a.bus_address.cmp(&b.bus_address));
    devices
}

fn read_hex_u16(p: &Path) -> u16 {
    fs::read_to_string(p)
        .ok()
        .and_then(|s| u16::from_str_radix(s.trim(), 16).ok())
        .unwrap_or(0)
}

fn read_trim(p: &Path) -> String {
    fs::read_to_string(p).map(|s| s.trim().to_string()).unwrap_or_default()
}
