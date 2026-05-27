//! `dragonwing-probe` — run on the target device, emit a JSON capability
//! report to stdout.
//!
//! Usage:
//!   dragonwing-probe                # pretty JSON to stdout
//!   dragonwing-probe -o report.json # also write to file
//!
//! Exit code is 0 if any subsystem was probed (the binary is best-effort and
//! does not fail on a missing GPU/DSP).

use std::env;
use std::fs;
use std::io::{self, Write};
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut out_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                out_path = args.get(i).cloned();
            }
            "-h" | "--help" => {
                print_help();
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                print_help();
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let host = hostname();
    let captured_at = iso8601_now();
    let caps = dragonwing_hal::probe_all(host, captured_at);

    let json = serde_json::to_string_pretty(&caps).expect("serialise capabilities");

    // Always stdout.
    let stdout = io::stdout();
    let mut h = stdout.lock();
    let _ = h.write_all(json.as_bytes());
    let _ = h.write_all(b"\n");

    if let Some(path) = out_path
        && let Err(e) = fs::write(&path, &json)
    {
        eprintln!("warning: failed to write {path}: {e}");
    }
}

fn print_help() {
    println!("dragonwing-probe — emit a HardwareCapabilities JSON report");
    println!();
    println!("USAGE:");
    println!("    dragonwing-probe [-o PATH]");
}

fn hostname() -> String {
    // Read /proc/sys/kernel/hostname — present on every Linux. Falls back to
    // empty string on non-Linux (the probe is Linux-only by design).
    fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn iso8601_now() -> String {
    // Minimal ISO-8601 UTC "YYYY-MM-DDTHH:MM:SSZ" without pulling in chrono.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_iso8601_utc(secs)
}

/// Convert a Unix timestamp (seconds since 1970-01-01) into an ISO-8601 UTC
/// string. Implemented inline to avoid a chrono/time dependency for a single
/// formatting call.
fn format_iso8601_utc(unix_secs: u64) -> String {
    let secs_per_day: u64 = 86_400;
    let mut days = unix_secs / secs_per_day;
    let time_of_day = unix_secs % secs_per_day;
    let hour = time_of_day / 3600;
    let minute = (time_of_day % 3600) / 60;
    let second = time_of_day % 60;

    // Convert `days` (since 1970-01-01) into Y/M/D using the civil_from_days
    // algorithm by Howard Hinnant (public domain).
    days += 719_468; // shift epoch to 0000-03-01
    let era = days / 146_097;
    let doe = days - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };

    format!("{year:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z")
}
