//! Kernel / OS / libc identification.

use dragonwing_core::capabilities::KernelInfo;
use std::fs;
use std::process::Command;

#[must_use]
pub fn probe() -> KernelInfo {
    KernelInfo {
        uname: run("uname", &["-a"]).unwrap_or_default(),
        arch: run("uname", &["-m"]).unwrap_or_default(),
        os_pretty_name: read_os_pretty_name(),
        libc: detect_libc(),
    }
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn read_os_pretty_name() -> String {
    // /etc/os-release is a `KEY=VALUE` file. Values may be quoted.
    let Ok(text) = fs::read_to_string("/etc/os-release") else {
        return String::new();
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("PRETTY_NAME=") {
            return rest.trim_matches('"').to_string();
        }
    }
    String::new()
}

fn detect_libc() -> String {
    // `ldd --version` first line typically contains the libc identifier.
    if let Some(out) = run("ldd", &["--version"])
        && let Some(first) = out.lines().next()
    {
        return first.to_string();
    }
    String::new()
}
