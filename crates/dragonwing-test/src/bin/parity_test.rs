//! Cross-backend parity test binary.
//!
//! Runs the same ops on CPU (NEON) and Vulkan (Turnip) backends and
//! verifies they produce numerically equivalent results.

use dragonwing_core::Backend;
use dragonwing_cpu::CpuBackend;
use dragonwing_vulkan::{VulkanBackend, VulkanConfig};
use dragonwing_test::{ParityTest, TestConfig};

fn main() {
    println!("=== dragonwing-edge cross-backend parity test ===\n");

    // Initialize backends
    let cpu = CpuBackend::new();
    println!("CPU backend: {}", cpu.name());

    let vulkan = match VulkanBackend::new(VulkanConfig::default()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ERROR: Failed to initialize Vulkan backend: {e}");
            std::process::exit(1);
        }
    };
    println!("Vulkan backend: {} ({})", vulkan.device_name(), vulkan.driver_info());
    println!();

    // Run parity tests
    let config = TestConfig {
        num_trials: 5,
        verbose: true,
        ..Default::default()
    };

    let results = ParityTest::run_all(&cpu, &vulkan, &config);
    results.print_summary();

    if results.all_passed() {
        println!("\nAll parity tests passed!");
        std::process::exit(0);
    } else {
        println!("\nSome parity tests failed!");
        std::process::exit(1);
    }
}
