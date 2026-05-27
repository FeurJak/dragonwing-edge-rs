//! Debug binary to investigate device selection.

use dragonwing_vulkan::{VulkanBackend, VulkanConfig};

fn main() {
    println!("=== Vulkan Device Selection Debug ===\n");

    // Try with default config
    match VulkanBackend::new(VulkanConfig::default()) {
        Ok(backend) => {
            println!("Selected device: {}", backend.device_name());
            println!("Driver info: {}", backend.driver_info());
        }
        Err(e) => {
            println!("Failed: {e}");
        }
    }

    // Try forcing device index 0 (should be Turnip based on vulkaninfo)
    println!("\n--- Forcing device_index=0 ---");
    let config = VulkanConfig {
        device_index: Some(0),
        ..Default::default()
    };
    match VulkanBackend::new(config) {
        Ok(backend) => {
            println!("Device 0: {}", backend.device_name());
            println!("Driver: {}", backend.driver_info());
        }
        Err(e) => {
            println!("Device 0 failed: {e}");
        }
    }

    // Try device index 1
    println!("\n--- Forcing device_index=1 ---");
    let config = VulkanConfig {
        device_index: Some(1),
        ..Default::default()
    };
    match VulkanBackend::new(config) {
        Ok(backend) => {
            println!("Device 1: {}", backend.device_name());
            println!("Driver: {}", backend.driver_info());
        }
        Err(e) => {
            println!("Device 1 failed: {e}");
        }
    }
}
