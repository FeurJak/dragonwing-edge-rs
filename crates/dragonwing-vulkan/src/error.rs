//! Translate `ash::vk::Result` / loader errors into `dragonwing_core::Error`.

use ash::vk;
use dragonwing_core::Error;

/// Convert an `ash` Vulkan result into our backend error type.
pub(crate) fn vk_err(context: &str, r: vk::Result) -> Error {
    Error::Backend(format!("vulkan: {context}: {r:?}"))
}

/// Convert an `ash::LoadingError` (failure to load `libvulkan.so.1`).
pub(crate) fn loading_err(e: ash::LoadingError) -> Error {
    Error::Backend(format!("vulkan: failed to load libvulkan: {e}"))
}
