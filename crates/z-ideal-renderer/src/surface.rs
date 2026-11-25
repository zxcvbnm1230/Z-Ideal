//! Vulkan window-surface ownership and presentation capability queries.
use std::{ffi::{CStr, CString}, sync::Arc};

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::{VulkanError, VulkanInstance};

pub(crate) fn required_instance_extensions(
    window: &Window,
) -> Result<Vec<CString>, VulkanError> {
    let display_handle = window
        .display_handle()
        .map_err(|error| VulkanError::Surface(error.to_string()))?
        .as_raw();
    let required = ash_window::enumerate_required_extensions(display_handle).map_err(|error| {
        VulkanError::Api {
            operation: "enumerate required Vulkan window extensions",
            result: error,
        }
    })?;
    required
        .iter()
        .map(|pointer| {
            // SAFETY: ash-window returns pointers to static, NUL-terminated extension names.
            Ok(unsafe { CStr::from_ptr(*pointer) }.to_owned())
        })
        .collect()
}

/// Owns a Vulkan presentation surface tied to the instance that created it.
///
/// The native window must outlive the surface and all swapchains created from it.
pub struct VulkanSurface {
    pub(crate) instance: Arc<VulkanInstance>,
    pub(crate) loader: ash::khr::surface::Instance,
    pub(crate) handle: vk::SurfaceKHR,
}

impl VulkanSurface {
    /// Creates a Vulkan surface from a live winit window.
    pub fn new(
        instance: Arc<VulkanInstance>,
        window: &Window,
    ) -> Result<Self, VulkanError> {
        let display_handle = window
            .display_handle()
            .map_err(|error| VulkanError::Surface(error.to_string()))?
            .as_raw();
        let window_handle = window
            .window_handle()
            .map_err(|error| VulkanError::Surface(error.to_string()))?
            .as_raw();

        // SAFETY: The window remains alive for this call; raw handles are obtained
        // directly from that window, and the instance/entry outlive the surface.
        let handle = unsafe {
            ash_window::create_surface(
                &instance._entry,
                &instance.handle,
                display_handle,
                window_handle,
                None,
            )
            .map_err(|error| VulkanError::Api {
                operation: "create Vulkan window surface",
                result: error,
            })?
        };
        let loader = ash::khr::surface::Instance::new(&instance._entry, &instance.handle);

        Ok(Self {
            instance,
            loader,
            handle,
        })
    }

    /// Returns the raw surface handle for renderer-internal integration.
    #[must_use]
    pub const fn handle(&self) -> vk::SurfaceKHR {
        self.handle
    }

    /// Returns whether a physical-device queue family can present to this surface.
    pub fn supports_present(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
    ) -> Result<bool, VulkanError> {
        // SAFETY: The surface and physical device belong to the retained live instance.
        unsafe {
            self.loader
                .get_physical_device_surface_support(physical_device, queue_family, self.handle)
                .map_err(|error| VulkanError::Api {
                    operation: "query Vulkan surface presentation support",
                    result: error,
                })
        }
    }

    /// Queries the surface capabilities supported by a physical device.
    pub fn capabilities(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<vk::SurfaceCapabilitiesKHR, VulkanError> {
        // SAFETY: The surface and physical device belong to the retained live instance.
        unsafe {
            self.loader
                .get_physical_device_surface_capabilities(physical_device, self.handle)
                .map_err(|error| VulkanError::Api {
                    operation: "query Vulkan surface capabilities",
                    result: error,
                })
        }
    }

    /// Queries supported surface image formats.
    pub fn formats(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::SurfaceFormatKHR>, VulkanError> {
        // SAFETY: The surface and physical device belong to the retained live instance.
        unsafe {
            self.loader
                .get_physical_device_surface_formats(physical_device, self.handle)
                .map_err(|error| VulkanError::Api {
                    operation: "query Vulkan surface formats",
                    result: error,
                })
        }
    }

    /// Queries supported presentation modes.
    pub fn present_modes(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::PresentModeKHR>, VulkanError> {
        // SAFETY: The surface and physical device belong to the retained live instance.
        unsafe {
            self.loader
                .get_physical_device_surface_present_modes(physical_device, self.handle)
                .map_err(|error| VulkanError::Api {
                    operation: "query Vulkan surface present modes",
                    result: error,
                })
        }
    }
}

impl Drop for VulkanSurface {
    fn drop(&mut self) {
        // SAFETY: This wrapper uniquely owns the surface; its instance is retained.
        unsafe {
            self.loader.destroy_surface(self.handle, None);
        }
    }
}
