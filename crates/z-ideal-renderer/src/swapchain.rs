//! Owned Vulkan swapchain resources and acquire/present operations.
use ash::vk;

use crate::{VulkanDevice, VulkanError, VulkanSurface, api_error};

/// Swapchain presentation preferences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwapchainConfig {
    /// Prefer FIFO presentation when true; otherwise prefer mailbox.
    pub vsync: bool,
}

impl Default for SwapchainConfig {
    fn default() -> Self {
        Self { vsync: true }
    }
}

/// Result of acquiring a swapchain image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquireResult {
    /// An image is ready for rendering.
    Acquired {
        /// Index into the swapchain image and image-view arrays.
        image_index: u32,
        /// The presentation engine considers the swapchain suboptimal.
        suboptimal: bool,
    },
    /// The swapchain must be recreated before another image can be acquired.
    OutOfDate,
}

/// Result of submitting a swapchain image for presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentResult {
    /// Presentation was queued.
    Presented {
        /// The presentation engine considers the swapchain suboptimal.
        suboptimal: bool,
    },
    /// The swapchain must be recreated before another image can be presented.
    OutOfDate,
}

struct SwapchainResources<'a> {
    device: &'a ash::Device,
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    extent: vk::Extent2D,
    present_mode: vk::PresentModeKHR,
}

impl Drop for SwapchainResources<'_> {
    fn drop(&mut self) {
        // SAFETY: This owner destroys every view before the swapchain that owns its images.
        unsafe {
            for view in self.image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            self.device.destroy_swapchain_resources_placeholder();
        }
    }
}

/// Owns a swapchain and its image views while borrowing the device and surface.
///
/// The window used to create the surface must remain alive for this value's lifetime.
/// The owner must externally synchronize queue access and provide semaphores/fences.
pub struct VulkanSwapchain<'a> {
    _device: &'a VulkanDevice,
    _surface: &'a VulkanSurface,
    loader: ash::khr::swapchain::Device,
    config: SwapchainConfig,
    resources: Option<SwapchainResources<'a>>,
}

impl<'a> VulkanSwapchain<'a> {
    /// Creates a swapchain for the current drawable window extent.
    pub fn new(
        device: &'a VulkanDevice,
        surface: &'a VulkanSurface,
        window_extent: [u32; 2],
        config: SwapchainConfig,
    ) -> Result<Self, VulkanError> {
        if !std::sync::Arc::ptr_eq(&device._instance, &surface.instance) {
            return Err(VulkanError::Surface(
                "swapchain device and surface must belong to the same Vulkan instance".to_owned(),
            ));
        }
        let loader = ash::khr::swapchain::Device::new(&device._instance.handle, &device.handle);
        let resources = create_resources(device, surface, &loader, window_extent, config, vk::SwapchainKHR::null())?;
        Ok(Self {
            _device: device,
            _surface: surface,
            loader,
            config,
            resources: Some(resources),
        })
    }

    /// Recreates the swapchain after waiting for device work to finish.
    ///
    /// A zero-sized extent is rejected without destroying the current swapchain.
    /// Other recreation failures leave this owner without a swapchain; callers may retry.
    pub fn recreate(&mut self, window_extent: [u32; 2]) -> Result<(), VulkanError> {
        if window_extent[0] == 0 || window_extent[1] == 0 {
            return Err(VulkanError::ZeroExtent);
        }
        self._device.wait_idle()?;
        self.resources.take();
        let resources = create_resources(
            self._device,
            self._surface,
            &self.loader,
            window_extent,
            self.config,
            vk::SwapchainKHR::null(),
        )?;
        self.resources = Some(resources);
        Ok(())
    }

    /// Whether this owner currently has a usable swapchain.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.resources.is_some()
    }

    /// Returns swapchain images.
    #[must_use]
    pub fn images(&self) -> &[vk::Image] {
        self.resources.as_ref().map_or(&[], |resources| &resources.images)
    }

    /// Returns image views corresponding by index to images.
    #[must_use]
    pub fn image_views(&self) -> &[vk::ImageView] {
        self.resources.as_ref().map_or(&[], |resources| &resources.image_views)
    }

    /// Returns the selected swapchain image format.
    #[must_use]
    pub fn format(&self) -> Option<vk::Format> {
        self.resources.as_ref().map(|resources| resources.format)
    }

    /// Returns the selected surface color space.
    #[must_use]
    pub fn color_space(&self) -> Option<vk::ColorSpaceKHR> {
        self.resources.as_ref().map(|resources| resources.color_space)
    }

    /// Returns the selected swapchain extent.
    #[must_use]
    pub fn extent(&self) -> Option<vk::Extent2D> {
        self.resources.as_ref().map(|resources| resources.extent)
    }

    /// Returns the selected presentation mode.
    #[must_use]
    pub fn present_mode(&self) -> Option<vk::PresentModeKHR> {
        self.resources.as_ref().map(|resources| resources.present_mode)
    }

    /// Acquires the next image, signaling the caller-provided semaphore or fence.
    pub fn acquire_next_image(
        &self,
        timeout: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> Result<AcquireResult, VulkanError> {
        let resources = self.resources.as_ref().ok_or(VulkanError::MissingSwapchain)?;
        // SAFETY: The swapchain is live and synchronization handles are owned by the caller.
        match unsafe {
            self.loader.acquire_next_image(resources.handle, timeout, semaphore, fence)
        } {
            Ok((image_index, suboptimal)) => Ok(AcquireResult::Acquired {
                image_index,
                suboptimal,
            }),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(AcquireResult::OutOfDate),
            Err(error) => Err(api_error("acquire Vulkan swapchain image", error)),
        }
    }

    /// Queues presentation of an acquired image on the selected presentation queue.
    pub fn present(
        &self,
        image_index: u32,
        wait_semaphores: &[vk::Semaphore],
    ) -> Result<PresentResult, VulkanError> {
        let resources = self.resources.as_ref().ok_or(VulkanError::MissingSwapchain)?;
        if image_index as usize >= resources.images.len() {
            return Err(VulkanError::InvalidSwapchainImageIndex {
                index: image_index,
                image_count: resources.images.len(),
            });
        }
        let swapchains = [resources.handle];
        let image_indices = [image_index];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        // SAFETY: The swapchain/image index are validated; synchronization is caller-owned.
        match unsafe { self.loader.queue_present(self._device.present_queue, &info) } {
            Ok(suboptimal) => Ok(PresentResult::Presented { suboptimal }),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(PresentResult::OutOfDate),
            Err(error) => Err(api_error("present Vulkan swapchain image", error)),
        }
    }
}

fn create_resources<'a>(
    device: &'a VulkanDevice,
    surface: &VulkanSurface,
    loader: &ash::khr::swapchain::Device,
    window_extent: [u32; 2],
    config: SwapchainConfig,
    old_swapchain: vk::SwapchainKHR,
) -> Result<SwapchainResources<'a>, VulkanError> {
    if window_extent[0] == 0 || window_extent[1] == 0 {
        return Err(VulkanError::ZeroExtent);
    }

    let capabilities = surface.capabilities(device.physical_device)?;
    if !capabilities
        .supported_usage_flags
        .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
    {
        return Err(VulkanError::UnsupportedSurfaceUsage);
    }

    let formats = surface.formats(device.physical_device)?;
    let surface_format = choose_surface_format(&formats).ok_or(VulkanError::NoSurfaceFormats)?;
    let present_modes = surface.present_modes(device.physical_device)?;
    let present_mode = choose_present_mode(&present_modes, config.vsync)
        .ok_or(VulkanError::NoPresentModes)?;

    let extent = if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: window_extent[0].clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: window_extent[1].clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    };
    if extent.width == 0 || extent.height == 0 {
        return Err(VulkanError::ZeroExtent);
    }

    let requested_image_count = capabilities.min_image_count.saturating_add(1);
    let image_count = if capabilities.max_image_count > 0 {
        requested_image_count.min(capabilities.max_image_count)
    } else {
        requested_image_count
    };
    let mut queue_family_indices = vec![device.graphics_queue_family];
    if device.present_queue_family != device.graphics_queue_family {
        queue_family_indices.push(device.present_queue_family);
    }
    let sharing_mode = if queue_family_indices.len() > 1 {
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };
    let composite_alpha = choose_composite_alpha(capabilities.supported_composite_alpha)
        .ok_or(VulkanError::NoCompositeAlphaMode)?;
    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface.handle)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(composite_alpha)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(old_swapchain);

    // SAFETY: Surface/device are compatible and all create-info backing arrays remain alive.
    let handle = unsafe {
        loader
            .create_swapchain(&create_info, None)
            .map_err(|error| api_error("create Vulkan swapchain", error))?
    };
    // SAFETY: The swapchain is newly created and owned by this function.
    let images = match unsafe { loader.get_swapchain_images(handle) } {
        Ok(images) => images,
        Err(error) => {
            // SAFETY: No image views were created and the swapchain is uniquely owned here.
            unsafe { loader.destroy_swapchain(handle, None) };
            return Err(api_error("get Vulkan swapchain images", error));
        }
    };

    let mut image_views = Vec::with_capacity(images.len());
    for image in &images {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(*image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(surface_format.format)
            .components(vk::ComponentMapping::default())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        // SAFETY: The image is owned by the live swapchain and the view range is one color mip/layer.
        match unsafe { device.handle.create_image_view(&view_info, None) } {
            Ok(view) => image_views.push(view),
            Err(error) => {
                // SAFETY: All views created so far and the swapchain are uniquely owned here.
                unsafe {
                    for view in image_views.drain(..) {
                        device.handle.destroy_image_view(view, None);
                    }
                    loader.destroy_swapchain(handle, None);
                }
                return Err(api_error("create Vulkan swapchain image view", error));
            }
        }
    }

    Ok(SwapchainResources {
        device: &device.handle,
        handle,
        images,
        image_views,
        format: surface_format.format,
        color_space: surface_format.color_space,
        extent,
        present_mode,
    })
}

fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
    let preferred = [
        vk::Format::B8G8R8A8_SRGB,
        vk::Format::R8G8B8A8_SRGB,
    ];
    formats
        .iter()
        .copied()
        .find(|format| {
            preferred.contains(&format.format)
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| {
            formats.first().copied().map(|mut format| {
                if format.format == vk::Format::UNDEFINED {
                    format.format = vk::Format::B8G8R8A8_SRGB;
                }
                format
            })
        })
}

fn choose_present_mode(
    modes: &[vk::PresentModeKHR],
    vsync: bool,
) -> Option<vk::PresentModeKHR> {
    if vsync {
        return modes.contains(&vk::PresentModeKHR::FIFO).then_some(vk::PresentModeKHR::FIFO);
    }
    [
        vk::PresentModeKHR::MAILBOX,
        vk::PresentModeKHR::IMMEDIATE,
        vk::PresentModeKHR::FIFO,
    ]
    .into_iter()
    .find(|mode| modes.contains(mode))
}

fn choose_composite_alpha(
    supported: vk::CompositeAlphaFlagsKHR,
) -> Option<vk::CompositeAlphaFlagsKHR> {
    [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::INHERIT,
    ]
    .into_iter()
    .find(|mode| supported.contains(*mode))
}

#[cfg(test)]
mod tests {
    use super::{choose_composite_alpha, choose_present_mode, choose_surface_format};
    use ash::vk;

    #[test]
    fn surface_format_prefers_srgb_bgra() {
        let formats = [
            vk::SurfaceFormatKHR {
                format: vk::Format::R8G8B8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_SRGB,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        ];
        assert_eq!(
            choose_surface_format(&formats).map(|format| format.format),
            Some(vk::Format::B8G8R8A8_SRGB)
        );
    }

    #[test]
    fn vsync_uses_fifo_and_non_vsync_prefers_mailbox() {
        let modes = [
            vk::PresentModeKHR::FIFO,
            vk::PresentModeKHR::MAILBOX,
            vk::PresentModeKHR::IMMEDIATE,
        ];
        assert_eq!(
            choose_present_mode(&modes, true),
            Some(vk::PresentModeKHR::FIFO)
        );
        assert_eq!(
            choose_present_mode(&modes, false),
            Some(vk::PresentModeKHR::MAILBOX)
        );
    }

    #[test]
    fn composite_alpha_selects_first_supported_mode() {
        let supported = vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
            | vk::CompositeAlphaFlagsKHR::INHERIT;
        assert_eq!(
            choose_composite_alpha(supported),
            Some(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
        );
    }
}
