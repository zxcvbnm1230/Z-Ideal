//! Explicitly owned Vulkan images, allocations, and image views.
use ash::vk;

use crate::memory::MemoryAllocation;
use crate::{VulkanDevice, VulkanError, api_error};

/// Options for creating an owned Vulkan image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageConfig {
    /// Image dimensionality.
    pub image_type: vk::ImageType,
    /// Width, height, and depth in texels.
    pub extent: vk::Extent3D,
    /// Image texel format.
    pub format: vk::Format,
    /// Number of mip levels; must fit within the largest extent dimension.
    pub mip_levels: u32,
    /// Number of array layers.
    pub array_layers: u32,
    /// Number of samples per texel.
    pub samples: vk::SampleCountFlags,
    /// Linear or optimal image memory layout.
    pub tiling: vk::ImageTiling,
    /// Required image usage flags.
    pub usage: vk::ImageUsageFlags,
    /// Image creation flags, including cube compatibility when required.
    pub flags: vk::ImageCreateFlags,
}

/// Options for creating a view into an owned Vulkan image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageViewConfig {
    /// View dimensionality.
    pub view_type: vk::ImageViewType,
    /// Image aspects exposed by the view.
    pub aspect: vk::ImageAspectFlags,
    /// First mip level visible through the view.
    pub base_mip_level: u32,
    /// Number of mip levels visible through the view.
    pub level_count: u32,
    /// First array layer visible through the view.
    pub base_array_layer: u32,
    /// Number of array layers visible through the view.
    pub layer_count: u32,
}

impl Default for ImageViewConfig {
    fn default() -> Self {
        Self {
            view_type: vk::ImageViewType::TYPE_2D,
            aspect: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
}

/// Owns a Vulkan image and its suballocation from the device memory pool.
///
/// Image memory prefers the device-local property. The selected memory type is
/// retained for inspection because Vulkan implementations may expose different
/// compatible memory configurations. Layout transitions, access barriers,
/// queue ownership and image-layout transitions remain explicit caller responsibilities
/// through the validated barrier and release/acquire APIs.
pub struct VulkanImage<'device> {
    pub(super) device: &'device VulkanDevice,
    handle: vk::Image,
    _allocation: MemoryAllocation,
    config: ImageConfig,
    memory_flags: vk::MemoryPropertyFlags,
}

impl<'device> VulkanImage<'device> {
    /// Creates an image and a compatible suballocation from the device memory pool.
    pub fn new(device: &'device VulkanDevice, config: ImageConfig) -> Result<Self, VulkanError> {
        validate_image_config(config)?;
        let info = vk::ImageCreateInfo::default()
            .flags(config.flags)
            .image_type(config.image_type)
            .format(config.format)
            .extent(config.extent)
            .mip_levels(config.mip_levels)
            .array_layers(config.array_layers)
            .samples(config.samples)
            .tiling(config.tiling)
            .usage(config.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // SAFETY: The device is live and the create info contains no borrowed pointers.
        let handle = unsafe {
            device
                .handle
                .create_image(&info, None)
                .map_err(|error| api_error("create Vulkan image", error))?
        };
        // SAFETY: The image belongs to this live device.
        let requirements = unsafe { device.handle.get_image_memory_requirements(handle) };
        let properties = device.memory_allocator.memory_properties();
        let Some(memory_type_index) =
            select_image_memory_type(properties, requirements.memory_type_bits)
        else {
            // SAFETY: The image has not been bound to memory and is uniquely owned here.
            unsafe { device.handle.destroy_image(handle, None) };
            return Err(VulkanError::NoCompatibleImageMemoryType);
        };
        let allocation = match device
            .memory_allocator
            .allocate(requirements, memory_type_index)
        {
            Ok(allocation) => allocation,
            Err(error) => {
                // SAFETY: The image has no bound memory when allocation fails.
                unsafe { device.handle.destroy_image(handle, None) };
                return Err(error);
            }
        };
        // SAFETY: The selected suballocation satisfies the image's memory requirements.
        if let Err(error) = unsafe {
            device
                .handle
                .bind_image_memory(handle, allocation.memory(), allocation.offset())
        } {
            // SAFETY: Binding failed; neither object is in use by submitted work.
            unsafe { device.handle.destroy_image(handle, None) };
            drop(allocation);
            return Err(api_error("bind Vulkan image memory", error));
        }

        let memory_flags = allocation.property_flags();
        Ok(Self {
            device,
            handle,
            _allocation: allocation,
            config,
            memory_flags,
        })
    }

    /// Returns the raw image handle for command recording and descriptor updates.
    #[must_use]
    pub const fn handle(&self) -> vk::Image {
        self.handle
    }

    /// Returns the image creation configuration.
    #[must_use]
    pub const fn config(&self) -> ImageConfig {
        self.config
    }

    /// Returns the actual memory properties selected for this image.
    #[must_use]
    pub const fn memory_properties(&self) -> vk::MemoryPropertyFlags {
        self.memory_flags
    }

    /// Creates a view whose lifetime cannot exceed this image's lifetime.
    pub fn create_view(
        &self,
        config: ImageViewConfig,
    ) -> Result<VulkanImageView<'_, 'device>, VulkanError> {
        validate_image_view_config(self.config, config)?;
        let components = vk::ComponentMapping::default();
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(config.aspect)
            .base_mip_level(config.base_mip_level)
            .level_count(config.level_count)
            .base_array_layer(config.base_array_layer)
            .layer_count(config.layer_count);
        let info = vk::ImageViewCreateInfo::default()
            .image(self.handle)
            .view_type(config.view_type)
            .format(self.config.format)
            .components(components)
            .subresource_range(subresource_range);

        // SAFETY: The image remains live for the returned view's lifetime and the
        // subresource range was validated against the image configuration.
        let handle = unsafe {
            self.device
                .handle
                .create_image_view(&info, None)
                .map_err(|error| api_error("create Vulkan image view", error))?
        };
        Ok(VulkanImageView {
            image: self,
            handle,
            config,
        })
    }
}

impl Drop for VulkanImage<'_> {
    fn drop(&mut self) {
        // SAFETY: Engine-recorded command buffers and borrowed image views retain
        // this image until submission fences complete. Raw Vulkan users must provide
        // equivalent external synchronization before dropping the image.
        unsafe { self.device.handle.destroy_image(self.handle, None) };
    }
}

/// Owns a Vulkan image view and borrows the image it views.
pub struct VulkanImageView<'image, 'device> {
    image: &'image VulkanImage<'device>,
    handle: vk::ImageView,
    config: ImageViewConfig,
}

impl<'image, 'device> VulkanImageView<'image, 'device> {
    /// Returns the raw image-view handle for descriptor and framebuffer creation.
    #[must_use]
    pub const fn handle(&self) -> vk::ImageView {
        self.handle
    }

    /// Returns the image-view configuration.
    #[must_use]
    pub const fn config(&self) -> ImageViewConfig {
        self.config
    }

    /// Returns the image referenced by this view.
    #[must_use]
    pub const fn image(&self) -> &'image VulkanImage<'device> {
        self.image
    }
}

impl Drop for VulkanImageView<'_, '_> {
    fn drop(&mut self) {
        // SAFETY: Framebuffers and descriptor leases borrow this view through command
        // submission completion. The parent image remains borrowed for this view's life.
        unsafe {
            self.image
                .device
                .handle
                .destroy_image_view(self.handle, None);
        }
    }
}

fn validate_image_config(config: ImageConfig) -> Result<(), VulkanError> {
    if config.extent.width == 0 || config.extent.height == 0 || config.extent.depth == 0 {
        return Err(VulkanError::InvalidImageExtent);
    }
    if config.format == vk::Format::UNDEFINED {
        return Err(VulkanError::InvalidImageFormat);
    }
    if config.mip_levels == 0 || config.array_layers == 0 {
        return Err(VulkanError::InvalidImageSubresourceCount);
    }
    if config.usage.is_empty() {
        return Err(VulkanError::InvalidImageUsage);
    }
    if !matches!(
        config.samples,
        vk::SampleCountFlags::TYPE_1
            | vk::SampleCountFlags::TYPE_2
            | vk::SampleCountFlags::TYPE_4
            | vk::SampleCountFlags::TYPE_8
            | vk::SampleCountFlags::TYPE_16
            | vk::SampleCountFlags::TYPE_32
            | vk::SampleCountFlags::TYPE_64
    ) {
        return Err(VulkanError::InvalidImageSampleCount);
    }
    if !matches!(
        config.tiling,
        vk::ImageTiling::OPTIMAL | vk::ImageTiling::LINEAR
    ) {
        return Err(VulkanError::InvalidImageTiling);
    }

    let max_dimension = match config.image_type {
        vk::ImageType::TYPE_1D => {
            if config.extent.height != 1 || config.extent.depth != 1 {
                return Err(VulkanError::InvalidImageExtent);
            }
            config.extent.width
        }
        vk::ImageType::TYPE_2D => {
            if config.extent.depth != 1 {
                return Err(VulkanError::InvalidImageExtent);
            }
            config.extent.width.max(config.extent.height)
        }
        vk::ImageType::TYPE_3D => {
            if config.array_layers != 1 {
                return Err(VulkanError::InvalidImageSubresourceCount);
            }
            config
                .extent
                .width
                .max(config.extent.height)
                .max(config.extent.depth)
        }
        _ => return Err(VulkanError::InvalidImageType),
    };
    let max_mip_levels = u32::BITS - max_dimension.leading_zeros();
    if config.mip_levels > max_mip_levels {
        return Err(VulkanError::InvalidImageMipLevels);
    }
    if config.samples != vk::SampleCountFlags::TYPE_1
        && (config.image_type != vk::ImageType::TYPE_2D
            || config.mip_levels != 1
            || config.tiling != vk::ImageTiling::OPTIMAL)
    {
        return Err(VulkanError::InvalidImageSampleCount);
    }
    if config.flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE)
        && (config.image_type != vk::ImageType::TYPE_2D
            || config.extent.width != config.extent.height
            || config.array_layers < 6)
    {
        return Err(VulkanError::InvalidImageExtent);
    }
    Ok(())
}

fn validate_image_view_config(
    image: ImageConfig,
    view: ImageViewConfig,
) -> Result<(), VulkanError> {
    if view.aspect.is_empty() || view.level_count == 0 || view.layer_count == 0 {
        return Err(VulkanError::InvalidImageViewRange);
    }
    let mip_end = view
        .base_mip_level
        .checked_add(view.level_count)
        .ok_or(VulkanError::InvalidImageViewRange)?;
    let layer_end = view
        .base_array_layer
        .checked_add(view.layer_count)
        .ok_or(VulkanError::InvalidImageViewRange)?;
    if mip_end > image.mip_levels || layer_end > image.array_layers {
        return Err(VulkanError::InvalidImageViewRange);
    }

    let compatible_type = match view.view_type {
        vk::ImageViewType::TYPE_1D => {
            image.image_type == vk::ImageType::TYPE_1D && view.layer_count == 1
        }
        vk::ImageViewType::TYPE_1D_ARRAY => image.image_type == vk::ImageType::TYPE_1D,
        vk::ImageViewType::TYPE_2D => {
            image.image_type == vk::ImageType::TYPE_2D && view.layer_count == 1
        }
        vk::ImageViewType::TYPE_2D_ARRAY => image.image_type == vk::ImageType::TYPE_2D,
        vk::ImageViewType::TYPE_3D => {
            image.image_type == vk::ImageType::TYPE_3D && view.layer_count == 1
        }
        vk::ImageViewType::CUBE => {
            image.image_type == vk::ImageType::TYPE_2D
                && image.flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                && view.layer_count == 6
                && view.base_array_layer.is_multiple_of(6)
        }
        vk::ImageViewType::CUBE_ARRAY => {
            image.image_type == vk::ImageType::TYPE_2D
                && image.flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                && view.layer_count.is_multiple_of(6)
                && view.base_array_layer.is_multiple_of(6)
        }
        _ => false,
    };
    if !compatible_type {
        return Err(VulkanError::IncompatibleImageViewType);
    }
    Ok(())
}

fn select_image_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    compatible_bits: u32,
) -> Option<u32> {
    (0..properties.memory_type_count)
        .filter(|index| compatible_bits & (1 << index) != 0)
        .max_by_key(|index| {
            let flags = properties.memory_types[*index as usize].property_flags;
            (
                flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
                std::cmp::Reverse(*index),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::{
        ImageConfig, ImageViewConfig, select_image_memory_type, validate_image_config,
        validate_image_view_config,
    };
    use ash::vk;

    fn image_config() -> ImageConfig {
        ImageConfig {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D {
                width: 256,
                height: 128,
                depth: 1,
            },
            format: vk::Format::R8G8B8A8_UNORM,
            mip_levels: 9,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            flags: vk::ImageCreateFlags::empty(),
        }
    }

    #[test]
    fn accepts_valid_two_dimensional_image() {
        assert!(validate_image_config(image_config()).is_ok());
    }

    #[test]
    fn rejects_zero_extent_and_undefined_format() {
        let mut config = image_config();
        config.extent.width = 0;
        assert!(validate_image_config(config).is_err());
        let mut config = image_config();
        config.format = vk::Format::UNDEFINED;
        assert!(validate_image_config(config).is_err());
    }

    #[test]
    fn rejects_mip_count_larger_than_extent_allows() {
        let mut config = image_config();
        config.mip_levels = 10;
        assert!(validate_image_config(config).is_err());
    }

    #[test]
    fn rejects_invalid_three_dimensional_array_layers() {
        let mut config = image_config();
        config.image_type = vk::ImageType::TYPE_3D;
        config.extent.depth = 8;
        config.array_layers = 2;
        assert!(validate_image_config(config).is_err());
    }

    #[test]
    fn multisampled_images_require_single_mip_two_dimensional_optimal_tiling() {
        let mut config = image_config();
        config.samples = vk::SampleCountFlags::TYPE_4;
        config.mip_levels = 1;
        assert!(validate_image_config(config).is_ok());

        config.image_type = vk::ImageType::TYPE_3D;
        config.extent.depth = 2;
        assert!(validate_image_config(config).is_err());

        config.image_type = vk::ImageType::TYPE_2D;
        config.extent.depth = 1;
        config.samples = vk::SampleCountFlags::from_raw(128);
        assert!(validate_image_config(config).is_err());
    }

    #[test]
    fn cube_compatible_images_require_square_two_dimensional_extent_and_six_layers() {
        let mut config = image_config();
        config.flags = vk::ImageCreateFlags::CUBE_COMPATIBLE;
        config.extent.height = config.extent.width;
        config.array_layers = 6;
        assert!(validate_image_config(config).is_ok());

        config.extent.height = 64;
        assert!(validate_image_config(config).is_err());
    }

    #[test]
    fn validates_view_subresource_bounds() {
        let image = image_config();
        let valid = ImageViewConfig {
            level_count: 9,
            ..ImageViewConfig::default()
        };
        assert!(validate_image_view_config(image, valid).is_ok());
        let invalid = ImageViewConfig {
            base_mip_level: 8,
            level_count: 2,
            ..ImageViewConfig::default()
        };
        assert!(validate_image_view_config(image, invalid).is_err());
    }

    #[test]
    fn rejects_cube_view_without_cube_compatible_image() {
        let image = ImageConfig {
            array_layers: 6,
            ..image_config()
        };
        let view = ImageViewConfig {
            view_type: vk::ImageViewType::CUBE,
            layer_count: 6,
            ..ImageViewConfig::default()
        };
        assert!(validate_image_view_config(image, view).is_err());
    }

    #[test]
    fn prefers_device_local_memory_and_falls_back_to_compatible_memory() {
        let mut properties = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 3,
            ..Default::default()
        };
        properties.memory_types[0].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE;
        properties.memory_types[1].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        properties.memory_types[2].property_flags =
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE;
        assert_eq!(select_image_memory_type(&properties, 0b110), Some(1));
        assert_eq!(select_image_memory_type(&properties, 0b001), Some(0));
    }
}
